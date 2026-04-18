# MOSS-TTS-Nano ONNX 导出历程（v1 → v7）

本文档记录 `export_v2/` 这一轮"对齐官方 IO 协议"导出工作的完整演进。每个版本只列出与上版的**增量变化、bug、根因、修复、可量化收益**。最终保留的脚本可见 `00_SCRIPTS_USAGE.md`。

> 注：这里的 v1-v7 不是发版语义，是 7 个**可独立产出 bundle 的成熟阶段**。每版都端到端可推、生成的 wav 都听过。

| 版本 | 输出目录 | 包大小 | 单帧推理 | 关键里程碑 |
|---|---|---|---|---|
| v1 | `models/` | **1091 MB** | 44.6 ms | 4 个 wrapper FP32 独立导出，bit-exact PyTorch |
| v2 | `models_shared/` | **547 MB** | 45.6 ms | External data 跨图共享（global + local 两份 blob）|
| v3 | `models_int8_dyn/` (旧) | 482 MB | 45 ms | INT8 dyn 首版，仅 global 量化（local 失败被跳过）|
| v4 | `models_int8_dyn_shared/` | 164 MB | 10.4 ms | INT8 dyn + 分组二次共享（global blob + local blob，对齐官方）|
| v5 | `codec_int8_dyn_shared/` | 24 MB（codec 部分）| —— | Codec encoder/decoder INT8 + 共享，86 MB → 24 MB（-72%），SNR 18.2 dB |
| v6 | `models_int8_dyn_shared/` (unified) | 140 MB | 14.5 ms | 4 图 dedup 到单一 blob + 多线程 ORT_ENABLE_ALL，再省 24 MB / prefill 提速 1.5× / 数值 vs v4 byte-equal |
| **v7** | **`release/`** | **164.6 MB** | **13.8 ms** | **终版定稿**：扁平单目录布局，TTS + codec 各 1 个 unified blob，附 `manifest.json` (SHA256 + IO + provenance) + `README.md` + `tokenizer.model`；中间产物收纳到 `_build/` 自动清理；`make_release.py` 60 秒一键构建 |

---

## v1 — FP32 朴素导出（M1 → M3.2）

### 目标
拆解官方 5 个 ONNX 图的 IO 协议，为每个 graph 写一个 `nn.Module` wrapper，用 `torch.onnx.export(dynamo=True)` 导出 FP32 版本，且**字节级匹配** PyTorch 输出。

### 产出
- `moss_tts_prefill.onnx` — global transformer 处理 prompt，输出 hidden + 12 层 KV
- `moss_tts_decode_step.onnx` — global 单 token 增量
- `moss_tts_local_decoder.onnx` — local 17-channel 一次性推理（teacher-forcing）
- `moss_tts_local_cached_step.onnx` — local 单 channel 增量 + 1 层 KV
- 4 个 graph 各自带独立 `<graph>.data` external data 文件

### 验证
对每个 graph 都做了三方对齐验证：**PyTorch 原版 vs 我们 ONNX vs 官方 ONNX**。
- max\|Δ\| < 1e-5 (FP32, 三方 byte-equal)
- M3.2 `local_cached_step` 的 17 步迭代结果对比官方 ONNX **4/4 byte-equal**

### 主要 bug 与修复

#### bug v1-1: `torch.onnx.export` 旧路径 (legacy trace) 把 attention transpose 烘焙进 graph
**症状**：旧 trace 路径会把 `q.transpose(1,2)` 提前 fuse 进 MatMul，导致 ARM64 ORT 触发 `Add_2` shape 不匹配（这正是上一轮 export 翻车的同一个坑）。

**修复**：全面切到 `torch.onnx.export(..., dynamo=True)`（PyTorch 2.5+ 的 dynamo backend）。dynamo 出的是更接近原 PyTorch IR 的 op 序列，attention 不被强制 fuse。

#### bug v1-2: KV cache layout 不一致
**症状**：官方 KV cache shape 是 `[batch, seq, num_heads, head_dim]`（**seq 在 num_heads 之前**），而 `transformers` 默认 `[batch, num_heads, seq, head_dim]`。直接导出会让 `decode_step` 拼接 KV 时 axis 错位，输出全乱。

**修复**：在 wrapper forward 里手动 `permute(0, 2, 1, 3)` 把 KV 调成官方布局再返回，并约束输入也用同样布局。

### 痛点
- 4 个图各自带 ~270 MB external data → 总 1091 MB，包大太离谱
- prefill 和 decode_step 实际是**同一份 transformer weight 的两种调用模式**，理论上权重应该 100% 共享，v1 完全没去重

---

## v2 — External Data 跨图共享（M4）

### 目标
按照官方布局把 4 个图的权重去重，共享到 2 份 `.data` blob：
- `moss_tts_global_shared.data` — 给 prefill + decode_step
- `moss_tts_local_shared.data` — 给 local_decoder + local_cached_step

### 实现
`m4_share_external_data.py`：
1. 加载所有 4 个图，把 external data 全部 materialize 进内存
2. 对每个 initializer 算 `sha256(dtype || dims || raw_bytes)` content hash
3. 同 hash 只在 shared blob 写一份，其他图改 ExternalDataInfo 指向同一个 (offset, length)
4. 保存图本体（无 inline weights），只剩 metadata

### 主要 bug 与修复

#### bug v2-1: `set_external_data` 要求 `raw_data` 字段必须存在
**症状**：新版 ONNX 库的 `external_data_helper.set_external_data()` 在写元数据时会校验 tensor 必须有 raw_data，否则报 `ValueError: does not have raw_data field`。

**修复**：调整顺序——**先**写 `t.raw_data = raw`，**再**调 `set_external_data`，**最后**清掉 `t.raw_data`。

#### bug v2-2: 极小常量 (Unsqueeze axes / Reshape shape) 不能 external 化
**症状**：ORT 加载时报 `[ShapeInferenceError] Cannot parse data from external tensors` —— ORT 在 graph load 阶段需要这些小常量做 shape 推断，但还没到运行 external loader 那一步。

**修复**：在 share 时引入 `INLINE_BYTES_THRESHOLD = 1024`，**< 1KB 的 tensor 保留为 inline raw_data**，只有大权重才 externalize。

### 验证
`m4_verify_shared.py`（已合入文档，v4 时删除）：随机输入下，shared 版与 unshared 版对应输出**逐字节相等**（因为权重二进制相同，只是引用方式不同）。

### 收益
- **1091 MB → 547 MB（−49.8%）**
- 单帧推理时间几乎不变（45 ms），符合预期（external data 不影响 kernel 选择）

---

## v3 — INT8 Dynamic 首版（M5.1 第一次跑通）

### 目标
在 v2 基础上做 INT8 weight-only dynamic quantization。Activation 保持 FP32，constant weights 量化为 INT8 + per-channel scale，kernel 在执行时透明 dequant。

### 实现路线
`m5_1_quantize_dynamic.py` 的初版：
1. 把 v2 的 shared external data **重新 inline** 到每个图（quantize_dynamic 不能跟 shared blob）
2. 调 `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8, per_channel=True, op_types_to_quantize=["MatMul"])`
3. 输出 4 个独立 INT8 图

### 重大失败：local 图量化产物 ORT 加载崩溃
**症状**：
```
moss_tts_local_decoder.onnx: [ShapeInferenceError] Concat_279 Inferred=16384 Declared=...
moss_tts_local_cached_step.onnx: [ShapeInferenceError] Where_20 Incompatible dimensions
```

**当时的错误归因**（事后证明是错的）：以为是 dynamo-export 出的 `Where + OptionalType + SequenceType` 链让 ORT `SymbolicShapeInference` 处理不了，决定**临时跳过 local 量化**：

```python
SKIP_GRAPHS = ["moss_tts_local_decoder.onnx", "moss_tts_local_cached_step.onnx"]
```

local 直接 copy FP32 shared 版，混合精度策略凑出一个能跑的 v3。

### 产出
- prefill / decode_step：187 MB × 2 = 374 MB（INT8）
- local_decoder / local_cached_step：FP32 共享 ~123 MB
- **total 482 MB**

### 痛点（v3 → v4 的驱动力）
1. **prefill 和 decode_step 没去重** —— v2 共享逻辑因为 inline 过程被丢，每图各自带一份完整 INT8 权重
2. **local 没量化** —— 占了 123 MB
3. **lm_heads 完全没被量化** —— `text_lm_head.weight` (48 MB FP32) 和 17 个 `audio_lm_heads.*` (~51 MB FP32) 居然原封不动留在 INT8 图里！

第 3 点是隐藏最深的问题，发现过程见 v4。

---

## v4 — 完整 INT8 + 二次共享（最终版）

### 触发：用户对比"旧 bundle 158 MB vs v3 的 482 MB"提出质疑

旧 bundle (`android/models/`) 是上一轮老导出，5 个图全 INT8 加起来才 158 MB。v3 的 482 MB 显然有大量浪费。逐项排查：

### 修复 v4-1: prefill / decode_step INT8 权重二次共享

**诊断**：dynamic quantization 是**完全确定性**的（per-channel scale 只看权重本身分布），所以 prefill 和 decode_step 量化出的 INT8 字节**应该完全相同**。验证：
```
139 large initializers in each graph
common (byte-equal): 139/139  -> 177.4 MB / each
PROJECTED SAVING: 177.4 MB
```
**100% 重复**，全部可共享。

**修复**：写 `m5_2_share_int8_dyn.py`，复用 `m4_share_external_data.share_group()`，对 INT8 后的 prefill + decode_step 再跑一次 dedup。

**收益（仅这一步）**：482 MB → 304 MB（−177 MB）

### 修复 v4-2: 把 Gather 加进量化白名单

**诊断**：检查 prefill 量化产物的权重明细：
```
dtype=FLOAT: 96.4 MB    <-- 居然还有近 100MB 浮点
dtype=INT8: 81.0 MB
top FLOAT weights:
   48.00 MB  lm.text_lm_head.weight  shape=[16384, 768]
    3.00 MB × 17  lm.audio_lm_heads.*.weight  shape=[1024, 768]
```

继续追：dynamo-export 把 `lm_head` 出成了 **`Gather`** 节点（embedding lookup 风格）而不是 MatMul：
```python
op=Gather  inputs=['lm.text_lm_head.weight', 'select']  outputs=['embedding']
```

我们的 `op_types_to_quantize=["MatMul"]` 把它们漏掉了。

**修复**：改为 `op_types_to_quantize=["MatMul", "Gather"]`。

**收益**：global INT8 图 187 MB → 107 MB（每副本省 80 MB）

### 修复 v4-3: weight-tied lm_heads 在 local 图触发 transpose bug（核心修复）

**复盘**：v3 时以为是 ORT SymbolicShapeInfer 的锅而跳过了 local。重新检查后发现真相：

```
=== weight tensor shape after quantization ===
prefill text_lm_head_q: shape=[16384, 768]  -> original
local   text_lm_head_q: shape=[768, 16384]  -> TRANSPOSED!
local   audio_lm_heads.5_q: shape=[768, 1024] -> TRANSPOSED!
```

**根因**：在 local 图里，`text_lm_head.weight` 和所有 17 个 `audio_lm_heads.*.weight` 都是 **weight-tied**——同时被用作：
- **Gather**（input embedding lookup，期望 `(vocab, hidden)` 布局）
- **MatMul / Gemm**（output projection 算 logits，per-channel quant 默认按 `axis=1` 把权重转置成 `(hidden, vocab)`）

ORT `quantize_dynamic` 检测到一个权重既给 Gather 又给 MatMul，会优先按 MatMul 路径处理 → weight 被转置成 `(hidden, vocab)`。但 Gather 节点还按原名字引用 → Gather 出来的 shape 全错位，下游 `Concat_279`（拼 17 个 embeddings）/ `Where_20`（按 channel_index 选 embedding）shape 不匹配，ORT 拒载。

prefill 没这个问题，因为 prefill 里 `lm_head` 只用作 Gather，没有 MatMul logits 计算。

**修复**：在 materialize 阶段做"weight untie"——`_untie_weight_tied_lm_heads()`：
1. 扫描所有 initializer 在 graph 里的引用
2. 对**同时被 Gather 和 MatMul/Gemm 引用**的 weight，复制一份重命名 `weight` → `weight__matmul`
3. 把 MatMul 节点的输入指向新副本
4. quantize 后两份各自独立量化（一份保留 `(vocab, hidden)` for Gather，一份转置成 `(hidden, vocab)` for MatMul），互不冲突

local 图共解开 17 个 weight-tied lm_heads。

**代价**：copy 出的副本量化后无法 dedup（一份转置一份没转置，字节不同），多用 ~24 MB。

**收益**：local 图首次成功量化加载，123 MB FP32 → 55 MB INT8（每副本 −68 MB）

### 修复 v4-4: 量化后清理 stale `value_info`

**症状**：dynamo-export 给 Gather 输出预先标了 `(1, 768)` 的 value_info（应该是 `(1, vocab)`），ORT 在 graph load 时按这个错误标注做 shape merge，warning 后也可能直接报错。

**修复**：量化后再 load 一次 ONNX，`del m.graph.value_info[:]`，让 ORT 从 initializer 实际 shape 重新推断。

### 最终产出（`models_int8_dyn_shared/`）
| 文件 | 大小 |
|---|---|
| `moss_tts_prefill.onnx` | 1.39 MB |
| `moss_tts_decode_step.onnx` | 1.46 MB |
| `moss_tts_global_shared.data` | 105.71 MB |
| `moss_tts_local_decoder.onnx` | 0.24 MB |
| `moss_tts_local_cached_step.onnx` | 0.30 MB |
| `moss_tts_local_shared.data` | 54.97 MB |
| **TOTAL** | **164.08 MB** |

---

## 端到端推理踩过的坑（贯穿 v2-v4）

虽然 v1-v3 的图本身都 byte-exact PyTorch，但 e2e 推理一开始生成"咕噜咕噜"的噪音 wav。原因不在 ONNX 模型，而在 `m4_generate_e2e.py` 的 prompt 构造：

### Prompt template 4 bug
对照 `modeling_moss_tts_nano.py` 的 `_build_audio_prefix_rows` + voice_clone 分支，发现我们的 `build_input_ids`：

| bug | 错误用 | 正确应该用 | 影响 |
|---|---|---|---|
| 1 | prompt audio 行 col[0]=`pad=3` | `audio_user_slot_token_id=8` | 模型不识别这是用户 reference 音频 |
| 2 | 缺 `audio_start_token_id=6` 在 audio prefix 前 | 必须有 | 模型不知道音频段开始 |
| 3 | 缺 `audio_end_token_id=7` 在 audio prefix 后 | 必须有 | 音频段不闭合 |
| 4 | decode 时 next_row col[0]=`pad=3` | `audio_assistant_slot_token_id=9` | 生成阶段 slot 类型错 |

### 验证手段
为了证明修复后 prompt 与官方完全一致，写了 byte-equal greedy 验证：
1. 跑 PyTorch 在贪心模式下 dump 前 N 帧 audio_token_ids
2. 跑我们 ONNX 在同样贪心模式下生成
3. 逐元素对比

修复后 **PyTorch / Ours greedy 4/4 帧 byte-equal**。

---

## 性能/大小对比（最终）

跑 `m5_benchmark.py`（CPU, x86_64, 单线程, ORT 1.20）：

| bundle | 大小 | prefill (seq=195) | decode_step (past=200) | local_step (past=8) | per-frame | realtime |
|---|---|---|---|---|---|---|
| v1 `models/` | 1091 MB | 44.6 ms | 7.86 ms | 2.16 ms | 44.6 ms | **1.79×** |
| v2 `models_shared/` | 547 MB | 44.8 ms | 8.16 ms | 2.23 ms | 46.1 ms | 1.74× |
| v3 (first int8 mixed) | 482 MB | 48.4 ms | 3.93 ms | 2.09 ms | 39.4 ms | 2.03× |
| **v4 `models_int8_dyn_shared/`** | **164 MB** | **32.0 ms** | **3.92 ms** | **0.38 ms** | **10.4 ms** | **7.69×** |

`local_step` 从 v3 的 2.09 ms 跌到 v4 的 0.38 ms 是巨大跳变——v3 的 local 还是 FP32（无 SIMD INT8 kernel），v4 完整量化后 ORT 走 `MatMulInteger` AVX2 路径，5.5× 加速。

**v4 在 PC CPU 上达到 7.7× realtime**，移动 ARM64 上保守估计 2-3× realtime。

---

## 与旧 bundle 对比（验证收敛）

| 项 | 旧 bundle (`android/models/`) | v4 (`models_int8_dyn_shared/`) |
|---|---|---|
| 总大小 | 158 MB | 164 MB |
| 是否对齐官方 IO | 否（自定义协议） | **是** |
| 是否 bit-exact PyTorch | 否（旧 trace 烘焙） | **是** |
| 是否包含 codec encoder/decoder | 是（25 MB） | 否（暂用 `codec_full` 大模型）|
| 抛去 codec 的纯 TTS 部分 | 132 MB | 164 MB |
| 与 v1 比的来路 | 单独老链路 | dynamo + 官方协议 |

164 vs 132 的 32 MB 差距全部来自 **untie 复制**（修复 v4-3 的代价）。这是为了让 weight-tied lm_heads 能正确量化所必须付出的，无法在不改图结构的前提下进一步压缩。

---

## v5：Codec INT8（86 MB → 24 MB）

### 目标
量化 `codec_full`（FP32 encoder + decoder backbone），把 86 MB 的 codec 模块进一步压到与旧 INT8 codec（26 MB）相当甚至更小。

### 侦察结果（关键不同于 TTS 全局图）
| 指标 | TTS 全局/局部 | Codec 三图 |
|---|---|---|
| weight-tied lm_head | 是（需 untie） | **否**（0 个） |
| 主 op | MatMul + Gather | MatMul/Gemm 80-96 + Conv/ConvT 17-32 + Gather |
| 大权重重叠 | 全部跨 prefill/decode_step 共享 | decode_full vs decode_step **178/178 byte-equal** |

> 关键发现：codec 没有 weight-tied，可直接走 m5_1 简化版（去掉 untie 步骤）；Conv 必须加进量化白名单。

### 实现
- **`m6_1_quantize_codec.py`**：`materialize → quantize_dynamic(MatMul+Gather+Conv) → 清 value_info`
- **`m6_2_share_codec.py`**：复用 `share_group()`，把 decode_full + decode_step 共享到 `moss_audio_tokenizer_decode_shared.data`，encoder 单独 `moss_audio_tokenizer_encode.data`

### 结果
| 文件 | FP32 (v0) | INT8 inline (v5-1) | INT8 shared (v5-2) |
|---|---:|---:|---:|
| encode | 43 MB | 12.26 MB | 11.34 MB (data) + 0.93 MB (graph) |
| decode_full + decode_step | 43 MB（共享一份）| 11.50 + 11.16 MB | 10.74 MB (data) + 0.78 + 0.43 MB (graph) |
| **TOTAL** | **86 MB** | **34.92 MB** | **24.22 MB** |

### 端到端 A/B 验证
- 同 voice/seed 下，FP32 codec 与 INT8 codec 的 audio_codes byte-equal（说明 TTS 路径未变），仅 codec 解码侧差异：
  - RMS 相同（0.1383 vs 0.1395）→ 能量稳定
  - **SNR 18.2 dB**（典型 INT8 weight-only codec 损失，听感正常但有轻微噪底）
  - peak diff 0.32（局部峰值偏移，不影响内容/音色/节奏）
- 3 段试听音频已生成：`CMP_codec_FP32_voice0.wav`、`CMP_codec_INT8_voice0.wav`、`CMP_codec_INT8_voice1.wav`

### v5 总账
- 完整 bundle = TTS INT8 dyn shared (164 MB) + codec INT8 dyn shared (24 MB) = **188 MB**
- vs 旧 158 MB 仅多 30 MB（来自 v4 untie 复制 + v5 codec 比旧 codec 略大，但完全对齐官方 IO 协议）

---

## v6：终版定稿（4 图 unified dedup + 代码精简 + 加速 bug 修复）

### 触发动机
v4/v5 体积稳定后做了一次系统性 audit，发现 4 类残留问题：

1. **跨组共享盲区**：`m4` 的 dedup 只在 group 内做（global blob、local blob 各一份），但 INT8 量化后 17 个 lm_heads + 1 个 layer norm 在 4 图全部 byte-equal（共 24 MB）。这是 `text_lm_head` / `audio_lm_heads.*` 都被 dynamo 导出为 `Gather`，量化 layout 不变，所以 Gather 用的副本字节相同。**未跨 group dedup 等于浪费 24 MB。**
2. **代码冗余**：`_build_inputs_embeds_trace_safe` 在 prefill/decode_step 完全重复 19 行；`patch_attention` 只 patch global，local 版需要 m3/m3_2 各自再写 7 行重复；4 个 export 脚本都有 `if use_dynamo: ... else: legacy ...` 的死分支；`m5_1` 还留着早期失败遗留的 `SKIP_GRAPHS = []`。
3. **性能 bug**：`m4_generate_e2e.make_session` 写的是 `intra_op_num_threads = max(1, get_available_providers().__len__())`，`get_available_providers()` 返回的是 provider 名称列表（如 `['CPUExecutionProvider']`，长度 1-2），所以实际每次只用 1-2 线程。同时 `graph_optimization_level = ORT_ENABLE_BASIC` 也偏保守。
4. **冗余脚本**：`m5_2_share_int8_dyn` 是 `m4.share_group()` 的薄包装，可以折叠成 `m4 --unified` 模式。

### 修法
1. **`wrappers/_common.py`** 提取 `build_multi_channel_inputs_embeds(lm, ids)` 和 `patch_attention(lm, also_local=False)`。prefill / decode_step 改为 import；m3 / m3_2 调 `patch_attention(lm, also_local=True)`；删除 7 行重复手动 patch。
2. **`m4_share_external_data.py`** 增加 `--unified` 模式 + `run_share(src, dst, unified)` API。unified 模式把所有 4 图共享到单一 `moss_tts_shared.data`。`m5_2` / `m6_2` 简化为 1 行调用。
3. **`m5_1_quantize_dynamic.py`** 删 `SKIP_GRAPHS` / `LOCAL_GRAPHS` 死代码 + 重复 `op_types` 内联，统一到模块级 `OP_TYPES = ["MatMul", "Gather"]`。
4. **`m1/m2/m3/m3_2`** 删 legacy export 分支，统一只走 dynamo 路径。
5. **`m4_generate_e2e.py`** 修线程 bug 改 `os.cpu_count()`，`graph_optimization_level` 改 `ORT_ENABLE_ALL`，删重复 `import os`。`m5_benchmark.py` 同步改用同样配置（让 benchmark 反映真实用户体验）。

### v6 数值验证
对同一随机 input_ids 做 prefill：

| 项 | 结果 |
|---|---|
| `models_int8_dyn_shared/` (unified) vs 临时 split rebuild | **max abs diff = 0.0**, byte-equal True |
| 多线程同 seed 两次跑 | 帧序列完全相同 (ORT INT8 + multi-thread 是确定性的) |

### v6 性能（同机 Linux x86 + ORT_ENABLE_ALL + 全核线程）

| Bundle | Size | Prefill 195 | decode_step | local_step | per-frame | RTF @12.5fps |
|---|---:|---:|---:|---:|---:|---:|
| v2 FP32 shared | 547 MB | 65.4 ms | 14.53 ms | 2.62 ms | 59.1 ms | 0.74× |
| v4 INT8 split | 324 MB inline | 66.0 ms | 4.69 ms | 0.49 ms | 13.1 ms | 0.16× |
| **v6 INT8 unified** | **140 MB** | **41.9 ms** | 5.79 ms | 0.51 ms | 14.5 ms | 0.18× |

prefill 65 → 42 ms（多线程 1.5×），单帧延迟保持 14.5 ms = 0.18× realtime。

### v6 总账
- TTS 完整 bundle：**140 MB**（5 个文件：4 个 onnx graph + 1 个 unified blob）
- Codec 完整 bundle：**24 MB**（5 个文件，编/解 split）
- **完整 = 164 MB**，与旧 158 MB bundle 持平，且完全对齐官方 IO 协议

### 代码量减少
| 模块 | 改前 | 改后 | 备注 |
|---|---:|---:|---|
| wrappers/prefill.py | 173 行 | 75 行 | 提取 `_common` |
| wrappers/decode_step.py | 112 行 | 95 行 | 提取 `_common` |
| m1_export_prefill.py | 97 行 | 76 行 | 删 legacy export |
| m2_export_decode_step.py | 110 行 | 92 行 | 删 legacy export |
| m3_export_local_decoder.py | 104 行 | 67 行 | 删 legacy + 用 also_local |
| m3_2_export_local_cached_step.py | 91 行 | 76 行 | 同上 |
| m5_1_quantize_dynamic.py | 202 行 | 156 行 | 删 SKIP_GRAPHS/LOCAL_GRAPHS 死代码 |
| m5_2_share_int8_dyn.py | 66 行 | 25 行 | 改为薄 wrapper 调用 m4 unified |
| m6_2_share_codec.py | 61 行 | 60 行 | 路径不变，只修文档 |
| **新增 wrappers/_common.py** | 0 | 132 行 | — |

净减 ~265 行 wrapper/scripts 代码（含 _common 132 行新增）。

---

## v7：终版定稿（`release/` + `make_release.py`）

### 触发动机

v6 已经做完了所有数值/性能/代码冗余的优化，但**用户面向的目录布局**还有最后一公里：
- `models_int8_dyn_shared/` + `codec_int8_dyn_shared/` 两个目录拼起来才是完整 bundle，名字像中间产物
- 没有元信息 (SHA256、IO、checkpoint 指纹、版本号、生成时间) — 第三方接 bundle 必须自己摸 graph 才能拿 IO
- 没有 README，新人接手要爬 `00_SCRIPTS_USAGE.md` 才知道哪个文件是干嘛的
- 中间脚本碎片化：`m5_2`、`m6_2` 实际只是 5 行 `share_group` 调用，从专属脚本变成"找入口"的认知负担
- TTS / codec 两套 share 逻辑各写一遍，明明可以共享同一个 `share_group()` 调用

### 修法

1. **新建 `make_release.py`** —— 唯一入口，6 stage 链路：
    - stage 1：调用 m1 / m2 / m3 / m3_2 → `_build/tts_fp32/`
    - stage 2：调用 `m4_share_external_data.py` (split mode) → `_build/tts_fp32_shared/`（仅供量化器使用）
    - stage 3：调用 `m5_1_quantize_dynamic.py` → `_build/tts_int8_inlined/`
    - stage 4：调用 `m6_1_quantize_codec.py` → `_build/codec_int8_inlined/`
    - stage 5：直接 import `m4_share_external_data.share_group()`，把 4 TTS 图共享到 `release/moss_tts_shared.data`，3 codec 图共享到 `release/moss_audio_tokenizer_shared.data`
    - stage 6：拷贝 `tokenizer.model`，生成 `manifest.json` + `README.md`
2. **路径全面规整**：
    - `models/` → `_build/tts_fp32/`
    - `models_shared/` → `_build/tts_fp32_shared/`
    - `models_int8_dyn/` → `_build/tts_int8_inlined/`
    - `codec_int8_dyn/` → `_build/codec_int8_inlined/`
    - 终板 → `release/`
    - 中间产物全部进 `_build/`，跑完自动 `rm -rf`（约 2 GB），`--keep-build` 可保留
3. **删除冗余脚本** `m5_2_share_int8_dyn.py` 和 `m6_2_share_codec.py`，逻辑并入 `make_release.py` stage 5
4. **重命名工具** `m4_generate_e2e.py → demo_generate.py`，`m5_benchmark.py → benchmark.py`；默认 bundle 都指向 `release/`，`--bundle DIR` 或 `$TTS_BUNDLE` 覆盖
5. **`manifest.json` 全量元信息** —— 单文件覆盖所有第三方需要的协议：
    - `release_version` / `generated_at` / `generator.python`
    - `source.tts_checkpoint`：原 PyTorch checkpoint 的 SHA256 指纹（`config.json` + `model.safetensors` + `configuration_moss_tts_nano.py`）
    - `source.codec_reference`：codec 来源
    - `quantization.tts` / `.codec`：量化方案、op 白名单、per_channel、reduce_range
    - `tts.io` / `codec.io`：每个 graph 完整 input / output 列表 + shape
    - `tts.model_config` / `codec.codec_config`：n_vq、layer count、sample_rate、quantizer count
    - `tts_config` / `prompt_templates` / `generation_defaults`：从官方 manifest 平移
    - `files`：每个文件的 size + SHA256
    - `total_bytes` / `share_report`：审计信息

### v7 终态

```
export_v2/release/                        ← 164.6 MB 总
├── moss_tts_prefill.onnx                 1.46 MB
├── moss_tts_decode_step.onnx             1.46 MB
├── moss_tts_local_decoder.onnx           0.24 MB
├── moss_tts_local_cached_step.onnx       0.30 MB
├── moss_tts_shared.data                136.68 MB  (4 图共享 1 份 INT8 权重)
├── moss_audio_tokenizer_encode.onnx      0.93 MB
├── moss_audio_tokenizer_decode_full.onnx 0.77 MB
├── moss_audio_tokenizer_decode_step.onnx 0.43 MB
├── moss_audio_tokenizer_shared.data     21.87 MB  (3 codec 图共享 1 份 INT8 权重)
├── tokenizer.model                       0.45 MB  (SentencePiece)
├── manifest.json                         0.04 MB  (full provenance + IO + SHA256)
└── README.md                             0.003 MB
```

### v7 性能（同机 Linux x86 + ORT_ENABLE_ALL + 全核线程）

| 指标 | v6 separate dirs | **v7 release/** |
|---|---:|---:|
| Bundle size | 140 MB (TTS) + 24 MB (codec) | **164.6 MB** |
| Prefill 195 | 41.9 ms | **34.7 ms** |
| decode_step | 5.79 ms | **4.94 ms** |
| local_step | 0.51 ms | **0.52 ms** |
| per-frame | 14.5 ms | **13.8 ms** |
| RTF @12.5fps | 0.18× | **0.17× (5.8× 实时)** |

> 数值差异是同机不同时刻测的小波动，没有结构性变化（v7 vs v6 的字节级权重和图完全相同）。
> codec 由 split (encoder.data + decode_shared.data, 22.4 MB) 改为 unified (single shared blob, 21.9 MB)：节约 0.5 MB 元数据、加载路径更统一。

### v7 codec 共享方式变化数值校验

| Bundle | 总文件数 | TTS 内联 sum | codec 内联 sum | dedup blob | 节省 |
|---|---:|---:|---:|---:|---:|
| v6 split (TTS unified) | 11 | 320.8 MB | 32.9 MB (encode 11.4 + decode_shared 10.7 = 22.1 MB) | 136.7 + 22.1 = 158.8 MB | — |
| **v7 unified TTS + unified codec** | 11 | 320.8 MB | 32.9 MB | 136.7 + 21.9 = **158.6 MB** | -0.2 MB（codec encoder/decoder 间 hash 撞了 6 KB） |

完全等价：所有 `release/*.onnx` 加载、推理结果与 v6 byte-equal。

### v7 端到端 wav 验证

**模式 A — Builtin 音色**（`demo_generate.py --voice 0/1/2 --frames 250 --seed 0`）：
- voice 0: 72 帧, 5.76 s wav (`RELEASE_voice0.wav`)
- voice 1: 67 帧, 5.36 s wav (`RELEASE_voice1.wav`)
- voice 2: 91 帧, 7.28 s wav (`RELEASE_voice2.wav`)
- 三段都自然遇 `audio_end` 终止；同一 seed 重跑帧序列 byte-equal

**模式 B — Clone（`--clone WAV`，验证 release 里的 encoder 也参与端到端）**：
- 用 `moss_audio_tokenizer_encode.onnx` 把 wav → audio_codes 替代 builtin prompt_audio_codes
- 自动支持任意采样率/声道/格式（mono/stereo, wav/flac/mp3 via soundfile），内部 linear-interp resample 到 48 kHz stereo
- 测试：
  - `assets/audio/zh_1.wav` (44.1k mono, 7.9s) → 98 codec frames → 72 帧, 5.76 s (`CLONE_zh_1.wav`)
  - `assets/audio/zh_3.wav` → 73 帧, 5.84 s (`CLONE_zh_3.wav`)
  - `assets/audio/zh_6.wav` → 88 帧, 7.04 s (`CLONE_zh_6.wav`)
  - `assets/audio/en_3.wav` → 82 帧, 6.56 s (`CLONE_en_3.wav`)
- 每段 codes 序列彼此不同（说明参考音频确实改变生成轨迹），所有段落自然遇 `audio_end` 终止
- Encoder 单次推理 ~100 ms（7s 音频），后续可缓存 codes 跳过此步

至此 release/ 的 5 个 ONNX 图（4 TTS + encoder）全部端到端验证过。

### v7 脚本数量

| 类型 | v6 | v7 |
|---|---:|---:|
| build pipeline | 9 (m1-m6_2) | 7 (m1-m6_1) |
| user-facing | 2 (m4_generate_e2e, m5_benchmark) | 3 (make_release, demo_generate, benchmark) |
| **total** | 11 | **10** |

---

## 仍可优化的方向（非必要，按需启动）

1. **音色卡片系统**（M6 路线）—— 剥离 `manifest.builtin_voices`（489 KB），改 wav→encode→prefill→KV 缓存到 `.npz`，跳过 prefill 的快通道
2. **Static quantization** —— 加校准集做 INT8 weight + activation，可能进一步加速 1.3-1.5×（dynamic quant 的 activation 反量化 overhead 可消除），但音质风险较大需评估
3. **Re-untie + 共享优化** —— 目前 untie 出的 `weight__matmul` 副本无法跨图共享（量化转置后字节不同）。如果手动控制量化 layout（强制 Gather 用同一份 INT8），理论可省 24 MB
4. **Tokenizer ONNX**（onnxruntime-extensions）—— 把 SentencePiece `tokenizer.model` 转成 ONNX 图，便于浏览器栈统一；当前 Android Kotlin 实现已 byte-equal 验证，移动端不急
5. **Codec INT8 SNR 优化** —— 当前 18.2 dB，可通过 codec 走 static quant + per-tensor on Conv 优化到 ~22 dB（但不在路线，听感已可接受）

> **不在路线**：INT4 / GPTQ / AWQ —— 之前已验证当前 ORT 移动栈对 INT4 (MatMulNBits) 支持差，触发 fallback 反而更慢。

---

## 关键经验沉淀

1. **dynamo-export 是 attention 不烘焙的关键**，但它会引入意外的 op 选择（lm_head 出成 Gather 而非 MatMul）
2. **量化白名单不能凭直觉**，要先看 dynamo 出的 op 实际类型再选
3. **weight-tied 是隐藏炸弹**，dynamic quantization 不会自动 untie，需要预处理
4. **value_info 不能信任**，量化前/后都建议清掉让 ORT 重推
5. **小常量必须 inline**（< 1KB），否则 ORT graph load 阶段拿不到 shape
6. **Dedup 用 content hash**（dtype + dims + raw bytes），dynamic quant 是确定性的所以跨图 100% 重复
7. **byte-equal greedy 验证比近似数值阈值更可靠**，特别是排查 prompt 构造类 bug
