# 官方 MOSS-TTS-Nano-100M-ONNX 架构剖析

来源：`OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX` @ HuggingFace
对照本地：`MOSS-TTS-Nano-100M/modeling_moss_tts_nano.py` (PyTorch HF 风格)

---

## 1. 五个 ONNX 图与协议总览

| 文件 | 大小 | 角色 | 调用频次 |
|---|---|---|---|
| `moss_tts_prefill.onnx` | 0.28 MB | Global transformer 处理变长 prompt，输出 last hidden + KV cache | 每次推理 1 次 |
| `moss_tts_decode_step.onnx` | 0.28 MB | Global transformer 单 token 增量推理 | 每帧 1 次 (≈ 12.5 次 / s) |
| `moss_tts_local_decoder.onnx` | 0.05 MB | Local decoder，**无 KV cache**（用于训练时校验或简化推理） | 通常不用 |
| `moss_tts_local_cached_step.onnx` | 0.05 MB | Local decoder 单 channel 增量（带 1 层 KV cache），**外部循环 17 次出 1 帧** | 17 × 帧数（备选） |
| `moss_tts_local_fixed_sampled_frame.onnx` | 0.46 MB | **杀手锏**：把 17 次 channel 循环 + 全套采样融合到一个图里。一次 run 出 16-channel frame token + EOS 信号 | 每帧 1 次（推荐） |

权重共享：所有 5 个图都用 `external_data` 形式共享 2 份 `.data` 文件（`moss_tts_global_shared.data` 420 MB / `moss_tts_local_shared.data` 219 MB），所以 graph 文件本身极小。

---

## 2. Prefill / DecodeStep IO 协议（标准 LLM 推理拆分）

**`prefill`** 输入 (2)：
```
input_ids        INT32 [batch, prefill_seq, 17]   # 17 = 1 (text/assistant slot) + 16 (n_vq)
attention_mask   INT32 [batch, prefill_seq]
```
输出 (25)：
```
global_hidden    FLOAT [batch, prefill_seq, 768]
present_key_i    FLOAT [batch, prefill_seq, 12, 64]   # i = 0..11，12 层
present_value_i  FLOAT [batch, prefill_seq, 12, 64]
```

**`decode_step`** 输入 (26)：
```
input_ids            INT32 [batch, step_seq=1, 17]
past_valid_lengths   INT32 [batch]                    # 同 batch 内不等长 prompt 的有效长度
past_key_i           FLOAT [batch, past_seq, 12, 64]
past_value_i         FLOAT [batch, past_seq, 12, 64]
```
输出同 prefill，shape 上 `total_seq = past_seq + step_seq`。

**关键设计**：
- 行宽 `17 = 1 text-slot + 16 audio-codebooks`，与我们的实现一致。
- `past_valid_lengths` 作为单独标量输入而非 `attention_mask`，是为了让 decode_step 的 KV cache 长度可以是固定 buffer（外部 padding）而不必每次重新分配。
- batch 维保留意味着官方支持多请求并发（serving 场景）。我们端侧先按 `batch=1` 走。

---

## 3. Local Decoder 三种 variant

### 3.1 `local_decoder` (基础版)
```
in : global_hidden [1,768], text_token_id [1], audio_prefix_token_ids [1,15]
out: text_logits [1,16384], audio_logits [1,16,1024]
```
一次 run 出**所有 16 个 channel 的 logits**，需要外部传入 15 个已经采样好的 audio token 作为前缀。这是 **teacher-forcing** 形态，主要用于训练校验或离线评估。

### 3.2 `local_cached_step` (单 channel 增量)
```
in : global_hidden [1,768], text_token_id [1], audio_token_id [1],
     channel_index [1], step_type [1], past_valid_lengths [1],
     local_past_key_0 [1, local_past_seq, 12, 64],
     local_past_value_0 [1, local_past_seq, 12, 64]
out: text_logits [1,16384], audio_logits [1,16,1024],
     local_present_key_0 [1, local_total_seq, 12, 64],
     local_present_value_0 [1, local_total_seq, 12, 64]
```
按 `channel_index` 步进，外部循环 17 次（1 text + 16 audio）出 1 帧。`step_type` 区分当前是 text 步还是 audio 步。这是**手动展开**的版本，给需要细粒度调度（例如自定义采样策略）的场景用。

### 3.3 `local_fixed_sampled_frame` (融合采样，推荐)
```
in : global_hidden [batch, 768]
     repetition_seen_mask [batch, 16, 1024]   # 每个 audio channel 已经出现过的 token 的 mask
     assistant_random_u [batch]                # 1 个均匀分布随机数 (text token sample)
     audio_random_u [batch, 16]                # 16 个均匀分布随机数 (audio token sample)
out: should_continue [batch, 1]                # 是否继续生成 (EOS 检测)
     frame_token_ids [batch, 16]               # 一帧的 16 channel 采样结果
```

把所有 17 次 transformer block + softmax + repetition penalty + top-k + top-p + 采样全部塞进一个图，一次 `session.run()` 出一帧。**所有随机性走外部输入**（`*_random_u`），ONNX 内部用 inverse-CDF 转化为采样。常量超参嵌入在 graph 里（来自 meta.json）：
```
text_temperature=1.0, text_top_p=1.0, text_top_k=50,
audio_temperature=0.8, audio_top_p=0.95, audio_top_k=25,
audio_repetition_penalty=1.2
```

---

## 4. 采样融合的 ONNX 实现（最重要的发现）

观察 `local_fixed_sampled_frame` 中重复 16 次的 audio sampling pattern（节点编号来自 `analyze_official.py` 输出）：

```
# repetition penalty (vectorized)
Less_2  : logit < 0
Mul_10  : logit * penalty
Div_0   : logit / penalty
Where_3 : Where(Less_2, Mul_10, Div_0)         # logit<0 ? mul : div
Where_4 : Where(seen_mask, Where_3, logit)     # 仅在 seen 位置应用

# temperature
Div_1   : Where_4 / temperature

# top-k
TopK_0  : TopK(Div_1, k=audio_top_k)            # values, indices
Softmax_3 : Softmax(TopK_0.values)              # 临时概率（用于 cumsum）

# top-p (exclusive cumsum)
CumSum_0 : CumSum(Softmax_3, axis=-1)
Sub_0    : CumSum_0 - Softmax_3                 # = exclusive cumsum
Less_3   : Sub_0 < top_p
Where_5  : Where(NOT Less_3, -inf, TopK_0.values)  # 超出 top-p 的 mask 成 -inf

# renormalize
Softmax_4 : Softmax(Where_5)

# inverse-CDF sampling (核心技巧)
CumSum_1 : CumSum(Softmax_4, axis=-1)
Less_4   : CumSum_1 < random_u                   # bool mask
ReduceSum_0 : ReduceSum(Less_4, axes=[-1])       # = 第一个 cumsum >= u 的位置 - 0
                                                 # 即采样到的"top-k 内索引"

# 还原回原 vocab id
Sub_1    : Gather_28 - 1   (off-by-one 校准)
... GatherElements (TopK_0.indices, idx_in_topk) → sampled_id
```

**纯 PyTorch 等价代码**（可直接 `torch.onnx.export`）：

```python
def sample_one_channel(logits, seen_mask, random_u, *,
                       penalty=1.2, temperature=0.8, top_k=25, top_p=0.95):
    # logits: [B, vocab], seen_mask: [B, vocab] bool, random_u: [B]
    pen = torch.where(logits < 0, logits * penalty, logits / penalty)
    logits = torch.where(seen_mask, pen, logits)
    logits = logits / temperature

    topk_v, topk_i = logits.topk(top_k, dim=-1)
    p_temp = topk_v.softmax(dim=-1)
    excl_cumsum = p_temp.cumsum(dim=-1) - p_temp
    topk_v = topk_v.masked_fill(excl_cumsum >= top_p, float('-inf'))
    p = topk_v.softmax(dim=-1)

    cdf = p.cumsum(dim=-1)
    idx_in_topk = (cdf < random_u.unsqueeze(-1)).sum(dim=-1, keepdim=True)
    idx_in_topk = idx_in_topk.clamp(max=top_k - 1)
    return topk_i.gather(-1, idx_in_topk).squeeze(-1)   # [B]
```

外部用 numpy 生成 `random_u = np.random.uniform(0, 1, (B,))`，注入图后即得到与官方完全可复现的随机采样。

---

## 5. Manifest 协议 (`browser_poc_manifest.json`, 503 KB)

顶层字段：
- `format_version`: 1
- `model_files`: 指向 `tts_meta` / `codec_meta` / `tokenizer_model`
- `tts_config`: 全部模型常量（`n_vq=16`, `hidden_size=768`, special token ids 等）
- `prompt_templates`: **直接给出 token IDs 数组**（不依赖端侧 tokenizer 模板拼接）
  - `user_prompt_prefix_token_ids` (12 个)
  - `user_prompt_after_reference_token_ids` (56 个)
  - `assistant_prompt_prefix_token_ids` (6 个)
- `generation_defaults`: 推理默认参数
- `builtin_voices`: **18 个内置音色**，每个带 `voice / display_name / group / audio_file / prompt_audio_codes`（已经预跑 `audio_encoder` 拿到的 `[T, 16]` token 矩阵，用户切换音色零成本）
- `text_samples`: 内置示范文本，带 `text_token_ids`

---

## 6. `tts_browser_onnx_meta.json` 元数据

定义 5 个图的文件名、external_data 关系、模型常量、所有图的 IO 名顺序。这是端侧 runtime 的"启动配置"。

---

## 7. 与我们当前 `export_onnx.py` 的差距

| 维度 | 我们 | 官方 | 影响 |
|---|---|---|---|
| Prefill / decode 拆分 | ✗ 单图 | ✓ | 单 token 推理图体积小 380×，加载更快 |
| Local 三 variant | ✗ text/audio 二分 | ✓ | 灵活 + 融合采样 |
| 采样融合 | ✗ 外部 numpy | ✓ inverse-CDF + ext random_u | 端侧免实现复杂 sampler |
| External data 共享 | ✗ 内嵌权重 | ✓ 共享 .data | 多 graph 0 重复存储 |
| Manifest 完整度 | ✗ 仅文件清单 | ✓ token IDs + 内置音色 | 端侧免做 prompt 模板与音色提取 |
| 图后处理 fusion | ✓ ORT_ENABLE_EXTENDED 烘焙 | ✗ 干净图 | **我们的 fusion 在 ARM64 ORT 触发 bug，官方无此问题** |
| 量化 | ✓ INT8 dynamic | ✗ FP32 | 我们体积优势 4× |
| opset | 14 | 17 | 官方更现代 |
| n_vq 行宽 | 17 | 17 | 一致 |
| hidden_size / heads / head_dim | 768 / 12 / 64 | 768 / 12 / 64 | 一致 |

---

## 8. 现有 PyTorch 资产清单

- `MOSS-TTS-Nano-100M/pytorch_model.bin` (224 MB, 194 keys) — 官方原版 checkpoint
- `MOSS-TTS-Nano-100M/modeling_moss_tts_nano.py` — `MossTTSNanoForCausalLM` HF 包装
  - `transformer.h.0..11` = global transformer (12 层)
  - `local_transformer.h.0` = local decoder (1 层)
  - 已有 `generate / generate_stream` Python 路径，不带 ONNX export
- `MOSS-TTS-Nano-100M/configuration_moss_tts_nano.py` — 配置类
- `MOSS-TTS-Nano-100M/tokenization_moss_tts_nano.py` — SentencePiece tokenizer
- `MOSS-TTS-Nano-100M/prompting.py` — prompt 模板
