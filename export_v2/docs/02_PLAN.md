# export_v2 实施计划

**总目标**：从零构建一套**完全对齐官方 IO 协议**的 ONNX 导出管线，并在此基础上做 **FP32 / INT8-dynamic / INT8-static 三档量化广度对比**，最终产出能在 ARM64 ORT 上稳定运行的端侧推理包。

---

## 0. 设计原则

1. **完全对齐官方 IO**：所有 graph 输入名、输出名、张量顺序、shape 命名（含 dynamic axis 名）严格匹配 `tts_browser_onnx_meta.json`，便于交叉验证与未来替换。
2. **不烘焙 fusion**：所有 `.onnx` 在 export 阶段保持 PyTorch 原生 op 序列，**不做 ORT-side optimization 烘焙**，避免重蹈 ARM64 `/Add_2` 覆辙。优化交给运行时按平台决定。
3. **External data 共享**：多 graph variant 通过 `save_as_external_data` 共享权重文件，匹配官方布局。
4. **采样融合用 PyTorch 写**：把 inverse-CDF 采样写成纯 PyTorch 代码再 `torch.onnx.export`，所有节点都标准 opset 17 op。
5. **每个 milestone 都有数值对齐验证**：用相同输入对比 (官方 ONNX) vs (我们新 ONNX) vs (PyTorch 原版)，bit-level 偏差 < 1e-3 (FP32) / < 0.1 (INT8) 才能继续。

---

## 1. Milestone 拆分

### M0 — 环境 + 加载基线  *(0.5 d)*
- [ ] M0.1 把 PyTorch checkpoint 加载到 `MossTTSNanoForCausalLM`，跑一段 zh 文本，dump 出每层中间张量作为黄金参考。
- [ ] M0.2 用 `onnxruntime` 加载官方 5 个 graph + `.data`（需先下载 640 MB）；用同样输入跑通官方推理，确认我们能复现官方音频输出。
- [ ] **退出条件**：(a) PyTorch 路径生成 1 段音频 OK；(b) 官方 ONNX 路径生成 1 段音频 OK；(c) 两者数值对齐（global_hidden 在前 5 步内 max|Δ| < 1e-2）。

### M1 — Prefill wrapper  *(1 d)*
- [ ] M1.1 在 `wrappers/prefill.py` 写一个 `PrefillWrapper(nn.Module)`，封装 `MossTTSNanoForCausalLM.transformer.forward`，输出 `[hidden, present_kv...]`。
- [ ] M1.2 用 `torch.onnx.export(opset=17, dynamic_axes={...})` 导出 `moss_tts_prefill.onnx`，IO 名严格按官方 meta。
- [ ] M1.3 写 `verify_prefill.py`：用同一个 `input_ids/attention_mask`，跑 (官方 prefill) vs (我们 prefill) vs (PyTorch)，三方对齐。
- [ ] **退出条件**：FP32 三方 max|Δ| < 1e-3 on `global_hidden` 和所有 12 层 KV cache。

### M2 — Decode_step wrapper  *(0.5 d)*
- [ ] M2.1 `wrappers/decode_step.py`：和 prefill 共用 transformer body，但前向接受 `past_key/value` 输入，做 KV cache concat 后 attention。
- [ ] M2.2 导出 + 三方对齐验证。
- [ ] **退出条件**：模拟一次 prefill + 5 次 decode_step，每步 hidden 与 PyTorch 对齐 < 1e-3。

### M3 — Local decoder 三 variant  *(1.5 d)*
- [ ] M3.1 `wrappers/local_decoder.py` (基础 teacher-forcing 版)：3 输入 / 2 输出。
- [ ] M3.2 `wrappers/local_cached_step.py`：单 channel 增量 + KV cache。
- [ ] M3.3 `wrappers/local_fixed_sampled_frame.py`：**重头戏**。把 17 次 forward + 完整采样融合到一个 `nn.Module.forward` 里。
  - 内部 17 次循环用 `torch.jit` 友好的方式展开（必要时用 Python `for` 由 ONNX 静态展开）。
  - 采样按"04 章伪代码"实现，所有 `*_random_u` 走外部输入。
  - 常量超参（temperature/top_k/top_p/repetition_penalty/EOS token id）作为 buffer 嵌入。
- [ ] M3.4 三方对齐：从同一个 `global_hidden` + 固定 seed 出发，三个 variant 都输出 `frame_token_ids[16]`，与官方 `local_fixed_sampled_frame` byte-equal。
- [ ] **退出条件**：(a) 给定 fixed `random_u` 后，frame_token_ids 与官方 ONNX 完全一致 (int 相等)；(b) `local_decoder` / `local_cached_step` 与 PyTorch logits max|Δ| < 1e-3。

### M4 — External data + manifest  *(0.5 d)*
- [ ] M4.1 重导出时用 `save_as_external_data=True, all_tensors_to_one_file=False, location=...`，让 5 份图共享 `moss_tts_global_shared.data` 和 `moss_tts_local_shared.data`。
- [ ] M4.2 生成 `tts_browser_onnx_meta.json`（与官方 byte-compatible）和 `browser_poc_manifest.json`（含 prompt_templates token IDs + 18 个 builtin voices —— builtin voices 可以直接复用官方 manifest 里的，免得重新跑 audio_encoder）。
- [ ] **退出条件**：用 `OrtCpuRuntime`（来自 `reference/`）加载我们的 bundle，能跑出和官方 bundle 一致的音频。

### M5 — 量化广度对比  *(2 d)*
针对每个 graph 分别跑三档：

| 档 | 方法 | 工具 | 校准集需求 |
|---|---|---|---|
| FP32 | 不量化 | 直接 export | 无 |
| INT8 dynamic | `onnxruntime.quantization.quantize_dynamic`（per-channel weight, 动态 activation） | onnxruntime | 无 |
| INT8 static | `quantize_static` + 校准集 | onnxruntime | 100~500 个真实推理样本 |

- [ ] M5.1 准备校准集：用 PyTorch 路径跑 ~50 段 zh + en 文本，dump 中间张量供 static quant calibration。
- [ ] M5.2 三档导出，每档生成完整 bundle。
- [ ] M5.3 评估指标：
  - 数值偏差：与 FP32 baseline 的 max|Δ| / mean|Δ|
  - 文件体积
  - PC 单线程延迟（prefill 1×、decode_step ×100 的 wall-clock）
  - **跨平台 smoke**：每档在 (Linux x86_64) + (Android ARM64 模拟器) 上各跑一次，确认能加载且不崩
- [ ] M5.4 写 `docs/03_QUANTIZATION_REPORT.md`：表格化对比 + 推荐档位 + 平台兼容性结论。
- [ ] **退出条件**：(a) 三档全部能跑出可听音频；(b) 报告完成；(c) 至少一档在 ARM64 ORT 上 verified。

### M6 — 跨平台 smoke + Android 链路验证  *(0.5 d)*
- [ ] M6.1 把推荐档位 bundle 拷到 `android/models/`，用 Android 现有 InferenceLoop 走通 prefill→decode_step→local_fixed_sampled_frame→pcm 全链路。
- [ ] M6.2 检查是否有遗留的 ARM64 ORT 兼容性问题。
- [ ] **退出条件**：能在真机出第一帧 PCM。

---

## 2. 风险与回退

| 风险 | 概率 | 缓解 |
|---|---|---|
| `local_fixed_sampled_frame` 17 次循环展开后导出失败（`torch.onnx` 无法正确处理动态控制流） | 中 | 保留 `local_cached_step` + 外部循环作为备选 |
| INT8 static 量化质量崩盘（音频明显失真） | 中 | 退回 INT8 dynamic 或混合精度（仅 MatMul 量化） |
| ARM64 ORT 仍触发未知 bug | 低 | 我们这次不烘焙 fusion，理论上规避了已知 bug；若仍有问题，逐 op 二分 |
| 官方 `.data` 权重布局有未文档化约束 | 低 | 用 `onnx.save_model(save_as_external_data=True)` 标准 API，不手动构造 |

---

## 3. 时间预算

合计 **≈ 6 工作日**。优先级 M0 → M1 → M2 → M3 → M4，M5/M6 并行可压缩。

---

## 4. 不在本计划范围

- 重写 Android Kotlin InferenceLoop（M6 只复用现有，新协议下的 InferenceLoop 重写算 V1.1）
- 浏览器 WASM 部署
- INT4 / GPTQ 等更激进量化
- Audio Tokenizer (codec) 的导出（沿用官方 `MOSS-Audio-Tokenizer-Nano-ONNX`）
