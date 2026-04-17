# MOSS-TTS-Nano · ONNX 部署套件

[English](README_ONNX.md) | [简体中文](README_ONNX_zh.md)

本文档说明 MOSS-TTS-Nano 的 ONNX 导出与推理流程，目标是把模型打包成一个
**自包含的 ONNX bundle**，仅依赖 ONNX Runtime，在 CPU 上即可推理，运行
时不再需要 PyTorch / Transformers。

## 目录

- [MOSS-TTS-Nano · ONNX 部署套件](#moss-tts-nano--onnx-部署套件)
  - [目录](#目录)
  - [为什么要做 ONNX](#为什么要做-onnx)
  - [导出了什么](#导出了什么)
  - [导出](#导出)
  - [推理](#推理)
  - [声音克隆](#声音克隆)
  - [性能](#性能)
  - [架构说明](#架构说明)
  - [仓库布局](#仓库布局)
  - [已知限制](#已知限制)

## 为什么要做 ONNX

仓库现有的 Python 入口（`infer.py`、`app.py`、`moss-tts-nano` CLI）依赖
PyTorch + Transformers + HuggingFace 缓存。这套依赖对研究和本地 demo 没
有问题，但要把模型嵌入到其他运行时（移动端、`onnxruntime-web` 浏览器、
桌面应用、微服务）就略显沉重。

`export_onnx.py` 产出的 bundle 把 MOSS-TTS-Nano 拆成 **6 张小型 ONNX 图
（INT8 总计约 165 MB）**，外加 SentencePiece tokenizer 与一份精简
`config.json`。`onnx_infer.py` 给出对应的推理实现，运行时只用
`onnxruntime`、`numpy`、`soundfile`、`sentencepiece`。

模型本身就是**自回归**的，因此推理循环也是逐帧的：每生成一帧 audio code
就立即解码出一段音频，**第一段音频在整句生成完之前就能拿到**。

## 导出了什么

成功执行 `python export_onnx.py` 后会在 `onnx_export/` 下生成（大小取自
一次样例运行，FP32 / INT8）：

| 文件 | 作用 | FP32 | INT8 |
|---|---|---:|---:|
| `audio_encoder.onnx` | 音频 tokenizer 编码器；用于声音克隆时把 prompt 音频编码为 token。 | 45.4 MB | 15.7 MB |
| `global_transformer.onnx` | 12 层 GPT2 + token embeddings；返回最后一个 token 的 hidden state，KV cache 显式作为 I/O。 | 441.1 MB | 111.0 MB |
| `local_decoder_text.onnx` | 1 层 local GPT2，输出候选文本/EOS token 的 logits。 | 28.4 MB | 7.1 MB |
| `local_decoder_audio.onnx` | 1 层 local GPT2，输出 16 个 audio codebook token。混合输入模式（外部 embed / 内部 lookup）。 | 78.7 MB | 19.7 MB |
| `audio_decoder.onnx` | 音频 tokenizer 解码器，**单帧调用**，KV cache 显式 I/O。 | 44.4 MB | 11.5 MB |
| `audio_decoder_state_spec.json` | KV cache 布局描述，运行时按它装配输入。 | 6 KB | – |
| `tokenizer.model` | SentencePiece tokenizer 原样复制。 | 0.5 MB | – |
| `config.json` | 推理参数：nq、词表、特殊 token id、采样默认值、采样率等。 | <1 KB | – |
| `manifest.json` | bundle 元信息：schema 版本、量化设置、文件清单与大小。 | <1 KB | – |

**总计（样例运行）：5 张图 FP32 ~640 MB / INT8 ~165 MB。**

INT8 权重通过 `onnxruntime.quantization.quantize_dynamic` 生成，参数：
`weight_type=QInt8`、`per_channel=False`、`reduce_range=False`。这组配置
在多次网格对比中被验证为**体积最小且推理最快**的选择，激活值保持 FP32。

## 导出

```bash
python export_onnx.py \
    --tts-checkpoint ./MOSS-TTS-Nano-100M \
    --audio-tokenizer-checkpoint ./MOSS-Audio-Tokenizer-Nano \
    --output-dir ./onnx_export
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--tts-checkpoint` | `./MOSS-TTS-Nano-100M` | LM checkpoint 的本地路径或 HF repo。 |
| `--audio-tokenizer-checkpoint` | `./MOSS-Audio-Tokenizer-Nano` | 音频 tokenizer 的本地路径或 HF repo。 |
| `--output-dir` | `./onnx_export` | bundle 输出目录。 |
| `--nq` | `16` | 音频 codebook 数（必须与模型一致）。 |
| `--device` | `cpu` | 导出设备，CPU 即可。 |
| `--skip-verify` | – | 跳过 ORT vs PyTorch 的数值一致性校验。 |
| `--skip-quantize` | – | 跳过 INT8 动态量化。 |

执行流程：

1. 用 `from_pretrained(..., attn_implementation="sdpa")` 加载
   `MOSS-TTS-Nano-100M`（LM）和 `MOSS-Audio-Tokenizer-Nano`（音频 codec）。
2. 把 5 个 PyTorch wrapper 通过 `torch.onnx.export` 导出为 ONNX
   （`audio_encoder`、`global_transformer`、`local_decoder_text`、
   `local_decoder_audio`、`audio_decoder`）。
3. 跑 ORT 端的图优化：global transformer 用
   `onnxruntime.transformers.optimizer` 做 GPT2 专项 fuse，其余模型用
   `ORT_ENABLE_EXTENDED`。
4. 在若干典型 shape 上做 ORT vs PyTorch 一致性校验
   （`max_diff` 应低于 `1e-3`）。
5. 对全部模型做 INT8 动态量化。
6. 拷贝 `tokenizer.model`，写出 `config.json` 与 `manifest.json`。


## 推理

```bash
python onnx_infer.py \
    --onnx-dir onnx_export \
    --precision int8 \
    --text "你好，这是一段使用 ONNX 推理生成的语音。" \
    --output output/hello.wav
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--onnx-dir` | `onnx_export` | bundle 路径。 |
| `--precision` | `int8` | `auto` / `int8` / `fp32`。`int8` 在缺失对应 `*_int8.onnx` 时会自动回退到 FP32。 |
| `--text` / `--text-file` | （二选一必填） | 输入文本或 UTF-8 文本文件。 |
| `--output` | （必填） | 输出 wav 路径。 |
| `--threads` | `2` | ORT intra-op 线程数。 |
| `--prompt-audio-path` | – | 声音克隆参考音频。 |
| `--max-frames` | `300` | 最多生成多少帧前强制 EOS。 |

推理时的具体步骤：

1. 文本经 `text_normalization_pipeline.prepare_tts_request_texts`
   归一化，再用 SentencePiece 切分为 token。
2. 构造 prompt，逐 token 喂入 `global_transformer`，KV cache 增量推进
   （前缀不重算）。
3. 每一步先用 `local_decoder_text` 采样下一个文本/EOS token，再用
   `local_decoder_audio` 串行 16 次采样出一帧完整的 audio code。
4. 这一帧立即送入 `audio_decoder.step()`，解码器自己维护 KV cache，
   每次返回一段 O(1) 的 waveform chunk。
5. 把所有 chunk 拼起来写成一个 wav 文件。

采样默认值从 `config.json` 读取，可以通过
`OnnxTTSEngine.generate(...)` 的关键字参数覆盖。

## 声音克隆

加上 `--prompt-audio-path` 即可：

```bash
python onnx_infer.py \
    --onnx-dir onnx_export \
    --precision int8 \
    --prompt-audio-path assets/audio/zh_1.wav \
    --text "复刻一下这段声音的音色。" \
    --output output/clone.wav
```

参考音频用 `soundfile` 读入，用 `resampy` 重采样到 48 kHz 立体声，再交
给 `audio_encoder.onnx` 编码为 16 个 codebook 的 token，作为音频前缀
喂给 LM 进行条件生成。

只要带了 `--prompt-audio-path`，模式会自动从 `continuation` 切到
`voice_clone`。

## 性能

下面是 4 核 CPU 笔记本上的实测，INT8、`inter_op_num_threads=1`、
`intra_op_num_threads=2`：

| 模式 | 输入文本 | 音频时长 | 首帧延迟 | RTF |
|---|---|---:|---:|---:|
| continuation | "你好，这是一个端到端测试。"（13 字） | 2.80 s | **80 ms** | 0.33 |
| voice_clone | "克隆模式测试。" + `assets/audio/zh_1.wav` | 1.52 s | 1023 ms | 0.95 |

说明：

- **continuation 模式首帧延迟约 50–90 ms**：CPU INT8 上一次完整 AR step
  （global + local-text + 16 次 local-audio）加一次 audio decoder 调用。
- **voice clone 的首帧延迟瓶颈在 prompt 编码**。约 92 % 的时间花在读
  音频 + 重采样 + `audio_encoder.run` 上。第一帧出来之后，后续每帧的成
  本与 continuation 完全相同。
- **稳态 RTF 都低于 1.0**（快于实时），流式消费者在预热后不会出现
  "断流"。

## 架构说明

几个关键设计选择：

- **拆成多个 wrapper 而不是单一 monolithic 导出。** 模型本身就由三块逻
  辑组成（文本/音频混合的 global LM、两个 local 头、音频 codec），分别
  导出可以让每张图都小到能用最朴素的 per-tensor INT8，并且让
  `onnxruntime.transformers.optimizer` 的 GPT2 专项 fuse 只作用在
  global transformer 上。
- **LM 与 audio decoder 都把 KV cache 暴露成 ONNX I/O。** cache 形状描
  述存在 `audio_decoder_state_spec.json` 里。这样既能做到逐帧 O(1) 解
  码（不重算前缀），又保留了 `torch.onnx.export` 的可追踪性。
- **`local_decoder_audio` 用混合输入模式。** 同一张图同时处理"第 0 个
  音频通道：消费 global transformer 的外部 hidden state"和"第 1–15 个
  通道：内部 lookup 上一个音频 token"两种路径，避免导出两张近乎重复的
  图。
- **per-tensor INT8 + `DefaultTensorType=FLOAT`。** `per_channel=True`
  在这些图上反而更大且没有可测速度收益；`reduce_range=True` 会让某些
  kernel 走 CPU EP 上较慢的路径。当前配置是小型网格对比里"体积+速度同
  时最优"的点。

## 仓库布局

ONNX 相关工作在仓库根目录新增三个 Python 文件：

```
MOSS-TTS-Nano/
├── export_onnx.py        # PyTorch -> ONNX bundle（本 PR）
├── onnx_infer.py         # CLI + OnnxTTSEngine（本 PR）
├── onnx_tts_utils.py     # SentencePiece + 采样工具（本 PR）
├── onnx_export/          # bundle 输出目录（已加入 .gitignore）
│   ├── audio_encoder[_int8].onnx
│   ├── global_transformer[_int8].onnx
│   ├── local_decoder_text[_int8].onnx
│   ├── local_decoder_audio[_int8].onnx
│   ├── audio_decoder[_int8].onnx
│   ├── audio_decoder_state_spec.json
│   ├── tokenizer.model
│   ├── config.json
│   └── manifest.json
└── ...（其它 PyTorch 入口与原仓库完全一致）
```

本 PR **不修改任何已有文件**：`infer.py`、`app.py`、
`moss_tts_nano_runtime.py`、`moss_tts_nano/` 包、文本归一化脚本、模型
权重、`assets/` 等等都保持原状。

## 已知限制

- **仅 CPU、仅 ONNX Runtime CPU EP**。bundle 在任何 ORT 支持的平台上
  都能运行，但没有做 GPU 专项优化。CUDA EP 和 `onnxruntime-web` 在原
  理上可以工作但未在本仓库验证。
- **batch_size = 1**。LM step 和 audio decoder 都按 `batch=1` 导出，
  以保持 KV cache 布局简单。批量推理需要重新导出。
- **最大序列长度**。global transformer 的 `seq_len` 轴是动态的，但
  local decoder 固定 `seq_len = NQ + 1 = 17`（一帧）。长文本通过外层
  循环拼接生成，而不是通过改这个形状。
- **没有打包到 CoreML / ORT-mobile**。导出的是标准 ONNX opset 17，转
  换到其它运行时需要使用方自行处理。
