# MOSS-TTS-Nano · ONNX Bundle

[English](README_ONNX.md) | [简体中文](README_ONNX_zh.md)

This document describes the ONNX export and inference pipeline added to
MOSS-TTS-Nano. The goal is to make the model deployable as a fully self-
contained ONNX bundle that runs on CPU via ONNX Runtime without any
PyTorch / Transformers dependency at inference time.

## Table of contents

- [MOSS-TTS-Nano · ONNX Bundle](#moss-tts-nano--onnx-bundle)
  - [Table of contents](#table-of-contents)
  - [Why ONNX](#why-onnx)
  - [What is exported](#what-is-exported)
  - [Export](#export)
  - [Inference](#inference)
  - [Voice cloning](#voice-cloning)
  - [Performance](#performance)
  - [Architecture notes](#architecture-notes)
  - [Repository layout](#repository-layout)
  - [Limitations](#limitations)

## Why ONNX

The reference Python entry points (`infer.py`, `app.py`, the
`moss-tts-nano` CLI) require PyTorch + Transformers + a HuggingFace cache.
That is fine for research and the local web demo, but it makes embedding
the model into other runtimes (mobile, browser via onnxruntime-web,
desktop apps, microservices) noticeably heavier.

The bundle produced by `export_onnx.py` lets you ship MOSS-TTS-Nano as
**six small ONNX graphs (~165 MB INT8 in total)** plus the SentencePiece
tokenizer and a tiny `config.json`. Inference is implemented in
`onnx_infer.py` using only `onnxruntime`, `numpy`, `soundfile`, and
`sentencepiece`.

The model is **autoregressive by nature**, so the inference loop is also
chunk-by-chunk: each generated code frame is decoded into audio
immediately, and the first chunk is available before the full sentence
finishes generating.

## What is exported

A successful run of `python export_onnx.py` writes the following files
into `onnx_export/` (sizes from a sample run, FP32 / INT8):

| File | Role | FP32 | INT8 |
|---|---|---:|---:|
| `audio_encoder.onnx` | Audio tokenizer encoder; encodes the prompt waveform for voice cloning. | 45.4 MB | 15.7 MB |
| `global_transformer.onnx` | 12-layer GPT2 + token embeddings. Returns the last-token hidden state with explicit KV cache I/O. | 441.1 MB | 111.0 MB |
| `local_decoder_text.onnx` | 1-layer local GPT2 head producing the candidate text-/EOS-token logits. | 28.4 MB | 7.1 MB |
| `local_decoder_audio.onnx` | 1-layer local GPT2 head producing the 16 audio codebook tokens. Hybrid input mode (external embed / internal lookup). | 78.7 MB | 19.7 MB |
| `audio_decoder.onnx` | Audio tokenizer decoder, one frame per call, with explicit KV-cache I/O. | 44.4 MB | 11.5 MB |
| `audio_decoder_state_spec.json` | KV-cache layout descriptor consumed by the runtime. | 6 KB | – |
| `tokenizer.model` | SentencePiece tokenizer copied as-is. | 0.5 MB | – |
| `config.json` | Inference parameters (nq, vocab size, special token ids, sampling defaults, sample rate). | <1 KB | – |
| `manifest.json` | Bundle metadata: schema, quantization settings, file sizes. | <1 KB | – |

**Totals (sample run): ~640 MB FP32 / ~165 MB INT8 for the five graphs.**

INT8 weights are produced via `onnxruntime.quantization.quantize_dynamic`
with `weight_type=QInt8`, `per_channel=False`, `reduce_range=False`. These
settings were validated as the smallest *and* fastest configuration on
these graphs across multiple grids; the activations stay in FP32.

## Export

```bash
python export_onnx.py \
    --tts-checkpoint ./MOSS-TTS-Nano-100M \
    --audio-tokenizer-checkpoint ./MOSS-Audio-Tokenizer-Nano \
    --output-dir ./onnx_export
```

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--tts-checkpoint` | `./MOSS-TTS-Nano-100M` | Local path or HF repo for the LM checkpoint. |
| `--audio-tokenizer-checkpoint` | `./MOSS-Audio-Tokenizer-Nano` | Local path or HF repo for the audio tokenizer. |
| `--output-dir` | `./onnx_export` | Where to write the bundle. |
| `--nq` | `16` | Number of audio codebooks (must match the model). |
| `--device` | `cpu` | Export device. CPU is fine; GPU is not required. |
| `--skip-verify` | – | Skip the ORT-vs-PyTorch numerical sanity check. |
| `--skip-quantize` | – | Skip INT8 dynamic quantization. |

Pipeline:

1. Loads `MOSS-TTS-Nano-100M` (causal LM) and `MOSS-Audio-Tokenizer-Nano`
   (codec) via `from_pretrained` with `attn_implementation="sdpa"`.
2. Traces five PyTorch wrappers to ONNX (`audio_encoder`,
   `global_transformer`, `local_decoder_text`, `local_decoder_audio`,
   `audio_decoder`).
3. Runs ORT-side graph optimizations: GPT2-specific operator fusion via
   `onnxruntime.transformers.optimizer` for the global transformer, and
   `ORT_ENABLE_EXTENDED` for the rest.
4. Verifies each graph against the original PyTorch wrapper on a few
   shapes (`max_diff` should stay below `1e-3`).
5. Runs INT8 dynamic quantization on every model.
6. Copies `tokenizer.model`, writes `config.json` and `manifest.json`.

The export takes ~5 minutes on a modern laptop CPU. No GPU required.

## Inference

```bash
python onnx_infer.py \
    --onnx-dir onnx_export \
    --precision int8 \
    --text "你好，这是一段使用 ONNX 推理生成的语音。" \
    --output output/hello.wav
```

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--onnx-dir` | `onnx_export` | Path to the bundle. |
| `--precision` | `int8` | One of `auto`, `int8`, `fp32`. `int8` falls back to FP32 for any model whose `*_int8.onnx` is missing. |
| `--text` / `--text-file` | (required, mutually exclusive) | Input text or UTF-8 file. |
| `--output` | (required) | Destination wav path. |
| `--threads` | `2` | ORT intra-op thread count. |
| `--prompt-audio-path` | – | Reference audio for voice cloning. |
| `--max-frames` | `300` | Max audio frames to generate before forcing EOS. |

What runs at inference time:

1. Text is normalized via `text_normalization_pipeline.prepare_tts_request_texts`
   and tokenized with SentencePiece.
2. A prompt is built and fed into `global_transformer` token-by-token with
   incremental KV cache (no recomputation).
3. Each step runs `local_decoder_text` to sample the next text/EOS token,
   then `local_decoder_audio` 16 times to sample one full audio frame.
4. The frame is immediately handed to `audio_decoder.step()`, which keeps
   its own KV cache and yields one waveform chunk in O(1).
5. The chunks are concatenated and written as a single wav file.

Sampling defaults are read from `config.json` and can be overridden via
`OnnxTTSEngine.generate(...)` keyword arguments.

## Voice cloning

Pass a reference wav with `--prompt-audio-path`:

```bash
python onnx_infer.py \
    --onnx-dir onnx_export \
    --precision int8 \
    --prompt-audio-path assets/audio/zh_1.wav \
    --text "复刻一下这段声音的音色。" \
    --output output/clone.wav
```

The prompt audio is read with `soundfile`, resampled to 48 kHz / stereo
with `resampy`, and encoded by `audio_encoder.onnx` into 16-codebook
tokens. Those tokens become the audio prefix that conditions the LM.

If `--prompt-audio-path` is supplied, mode automatically switches from
`continuation` to `voice_clone`.

## Performance

Numbers from a 4-core CPU laptop, INT8, single thread for inter-op,
2 threads for intra-op:

| Mode | Text | Audio length | First chunk | RTF |
|---|---|---:|---:|---:|
| continuation | "你好，这是一个端到端测试。" (13 chars) | 2.80 s | **80 ms** | 0.33 |
| voice_clone | "克隆模式测试。" + `assets/audio/zh_1.wav` | 1.52 s | 1023 ms | 0.95 |

Notes:

- **First-chunk latency in continuation mode is ~50–90 ms** on CPU INT8:
  one full AR step (global + local-text + 16× local-audio) plus one
  audio-decoder call.
- **Voice clone first-chunk latency is dominated by prompt encoding.**
  About 92 % of the ~1 s spent on the first chunk in clone mode is
  prompt I/O + resampling + `audio_encoder.run`. Once the first chunk is
  out, subsequent chunks have the same per-step cost as continuation.
- **Steady-state RTF is well below 1.0** in both modes (faster than
  realtime), so streaming consumers will never starve after warmup.

## Architecture notes

A few design decisions that explain why the bundle looks the way it does:

- **Wrapper-per-graph instead of one monolithic export.** The model has
  three logical pieces (text/audio token mixing global LM, two local
  heads, audio codec). Exporting them separately keeps each ONNX graph
  small enough for INT8 dynamic quantization to be applied with simple
  per-tensor settings, and lets `onnxruntime.transformers.optimizer`
  apply GPT2-specific fusions to the global transformer only.
- **Explicit KV-cache I/O for both the LM and the audio decoder.** The
  cache state is exposed as ONNX inputs/outputs, with shape information
  stored in `audio_decoder_state_spec.json`. This is what enables true
  per-frame O(1) decoding (no recompute over the prefix) while keeping
  the export traceable in plain `torch.onnx.export`.
- **`local_decoder_audio` uses hybrid input mode.** A single graph
  handles both the "first audio channel" path (consume external
  hidden-state from the global transformer) and the "next 15 channels"
  path (look up the previously emitted audio token internally). This
  avoids exporting two near-duplicate graphs.
- **Per-tensor INT8 with `DefaultTensorType=FLOAT`.** `per_channel=True`
  produced larger files with no measurable speedup on these graphs;
  `reduce_range=True` made some kernels fall back to a slower path on
  CPU EP. The chosen setting was validated as the smallest-and-fastest
  point in a small grid search.

## Repository layout

The ONNX work adds three Python files at the repository root:

```
MOSS-TTS-Nano/
├── export_onnx.py        # PyTorch -> ONNX bundle (this PR)
├── onnx_infer.py         # CLI + OnnxTTSEngine (this PR)
├── onnx_tts_utils.py     # SentencePiece + sampling utilities (this PR)
├── onnx_export/          # Bundle output directory (gitignored)
│   ├── audio_encoder[_int8].onnx
│   ├── global_transformer[_int8].onnx
│   ├── local_decoder_text[_int8].onnx
│   ├── local_decoder_audio[_int8].onnx
│   ├── audio_decoder[_int8].onnx
│   ├── audio_decoder_state_spec.json
│   ├── tokenizer.model
│   ├── config.json
│   └── manifest.json
└── ... (existing PyTorch entrypoints unchanged)
```

The PR does **not** touch any existing file: `infer.py`, `app.py`,
`moss_tts_nano_runtime.py`, the `moss_tts_nano/` package, the text
normalization scripts, the model checkpoints, or the assets folder all
behave exactly as before.

## Limitations

- **CPU only, ONNX Runtime CPU EP.** The bundle works on any platform
  ORT supports, but no GPU-specific optimizations have been done. CUDA
  EP and `onnxruntime-web` should both work but are not tested here.
- **Batch size 1.** Both the LM step and the audio decoder are exported
  with `batch=1` to keep the KV-cache layout simple. Batched inference
  would need a re-export.
- **Maximum sequence length.** The exported global transformer uses
  fully dynamic `seq_len` axes, but the local decoder fixes
  `seq_len = NQ + 1 = 17` (one frame). Long-form generation is handled
  by the streaming loop, not by changing this shape.
- **No CoreML / ORT-mobile packaging.** The graphs are vanilla ONNX
  opset 17. Conversion to other runtimes is left to downstream users.
