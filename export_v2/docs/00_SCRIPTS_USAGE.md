# 脚本与使用流程（v7 终版）

`export_v2/scripts/` 下的所有脚本及其使用方式，按"从源码 checkpoint 到 `release/` 终板"的顺序排列。

> **最简用法**：一条命令产出最终包：
>
> ```bash
> python3 export_v2/scripts/make_release.py
> ```
>
> 输出 `export_v2/release/`（**164.6 MB**, 11 个文件，含 `manifest.json` 和 `README.md`），全程 ~60 秒。

---

## 0. 前置条件

- Python 3.12 环境（`conda activate moss`）
- HuggingFace 上 `OpenMOSS-Team/MOSS-TTS-Nano-100M` checkpoint 已下载到 `MOSS-TTS-Nano-100M/`
- 官方 codec FP32 ONNX 在 `export_v2/reference/codec_full/`（量化输入）
- 官方 prompt template 在 `export_v2/reference/browser_poc_manifest.json`
- 官方 SentencePiece tokenizer 在 `export_v2/reference/full/tokenizer.model`

---

## 1. 一键构建终板

```bash
python3 export_v2/scripts/make_release.py            # 全流水线
python3 export_v2/scripts/make_release.py --keep-build  # 保留 _build/ 中间产物
python3 export_v2/scripts/make_release.py --skip-export  # 复用已有的 _build/tts_fp32/
python3 export_v2/scripts/make_release.py --skip-quant   # 复用已有的 _build/*_int8_inlined/
```

`make_release.py` 内部按以下 6 个 stage 顺序执行（subprocess 调用 stage 脚本）：

| stage | 脚本                       | 输入                              | 输出                              |
|-------|---------------------------|-----------------------------------|-----------------------------------|
| 1     | `m1_…` `m2_…` `m3_…` `m3_2_…` | PyTorch checkpoint           | `_build/tts_fp32/` (4 graph)      |
| 2     | `m4_share_external_data.py` | `_build/tts_fp32/`              | `_build/tts_fp32_shared/` (split mode, 量化前提) |
| 3     | `m5_1_quantize_dynamic.py`  | `_build/tts_fp32_shared/`       | `_build/tts_int8_inlined/` (4 inline INT8 graph) |
| 4     | `m6_1_quantize_codec.py`    | `reference/codec_full/`          | `_build/codec_int8_inlined/` (3 inline INT8 graph) |
| 5     | `m4_share_external_data.share_group(unified=True)`（直接调用） | `_build/*_int8_inlined/` | `release/` 中的两个 unified blob |
| 6     | manifest 生成 + README 生成 | release 内文件                   | `release/manifest.json` + `release/README.md` |

终态：

```
export_v2/release/                        ← 164.6 MB 总
├── moss_tts_prefill.onnx                 1.46 MB  (graph only)
├── moss_tts_decode_step.onnx             1.46 MB
├── moss_tts_local_decoder.onnx           0.24 MB
├── moss_tts_local_cached_step.onnx       0.30 MB
├── moss_tts_shared.data                136.68 MB  (4 个 TTS 图共享 1 份 INT8 权重)
├── moss_audio_tokenizer_encode.onnx      0.93 MB
├── moss_audio_tokenizer_decode_full.onnx 0.77 MB
├── moss_audio_tokenizer_decode_step.onnx 0.43 MB
├── moss_audio_tokenizer_shared.data     21.87 MB  (3 个 codec 图共享 1 份 INT8 权重)
├── tokenizer.model                       0.45 MB  (SentencePiece)
├── manifest.json                         0.04 MB  (full provenance + IO + SHA256)
└── README.md                                 3 KB
```

---

## 2. 端到端推理 + 性能验证

### 生成 wav

两种模式（互斥）：

```bash
# 模式 A: 使用 manifest 内置音色（不调 encoder，最快路径）
python3 export_v2/scripts/demo_generate.py \
    --voice 0 \
    --text zh_browser_poc \
    --frames 250 \
    --seed 0 \
    --out export_v2/outputs/test_voice0.wav

# 模式 B: clone 任意 wav（先过 encoder 提取 codes，再走 TTS）
python3 export_v2/scripts/demo_generate.py \
    --clone assets/audio/zh_1.wav \
    --text zh_browser_poc \
    --frames 250 \
    --seed 0 \
    --out export_v2/outputs/test_clone.wav
```

参数：
- `--bundle DIR` — bundle 目录，默认 `export_v2/release/`，可用 `$TTS_BUNDLE` 覆盖
- `--voice N` *(与 `--clone` 互斥)* — manifest `builtin_voices` 索引（0..17），不指定时默认 `--voice 0`
- `--clone WAV` *(与 `--voice` 互斥)* — 参考 wav 路径；自动支持任意采样率/声道/格式（mono/stereo, wav/flac），内部 resample 到 48 kHz stereo 后送 encoder；超过 20s 自动截断
- `--text` — `zh_browser_poc` / `en_browser_poc`
- `--frames` — 最大帧数（命中 `audio_end_token` 提前停止）
- `--seed` — 采样随机种子
- `--no-sample` — 切换为 greedy（去随机）
- `--out` — wav 输出路径

> Clone 模式额外用到 `moss_audio_tokenizer_encode.onnx`（release/ 已包含），仅在加载时多 ~150 ms session init + ~100 ms encoder 推理（7s 音频）。预先 cache codes 即可跳过这部分。

### 性能 benchmark

```bash
python3 export_v2/scripts/benchmark.py
# 多 bundle 对比：
python3 export_v2/scripts/benchmark.py release _build/tts_int8_inlined
```

最新 release/ 数据（CPU multi-thread, ORT_ENABLE_ALL）：

| 指标                                       | 值          |
|--------------------------------------------|-------------|
| bundle size                                | 164.6 MB    |
| prefill (seq=195) 单次延迟                 | 34.7 ms     |
| decode_step (past=200) 中位延迟            | 4.94 ms     |
| local_cached_step (past=8) 中位延迟        | 0.52 ms     |
| 单帧推理 (1 decode + 17 local) 估算        | 13.8 ms     |
| realtime factor @ 12.5 fps (80 ms/frame)   | **0.17×（5.8× 实时）** |

报告写入 `export_v2/outputs/benchmark.json`。

---

## 3. 各脚本职责一览

| 脚本                                  | 角色      | 职责                                               |
|---------------------------------------|-----------|----------------------------------------------------|
| `make_release.py`                     | **入口**  | 一键产出 `release/`（链路里调用所有 stage）         |
| `m1_export_prefill.py`                | stage 1   | PrefillWrapper → ONNX (dynamo)                     |
| `m2_export_decode_step.py`            | stage 1   | DecodeStepWrapper → ONNX (dynamo)                  |
| `m3_export_local_decoder.py`          | stage 1   | LocalDecoderWrapper → ONNX (dynamo)                |
| `m3_2_export_local_cached_step.py`    | stage 1   | LocalCachedStepWrapper → ONNX (dynamo)             |
| `m4_share_external_data.py`           | stage 2/5 | 跨图权重去重，支持 SPLIT(默认) / `--unified`        |
| `m5_1_quantize_dynamic.py`            | stage 3   | TTS INT8 dynamic（含 weight untie + value_info 清理）|
| `m6_1_quantize_codec.py`              | stage 4   | Codec INT8 dynamic (MatMul + Gather + Conv)        |
| `demo_generate.py`                    | tool      | release/ 端到端推理生成 wav（支持 `--voice` builtin / `--clone` wav 两种模式）|
| `benchmark.py`                        | tool      | 任一 bundle 的 size/init/prefill/decode/local 性能 |

> 删除的旧脚本（v7 之前）：`m5_2_share_int8_dyn.py`、`m6_2_share_codec.py` —— 它们的 sharing 逻辑已并入 `make_release.py` stage 5。

---

## 4. 目录布局

```
export_v2/
├── docs/
│   ├── 00_SCRIPTS_USAGE.md           (本文件)
│   ├── 01_OFFICIAL_ARCHITECTURE.md   (官方 5-graph IO 协议剖析)
│   ├── 02_PLAN.md                    (原始 6-milestone 计划，归档)
│   └── 03_EXPORT_JOURNEY.md          (v1 → v7 演进 + bug 复盘)
├── reference/
│   ├── browser_poc_manifest.json     (prompt template + 18 builtin voices)
│   ├── codec_full/                   (官方 FP32 codec ONNX —— 量化输入)
│   └── full/tokenizer.model          (SentencePiece —— release 复制源)
├── wrappers/                          (4 个 nn.Module wrapper + 共享代码)
│   ├── _common.py                    (patch_attention + build_multi_channel_inputs_embeds)
│   ├── prefill.py
│   ├── decode_step.py
│   ├── local_decoder.py
│   └── local_cached_step.py
├── _build/                            (make_release.py 中间产物，结束时自动清理；--keep-build 保留)
│   ├── tts_fp32/                     stage 1 输出
│   ├── tts_fp32_shared/              stage 2 输出
│   ├── tts_int8_inlined/             stage 3 输出
│   └── codec_int8_inlined/           stage 4 输出
├── release/                           ← **终板**（用户面向）
│   ├── *.onnx + *.data + tokenizer.model + manifest.json + README.md
├── outputs/                           (wav + benchmark.json)
└── scripts/                           (本文档涉及的 10 个脚本)
```

> 中间产物 `_build/` 默认在 `make_release.py` 收尾时自动删除（约 2 GB）。
> 整个流水线是确定性的，重跑得到 byte-equal 的 release/。
