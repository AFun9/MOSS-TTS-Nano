# Android ONNX Demo · 设计文档

- 状态：**FROZEN** — brainstorming 完成，所有 8 节定稿
- 创建日期：2026-04-16
- 定稿日期：2026-04-16
- 作者：模型 + AFun9
- 关联实现计划：`docs/plans/2026-04-16-android-onnx-demo-plan.md`（writing-plans 阶段产出）
- 关联活文档：`android/DEVLOG.md`（实现期持续维护）

---

## 文档地图

本特性涉及 3 份文档，职责互不重叠：

| 文档 | 路径 | 寿命 | 内容 |
|---|---|---|---|
| ① 设计文档（本文） | `docs/plans/2026-04-16-android-onnx-demo-design.md` | 一次性，定稿后不修改 | 目标、范围、架构、决策依据、显式排除项 |
| ② 实现计划 | `docs/plans/2026-04-16-android-onnx-demo-plan.md` | 一次性 | 拆分任务 / acceptance / 依赖 / checkpoint |
| ③ 开发日志 | `android/DEVLOG.md` | 长期活文档 | Done / Doing / Backlog / TODO / Known Issues / 决策记录 / 性能基线 / Changelog |

设计变更原则：定稿后若需调整，**新增** `docs/plans/YYYY-MM-DD-android-onnx-demo-design-revN.md` 并在头部声明对前一版的差异，**不在原文档原地改写**。

---

## § 1 / 8 — 目标与范围

### 一句话目标

在 `android/` 下交付一个 **Kotlin + Jetpack Compose** 的最小 Android App，**直接驱动我们刚导出的 ONNX bundle (INT8)**，把"端侧 ~80 ms 首帧 + 多语种零配置"这件事用 30 秒就能向人演示出来。

### MVP（= A2 形态）包含

1. 首启自动检测本地是否有模型，否则进**下载页**（GitHub Releases，单文件并行 + 进度 + sha256 校验 + 断点续传）。
2. 主界面：多行文本框 + 语种下拉 + 合成按钮 + 流式播放 + 进度条 + 状态行（首帧 ms / RTF / 量化标记）+ 保存 wav + 分享。
3. 语种下拉**只控制示例文本与文本规范化 locale，不传给模型**。
4. 真流式播放（`AudioTrack` STREAM 模式，每 chunk 立刻 `write()`）；同时内存累加 PCM 用于"保存 wav"。
5. 文本规范化：**L1 通用清理 + L2 ICU 数字读法**（中英日俄等 ICU 自带 locale 都覆盖）。

### 显式不包含

| 项 | 计划阶段 |
|---|---|
| Voice clone（录音 + 引导） | V2（A3 升级） |
| L3 中英日俄特化规则（缩写/符号映射） | V1.1 |
| L4 日期/时间/货币 | V2 |
| HuggingFace / ModelScope 镜像下载源切换 | V1.1 |
| 历史记录 / 收藏 / 后台朗读 / 通知栏 | V2+ |
| 采样参数 UI（temp/top_p/top_k） | 不计划，走默认 |
| Google Play / AAB / Play Asset Delivery | 不计划 |

### 不变约束

- **Python 端零侵入**：`infer.py`、`app.py`、`export_onnx.py`、`onnx_infer.py` 一行不改。所有新增内容只在 `android/` 子目录里。
- **不引入 onnxruntime-extensions**：tokenizer 走 sentencepiece-android JNI（参见决策 #1）。
- **arm64-v8a only**，`minSdk 26 / targetSdk 34`。

### 成功标准

- 在 4 核高通中端机（如 Snapdragon 7 Gen 1）上：
  - 首帧延迟 ≤ **150 ms**（continuation 模式短句）
  - RTF ≤ **0.5**
- 装好 + adb push 模型 → 输入"你好世界" → 听到声音的总操作 ≤ **3 步**
- 装包体积（不含模型）：**≤ 25 MB**

---

## § 2 / 8 — 架构与模块

### 总体架构

整个 App 进程是一个 Activity + Compose 单 UI 树 + 一个跨整个 App 生命周期的 `TtsEngine`。无 Service、无 Hilt、无多进程。

```
┌──────────────────────────────────────────────────────────────┐
│ MainActivity (Compose host)                                  │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ AppNavigation (StateFlow 驱动，无 Navigation 库)           │ │
│ │   ├─ DownloadScreen   (本地无模型时进入)                   │ │
│ │   ├─ MainScreen       (合成播放主界面)                     │ │
│ │   └─ AboutScreen      (版本/上游链接/PR 链接)              │ │
│ └──────────────────────────────────────────────────────────┘ │
│              ↑ collect StateFlow                              │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ MainViewModel  (MVI: State + Intent + Effect)             │ │
│ │   · 持有 TtsEngine 单例引用                                │ │
│ │   · 接收 UI Intent → 调度协程 → 发 State / Effect          │ │
│ └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────┘
                                 │
   ┌─────────────────────────────┴──────────────────────────┐
   │ TtsEngine (App-singleton, IO/Default 协程)              │
   │ ┌────────────────────────────────────────────────────┐ │
   │ │ ModelManager     · 加载/校验 9 文件 manifest         │ │
   │ │                  · 创建 5 个 OrtSession (复用)      │ │
   │ ├────────────────────────────────────────────────────┤ │
   │ │ Tokenizer (sentencepiece-android JNI)               │ │
   │ ├────────────────────────────────────────────────────┤ │
   │ │ TextNormalizerPipeline                              │ │
   │ │   L1 CommonNormalizer                               │ │
   │ │   L2 IcuNumberNormalizer                            │ │
   │ │  (L3 LangNormalizer 留接口，V1.1 接入)               │ │
   │ ├────────────────────────────────────────────────────┤ │
   │ │ PromptBuilder   (移植自 onnx_infer.py)               │ │
   │ ├────────────────────────────────────────────────────┤ │
   │ │ Sampler         (top_k/top_p + repetition penalty)   │ │
   │ ├────────────────────────────────────────────────────┤ │
   │ │ InferenceLoop   · 每步:                              │ │
   │ │                   global → local_text → local_audio  │ │
   │ │                   → audio_decoder → emit PCM chunk   │ │
   │ │                 · Flow<PcmChunk> 输出                 │ │
   │ ├────────────────────────────────────────────────────┤ │
   │ │ AudioPlayer (AudioTrack STREAM)                      │ │
   │ │   · 协程消费 Flow<PcmChunk> → write()                │ │
   │ │   · 同时累加到 PcmAccumulator 用于 saveWav           │ │
   │ └────────────────────────────────────────────────────┘ │
   └────────────────────────────────────────────────────────┘
                                 │
   ┌─────────────────────────────┴──────────────────────────┐
   │ ModelDownloader (独立组件，仅 DownloadScreen 用)        │
   │ · OkHttp 并行下载 9 个文件 + 进度 + sha256 + 断点续传    │
   │ · 写入 getExternalFilesDir(null)/onnx_export/           │
   └────────────────────────────────────────────────────────┘
```

### 模块清单与职责

| 模块 | Kotlin 包 | 职责 | 单测难度 |
|---|---|---|---|
| `ModelManager` | `core.model` | 解析 `manifest.json`、校验文件齐全/sha256、创建并复用 5 个 `OrtSession` | 易 |
| `Tokenizer` | `core.text` | sentencepiece-android JNI 包装；`encode/decode/bos/eos` | 易 |
| `TextNormalizerPipeline` | `core.text` | L1/L2/L3 链式调用，按 locale 取 L3 实现 | 易，规则即测试 fixture |
| `PromptBuilder` | `core.infer` | 构造 `(input_ids, attention_mask)`，逻辑 1:1 移植自 `onnx_infer.py` | 中（金标准对比） |
| `Sampler` | `core.infer` | top_k / top_p / temperature / repetition penalty | 易 |
| `InferenceLoop` | `core.infer` | 每帧调度 4 个 ORT session 调用，发 `Flow<PcmChunk>` | 中 |
| `AudioPlayer` | `core.audio` | `AudioTrack` STREAM 模式 + `PcmAccumulator` | 中（设备层） |
| `ModelDownloader` | `core.download` | OkHttp 并行下载 + 进度回调 + sha256 + 断点续传 | 中 |
| `MainViewModel` | `ui` | MVI State/Intent/Effect | 易 |
| `*Screen` (Compose) | `ui.screen` | 纯 UI | UI 测试可选 |

### 与 Python 端的对应

| Android 模块 | 对应 Python | 移植策略 |
|---|---|---|
| `Tokenizer` | `onnx_tts_utils.py:SPTokenizer` | 直接调 sentencepiece-android |
| `PromptBuilder` | `onnx_infer.py:PromptBuilder` | 逻辑 1:1 翻译；常量值（`im_start_token_id` 等）从 `config.json` 读 |
| `Sampler` | `onnx_tts_utils.py` 的采样函数 | 1:1 翻译 |
| `InferenceLoop` | `onnx_infer.py:OnnxTTSEngine.generate` | 1:1 翻译；NumPy → `OnnxTensor` + `LongArray/FloatArray` |
| `AudioPlayer` | `soundfile.write` | 完全替换为 AudioTrack |
| `TextNormalizer` | （Python 端没有，新增） | ICU + 规则 |

### 协程与线程模型

| 组件 | 调度器 | 备注 |
|---|---|---|
| `InferenceLoop` | `Dispatchers.Default` | CPU 重活 |
| `AudioPlayer.consume()` | `Dispatchers.IO` | `AudioTrack.write` 是阻塞 |
| `ModelDownloader` | `Dispatchers.IO` | 网络 IO |
| `ModelManager.loadAll()` | `Dispatchers.IO` | 首启在 splash 上等 |
| `MainViewModel` | `viewModelScope` | 不直接持有 `TtsEngine`，避免泄漏 |

`TtsEngine` 由 `Application` 持有单例（不上 Hilt）。

### 反压

`InferenceLoop` 用 `channelFlow` + `Channel.RENDEZVOUS`：

- 每帧 emit 一个 PCM chunk，下一帧解码会被 `AudioPlayer.write` 阻塞（直到 buffer 有空位）才继续推进
- 自然反压；不会出现"模型跑得比播放快、PCM 在内存堆爆"
- 也不会"跑得比播放慢导致 underrun"——RTF < 1，模型产 chunk 比播放消耗 chunk 快

### § 2 关键决策（追加进 § 4 决策表）

| # | 决定 | 替代方案 | 理由 |
|---|---|---|---|
| 11 | `ModelDownloader` 独立成模块，**不进 `TtsEngine`** | 进 `TtsEngine` 统一调度 | 仅 `DownloadScreen` 用一次；解耦更清晰 |
| 12 | 不上 Hilt；`TtsEngine` 由 `Application` 持有单例 | Hilt / Koin / 手写 ServiceLocator | A2 体量没必要 DI 框架 |
| 13 | 5 个 ORT session 在首屏 splash **一次性全部加载**，常驻内存 | 按需加载 / lazy | 启动后全程零延迟；MVP 重点是首帧 ms |

---

## § 3 / 8 — 数据流

### 主路径：用户按下「合成并播放」

```
┌─ UI (Main 线程) ──────────────────────────────────────────────┐
│  用户点击 "合成并播放"                                          │
│        │                                                     │
│        ▼  Intent.Synthesize(text, locale)                    │
│  MainViewModel.onIntent()                                    │
│        │                                                     │
│        ▼  state.copy(phase = Synthesizing, t0 = now())       │
│  emit StateFlow → Compose 重组（按钮变 "停止"，状态行清空）       │
└─────────────┬────────────────────────────────────────────────┘
              │ launch on viewModelScope
              ▼
┌─ Default 线程 ────────────────────────────────────────────────┐
│  ttsEngine.synthesize(text, locale): Flow<PcmChunk>          │
│                                                              │
│  ┌─[L1] CommonNormalizer.normalize(text)                    │
│  ▼                                                           │
│  ┌─[L2] IcuNumberNormalizer.normalize(text, locale)         │
│  ▼                                                           │
│  ┌─ Tokenizer.encode(text) → IntArray                       │
│  ▼                                                           │
│  ┌─ PromptBuilder.buildContinuation(textIds)                │
│  │       → (input_ids: LongTensor[1,T,17], mask)             │
│  ▼                                                           │
│  ┌─ InferenceLoop.start():                                   │
│  │   • global_session.run(input_ids, mask, kv_in)           │
│  │       → hidden, kv_out                                    │
│  │   • 循环每帧 t = 0..max:                                   │
│  │       1) local_text.run(hidden) → text_logits            │
│  │       2) sampler.sample(text_logits)  → text_token       │
│  │       3) 若 text_token == EOS → break                    │
│  │       4) local_audio.run(hidden, text_token) → 16 codes  │
│  │       5) audio_decoder.run(codes, kv_dec_in)             │
│  │            → pcm_chunk (Float[~1920], kv_dec_out)        │
│  │       6) emit pcm_chunk     ← 这里就能听到声音             │
│  │       7) global_session.run(next_inputs, kv_in=kv_out)   │
│  │                                                           │
│  └─ channelFlow + RENDEZVOUS                                 │
└─────────────┬────────────────────────────────────────────────┘
              │ 反压：write 阻塞时上游自动暂停
              ▼
┌─ IO 线程 (audioPlayerScope) ──────────────────────────────────┐
│  flow.collect { chunk ->                                      │
│      if (firstChunk) {                                        │
│          mainViewModel.onFirstChunk(now() - t0)              │
│          audioTrack.play()                                    │
│      }                                                        │
│      audioTrack.write(chunk.bytes, …, WRITE_BLOCKING)         │
│      pcmAccumulator.append(chunk)                             │
│      mainViewModel.onProgress(chunk.duration)                 │
│  }                                                            │
│  audioTrack.stop()                                            │
│  mainViewModel.onSynthesisDone(rtf, totalMs)                  │
└──────────────────────────────────────────────────────────────┘
```

### 关键时刻

| 时刻 | 触发点 | 上报字段 | 用户感知 |
|---|---|---|---|
| `t0` | `onIntent(Synthesize)` 进入 ViewModel | – | 按钮变"停止" |
| `t_first_chunk` | `audioTrack.write(firstChunk)` **之前** | `firstChunkMs = t_first_chunk - t0` | 听到第一声 |
| `t_done` | `InferenceLoop` 完成 | `totalSynthMs`, `rtf = totalSynthMs / audioDurationMs` | 按钮回到"合成并播放" |

状态行显示：`首帧 82 ms · RTF 0.31 · INT8`。

### 旁路：保存 wav

`PcmAccumulator` 在播放线程同步累加，合成结束时**仅**保留在内存的 `LastResult(pcm, params)` 对象里。**不自动落盘**。
- 点 ↓ 保存：调用一次 `WavWriter.write(downloadsFile, pcm, 24000)` → 落到 `Downloads/` 并通过 `MediaScannerConnection` 通知系统
- 点 ⤴ 分享：先写到 cache → `FileProvider` + `Intent.ACTION_SEND` 拉起分享面板

### 取消与重入

| 触发动作 | 行为 |
|---|---|
| 合成中切换语种下拉 | **不打断当前播放**；新 locale 仅影响下一次合成的规范化与示例填充 |
| 合成中编辑文本框 | **不打断当前播放**；新文本仅在下一次合成生效 |
| 合成中点击"▶ 合成"（变成"再合成"按钮态） | ① 取消合成协程 ② `audioTrack.flush()` ③ 用当前文本+locale 立即开始新合成 |
| 合成中点击"⏹ 停止" | ① 取消合成协程 ② `audioTrack.flush()` ③ 按钮回到"▶ 合成并播放" |

**设计哲学**：只有用户**显式**触发的"合成 / 停止"才会动当前播放；其他操作（切语种、改文本）都是"为下一次准备"，让当前播放自然进行。

合成中按钮文案随状态切换：
- 空闲：`▶ 合成并播放`
- 合成/播放中：`⏹ 停止`（次要按钮）+ `▶ 重新合成`（主要按钮，按下即取消旧的开始新的）

→ 合成中**两个按钮都显示**，用户能选择"停了就停了"还是"换新内容直接重合"。

### 启动数据流

```
App.onCreate()  (≈ 100 ms 冷启动后能看到 UI)
  ├─ ttsEngine = TtsEngine(applicationContext)   // 不加载模型
  └─ launch IO {
       if (modelDir.exists() && manifest.valid()) {
           ttsEngine.preload()        // 后台加载 5 个 session
       } else {
           navigate(DownloadScreen)
       }
     }

MainScreen 立即显示
  · 文本框 / 语种下拉 / 示例按钮：可用
  · 合成按钮：禁用 + 文案 "加载模型中…" + 圆形进度
  · 状态行：" 加载模型中… (NN / 165 MB) "
            ↑ 5 个 session 加载粒度的进度

preload 完成 (≈ 500-800 ms)
  · 合成按钮：可用，文案 "▶ 合成并播放"
  · 状态行：清空

首次合成 0 额外等待
```

`preload()` 时长由 5 次 `OrtSession.create()` 主导，全部冷启动成本一次性付清；之后所有合成都复用 session。

### 错误传播

```
异常 (ModelManager / Inference / Audio)
       │
       ▼
ttsEngine 内部 catch → throw 或 emit Result.Error
       │
       ▼
ViewModel collect 时 catch → State(phase = Error, msg, debugTrace)
       │
       ▼
Compose 显示 Snackbar（友好文案）+ "查看详情" 展开堆栈
```

| 错误类 | 示例 | UX |
|---|---|---|
| 模型缺失 / 校验失败 | sha256 mismatch / 文件少 | 跳回 `DownloadScreen`，提示"请重新下载" |
| ORT 运行时错误 | session.run 抛 | Snackbar + "查看详情" |
| 文本为空 | trim 为空 | 按钮置灰，提示"请先输入文字" |
| 文本超长 | > 300 字 | 自动截断 + 一次性 Toast |
| 音频设备占用 | AudioTrack 创建失败 | Snackbar，引导用户检查通话/录音 App |

### § 3 关键决策（追加进 § 4 决策表）

| # | 决定 | 替代方案 | 理由 |
|---|---|---|---|
| 14 | 切换语种、编辑文本均**不打断**当前播放；仅"▶ 合成"或"⏹ 停止"才打断 | 切语种自动重合成 / 编辑自动重合成 | 显式触发才动播放，符合用户预期；避免切语种意外吞掉正在听的句子 |
| 15 | PCM 在内存累积成 `LastResult`，仅在用户按"保存/分享"时落盘 | 自动落盘到 cache | 避免试听 10 次产生 10 个临时 wav |
| 16 | 启动后立即显示主屏；preload 后台进行；合成按钮在 preload 完成前显示"加载中" | splash 等 preload / 进主屏后再 preload | 主屏可见 ≤ 100 ms；模型加载与用户输入并行 |

---

## § 4 / 8 — 关键决策汇总

完整 26 条决策记录的"权威源"在 `android/DEVLOG.md`（带日期，会随 V1.1/V2 持续追加）。本节为定稿时的快照，按主题分组。

### 4.1 形态与边界

| # | 决定 |
|---|---|
| 2 | App 定位 A2（极简 Demo + 导出 wav + 语种快选） |
| 4 | 语种下拉保留，但只控制示例 + 规范化 locale，不传给模型 |
| 10 | 文本上限 300 字，截断 + Toast |
| 14 | 切语种、编辑文本不打断当前播放；仅"▶ 合成"或"⏹ 停止"才打断 |
| 22 | 锁定竖屏 + 跟系统主题，不做横屏适配 |
| 26 | MVP 不开 PR 到上游，只在 fork `feat/android-onnx-demo` 分支自用 |

### 4.2 技术栈与构建

| # | 决定 |
|---|---|
| 6 | Kotlin + Jetpack Compose；MVI + StateFlow；不上 Hilt |
| 7 | minSdk 26 / targetSdk 34；arm64-v8a only |
| 12 | 不上 Hilt；`TtsEngine` 由 `Application` 持有单例 |
| 17 | 包名 `com.afun.mosstts` |
| 18 | MVP 单 Gradle module（只 `:app`） |
| 19 | 20 语种示例文本放 `core/i18n/Examples.kt`，不放 strings.xml |
| 20 | sentencepiece-android AAR 优先；NDK 自建兜底 |

### 4.3 模型与下载

| # | 决定 |
|---|---|
| 1 | tokenizer 用 sentencepiece-android JNI，不嵌入 ONNX |
| 3 | 模型分发：adb push（开发） + GitHub Releases 下载（用户） |
| 11 | `ModelDownloader` 独立成模块，不进 `TtsEngine` |
| 13 | 5 个 ORT session 在首屏 splash 一次性全部加载，常驻内存 |
| 16 | 启动后立即进主屏；preload 后台进行；合成按钮分阶段可用 |
| 23 | 下载并发数 = 2 |
| 24 | 模型源 GitHub Releases `tag=onnx_model`，URL 模板写在 `DownloadManifest.kt` |
| 25 | MVP sha256 由 Android 端硬编码；V1.1 改"读 manifest.json 字段" |

### 4.4 推理与音频

| # | 决定 |
|---|---|
| 5 | 真流式 AudioTrack STREAM 播放 |
| 15 | PCM 仅在用户按"保存/分享"时落盘，不自动写文件 |
| 21 | MVP 不接 `AudioFocus`；来电中断后引导用户重试 |

### 4.5 文本规范化

| # | 决定 |
|---|---|
| 8 | MVP L1 通用清理 + L2 ICU 数字读法；V1.1 补 L3 中英日俄 |
| 9 | 规范化默认开启，设置里给"原始文本"开关 |

---

## § 5 / 8 — 目录结构

### 仓库整体（仅 `android/` 子目录新增）

```
MOSS-TTS-Nano/                     ← 现有 Python 工程，不动
├── android/                        ← 本特性新增，独立 Gradle 工程
│   ├── DEVLOG.md                   ← 长期活文档
│   ├── README.md                   ← Android App 说明（怎么 build / push 模型 / 跑）
│   ├── .gitignore                  ← /build /local.properties .idea/ *.apk
│   ├── build.gradle.kts            ← 根 Gradle
│   ├── settings.gradle.kts
│   ├── gradle.properties
│   ├── gradlew / gradlew.bat
│   ├── gradle/wrapper/...
│   └── app/                        ← 唯一的 Gradle module（单 module）
│       ├── build.gradle.kts
│       ├── proguard-rules.pro
│       └── src/
│           ├── main/
│           │   ├── AndroidManifest.xml
│           │   ├── kotlin/com/afun/mosstts/
│           │   │   ├── App.kt                          ← Application，持单例 TtsEngine
│           │   │   ├── MainActivity.kt                  ← Compose host
│           │   │   │
│           │   │   ├── core/
│           │   │   │   ├── model/
│           │   │   │   │   ├── ModelManager.kt
│           │   │   │   │   ├── Manifest.kt              ← data class + JSON 解析
│           │   │   │   │   └── ModelConfig.kt           ← 解析 config.json
│           │   │   │   ├── text/
│           │   │   │   │   ├── Tokenizer.kt
│           │   │   │   │   ├── TextNormalizer.kt        ← 接口
│           │   │   │   │   ├── CommonNormalizer.kt      ← L1
│           │   │   │   │   ├── IcuNumberNormalizer.kt   ← L2
│           │   │   │   │   ├── LangNormalizer.kt        ← L3 接口（V1.1 用）
│           │   │   │   │   └── NormalizerPipeline.kt
│           │   │   │   ├── infer/
│           │   │   │   │   ├── PromptBuilder.kt
│           │   │   │   │   ├── Sampler.kt
│           │   │   │   │   ├── SamplingConfig.kt
│           │   │   │   │   ├── KvCache.kt               ← 状态管理 + StateSpec 解析
│           │   │   │   │   └── InferenceLoop.kt
│           │   │   │   ├── audio/
│           │   │   │   │   ├── AudioPlayer.kt           ← AudioTrack STREAM
│           │   │   │   │   ├── PcmAccumulator.kt
│           │   │   │   │   ├── PcmChunk.kt
│           │   │   │   │   └── WavWriter.kt
│           │   │   │   ├── download/
│           │   │   │   │   ├── ModelDownloader.kt
│           │   │   │   │   ├── DownloadManifest.kt      ← 写死 URL + sha256 表
│           │   │   │   │   └── Sha256Verifier.kt
│           │   │   │   ├── tts/
│           │   │   │   │   ├── TtsEngine.kt             ← 顶层门面
│           │   │   │   │   ├── SynthRequest.kt
│           │   │   │   │   ├── SynthResult.kt
│           │   │   │   │   └── LastResult.kt
│           │   │   │   └── i18n/
│           │   │   │       ├── LanguageOption.kt        ← 20 项 enum
│           │   │   │       └── Examples.kt              ← 20 条示例文本（决策 #19）
│           │   │   │
│           │   │   ├── ui/
│           │   │   │   ├── theme/                       ← Material3 主题
│           │   │   │   │   ├── Color.kt
│           │   │   │   │   ├── Type.kt
│           │   │   │   │   └── Theme.kt
│           │   │   │   ├── nav/AppNavigation.kt         ← StateFlow 驱动，零依赖
│           │   │   │   ├── viewmodel/
│           │   │   │   │   ├── MainViewModel.kt
│           │   │   │   │   ├── MainState.kt
│           │   │   │   │   ├── MainIntent.kt
│           │   │   │   │   ├── MainEffect.kt
│           │   │   │   │   └── DownloadViewModel.kt
│           │   │   │   └── screen/
│           │   │   │       ├── MainScreen.kt
│           │   │   │       ├── DownloadScreen.kt
│           │   │   │       └── AboutScreen.kt
│           │   │   │
│           │   │   └── util/
│           │   │       ├── Logger.kt
│           │   │       └── PerfStopwatch.kt
│           │   │
│           │   └── res/
│           │       ├── values/strings.xml               ← 默认（英文）
│           │       ├── values-zh/strings.xml            ← 仅 UI 字符串
│           │       ├── values-ja/strings.xml
│           │       ├── values-ko/strings.xml
│           │       ├── drawable/
│           │       │   ├── ic_launcher_foreground.xml
│           │       │   └── ic_launcher_background.xml
│           │       └── mipmap-*/ic_launcher.png
│           │
│           ├── test/                                    ← JVM 单测
│           │   └── kotlin/com/afun/mosstts/
│           │       ├── core/text/CommonNormalizerTest.kt
│           │       ├── core/text/IcuNumberNormalizerTest.kt
│           │       ├── core/infer/PromptBuilderTest.kt
│           │       ├── core/infer/SamplerTest.kt
│           │       └── core/download/Sha256VerifierTest.kt
│           │
│           └── androidTest/                             ← 仪器测试（需要真机/模拟器）
│               └── kotlin/com/afun/mosstts/
│                   └── EndToEndSmokeTest.kt             ← 完整一次合成（需先 push 模型）
│
├── docs/plans/                     ← 已存在
│   └── 2026-04-16-android-onnx-demo-design.md
│
└── （现有 Python 文件不动）
```

### 几点说明

#### 包名
`com.afun.mosstts` —— fork 维护者命名空间，简洁。

#### Gradle 模块化？
**不分 module**。MVP 体量约 2-3k 行 Kotlin，单模块 `:app` 编译快、IDE 索引快、新人上手成本低。等达到 V2 + voice clone 才考虑拆 `core` / `ui` 双模块。

#### 资源文件
- `res/values*/strings.xml` 只放 **UI 字符串**（按钮文案、错误提示、关于页文案）。
- **20 语种的"示例文本"不放 strings.xml**，放在 `core/i18n/Examples.kt` 里，因为它们是 *demo 内容* 不是 *UI 文案*；放代码里方便集中管理 + 单测。

#### sentencepiece 库怎么进
按优先级尝试：

| 方式 | 体积 | 难度 |
|---|---|---|
| GitHub 上维护中的 SentencePiece-Android AAR | 小（~600 KB） | 低 |
| 自己 NDK 编译官方 sentencepiece | 最小 | 高 |
| Maven `org.tensorflow:tensorflow-lite-support`（自带 SP） | 大 | 低 |

→ 实施时先尝试方案一，找不到合适的就退到方案二。决策最终值在 #20。

#### onnxruntime 怎么进
Maven 中央：
```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
```
官方 AAR 已经带 arm64-v8a + armeabi-v7a + x86_64；用 `packagingOptions` 排除 armv7 与 x86_64，把包体压到只含 arm64。

#### `.gitignore`
仓库根 `.gitignore` 已有。在 `android/.gitignore` 追加：
```
/build
/app/build
/.gradle
/local.properties
.idea/
*.apk
*.aab
captures/
```

#### Gradle wrapper
跟随项目，包含 `gradlew` 和 `gradle/wrapper/gradle-wrapper.jar`。

### § 5 关键决策（追加进 § 4 决策表）

| # | 决定 | 替代方案 | 理由 |
|---|---|---|---|
| 17 | 包名 `com.afun.mosstts` | `com.openmoss.tts` | fork 维护者命名空间，简洁 |
| 18 | MVP 阶段单 Gradle module（只 `:app`） | 多 module（`:app` + `:core` + ...） | 2-3k 行 Kotlin，分模块的增量编译收益 ≈ 0 |
| 19 | 20 语种示例文本放 `core/i18n/Examples.kt`，不放 strings.xml | 全部 strings.xml | demo 内容非 UI 文案；集中管理 + 单测友好 |
| 20 | sentencepiece-android AAR 优先；找不到合适的退 NDK 自建 | onnxruntime-extensions / 纯 Kotlin | AAR 体积最小；NDK 是兜底；最终选项实施时锁 |

---

## § 6 / 8 — 错误处理与边界

### 错误模型

所有可恢复错误用 `sealed class TtsError` 表达，**不抛裸异常给 UI 层**：

```kotlin
sealed class TtsError(val userMsg: String, val devTrace: String? = null) {
    object EmptyText           : TtsError("请先输入文字")
    object TextTruncated       : TtsError("文本超过 300 字，已自动截断")
    object ModelMissing        : TtsError("模型文件未找到，请先下载")
    data class ModelChecksumFailed(val file: String) : TtsError("模型校验失败：$file，请重新下载")
    object ModelLoadFailed     : TtsError("模型加载失败，可能内存不足")
    object AudioDeviceBusy     : TtsError("无法占用音频设备，请检查通话/录音 App")
    data class InferenceCrash(val msg: String, val trace: String) : TtsError("合成出错", trace)
    data class DownloadFailed(val file: String, val cause: String) : TtsError("下载失败：$file（$cause）")
    object NoNetwork           : TtsError("没有网络连接")
    object DiskFull            : TtsError("存储空间不足")
}
```

Engine 层返回 `Result<T, TtsError>`，ViewModel `collect` 时 `onFailure → state.copy(phase = Error, error = ...)`。

### UX 反馈层级

| 严重度 | 反馈方式 | 可恢复 |
|---|---|---|
| info | Toast / Snackbar 短提示 | 是；自动消失 |
| warn | Snackbar + Action 按钮 | 是；用户操作 |
| error | Snackbar + "查看详情" → ErrorDialog | 视情况 |
| fatal | 跳回 `DownloadScreen` 或 EmptyState | 必须用户操作 |

| `TtsError` | 严重度 | UX |
|---|---|---|
| `EmptyText` | info | 按钮置灰 + Tooltip |
| `TextTruncated` | info | Toast 一次性 |
| `ModelMissing` | fatal | 直接跳 `DownloadScreen` |
| `ModelChecksumFailed` | fatal | 删坏文件 → 跳 `DownloadScreen`，提示"模型已损坏" |
| `ModelLoadFailed` | error | Snackbar + "重试"按钮，二次失败建议"清理缓存重启" |
| `AudioDeviceBusy` | warn | Snackbar + "重试"按钮 |
| `InferenceCrash` | error | Snackbar + "查看详情"（含 dev trace） |
| `DownloadFailed` | warn | DownloadScreen 内重试该文件（断点续传） |
| `NoNetwork` | warn | DownloadScreen 显示"无网络"，恢复后自动重试 |
| `DiskFull` | error | DownloadScreen 弹框，提示需 ≥ 200 MB 可用空间 |

### 关键边界条件

#### 文本输入

| 边界 | 处理 |
|---|---|
| 空字符串 / 全空白 | 按钮置灰，`EmptyText` |
| `length > 300` | 截断到 300，emit `TextTruncated`（info）；继续合成 |
| 含控制字符（`\u0000–\u001f`，除 `\n`） | L1 normalizer 直接剔除 |
| 含零宽字符（`\u200b–\u200f`，`\ufeff`） | L1 normalizer 剔除 |
| 极端 emoji 序列 | tokenizer 字节 fallback；可能合成无声音；不算错误 |
| RTL 文字（阿拉伯/希伯来） | 不做特殊处理；ICU 数字读法照走 locale |

#### 推理过程

| 边界 | 处理 |
|---|---|
| 单步推理 > 5 秒 | InferenceLoop 内置 step timeout，记 `InferenceCrash("step_timeout")` |
| max_new_frames（300 帧 ≈ 12 秒）用尽未 EOS | 强制结束，warn `max_frames_reached`；播放正常进行；状态行尾巴加 ⚠ |
| ORT `OrtException` | catch → `InferenceCrash` |
| OOM | catch `OutOfMemoryError` → `ModelLoadFailed`，提示重启；不试图恢复 |

#### 音频播放

| 边界 | 处理 |
|---|---|
| AudioTrack 创建失败 | `AudioDeviceBusy` |
| 播放中切到其他 App / 锁屏 | 不暂停（demo 简单点；后台继续播完） |
| 来电 | **MVP 不接 `AudioFocus`**；电话 App 强制接管设备，下次按合成可能 `AudioDeviceBusy`，引导重试 |
| AudioTrack underrun | log warn 不视为错误；下一帧追上 |

#### 下载

| 边界 | 处理 |
|---|---|
| 单文件中断（断电/切网） | OkHttp Range 续传；恢复后从 `length` 处接着下 |
| sha256 不匹配 | 删除该文件，重下一次；二次失败 `DownloadFailed` |
| 用户切到后台 | 下载继续（IO 协程；MVP 不开 foreground service，OS 杀进程则丢失） |
| 并发数 | **2**（避免吃光带宽 + 易于观察进度） |
| 磁盘空间不足 | 下载前 `getUsableSpace() ≥ 200 MB`；不够直接 `DiskFull` |
| 已存在且 sha256 对得上 | 跳过，进度直接置 100% |

#### 模型与运行时

| 边界 | 处理 |
|---|---|
| `manifest.json` 缺失某文件 | `ModelMissing`，跳下载 |
| `manifest.json` schema 版本不匹配 | `ModelLoadFailed`，提示"版本不兼容，请更新 App 或重下" |
| `manifest.json` 多余文件 | 不报错（向前兼容） |
| 设备 ABI 不是 arm64 | App 装不上（manifest 限制） |
| Android < 8.0 | App 装不上（minSdk 26） |

### 模型下载源（已确认）

- **Release**：[`AFun9/MOSS-TTS-Nano @ tag onnx_model`](https://github.com/AFun9/MOSS-TTS-Nano/releases/tag/onnx_model)
- **URL 模板**：`https://github.com/AFun9/MOSS-TTS-Nano/releases/download/onnx_model/<filename>`
- **Android 需要下载的 9 个文件**（INT8 子集 + 元数据）：

| 文件 | 大小 | 角色 |
|---|---:|---|
| `manifest.json` | < 1 KB | Bundle 描述 |
| `config.json` | < 1 KB | 推理参数 |
| `tokenizer.model` | 0.5 MB | SentencePiece tokenizer |
| `audio_decoder_state_spec.json` | 6 KB | KV cache 布局 |
| `audio_encoder_int8.onnx` | 15.7 MB | 提示音频编码 |
| `local_decoder_text_int8.onnx` | 7.1 MB | 文本/EOS 头 |
| `local_decoder_audio_int8.onnx` | 19.7 MB | 16-codebook 头 |
| `audio_decoder_int8.onnx` | 11.5 MB | 每帧解码 |
| `global_transformer_int8.onnx` | 111.0 MB | 12 层 GPT2 主体 |
| **合计** | **≈ 165 MB** | |

**FP32 版本不下载**（只在导出端做 sanity check 用）。

### sha256 来源策略

当前 `manifest.json` 里**没有** sha256 字段。MVP 期：
- **`DownloadManifest.kt` 在 Android 端硬编码 9 个文件的 sha256**
- 校验失败 → 删文件 → 重下；二次失败 → `DownloadFailed`

V1.1 待办（→ DEVLOG TODO）：在 `export_onnx.py` 写 manifest 时计算并写入 sha256；Android 改成"先下 manifest.json，再用其中 sha256 校验其他文件"，避免 App 端硬编码（每次重新导出模型都要改 App）。

### 取消与生命周期

- Activity `onDestroy` → ViewModel `onCleared` → cancel 当前合成 job + flush AudioTrack
- **不 release session**：`TtsEngine` 是 Application 级单例，Activity 销毁时只取消合成、停播放；session 留着等下个 Activity 复用
- App 进后台：什么都不做；OS 杀进程时 ORT 自动释放
- 配置变更：**禁用屏幕旋转 + 跟系统主题**，不需要处理 Activity 重建

### 日志策略

```
tag      : "MossTts"
等级     : DEBUG（debug build）/ INFO（release build）
不打印   : 用户输入文本（隐私）；模型权重；任何 PCM 内容
打印     : 时间戳 + 各阶段耗时 ms + 错误堆栈
```

性能关键点的耗时记录格式：

```
preload(): manager=120 ms total=586 ms
synth: t0=0 norm=2 tokenize=3 prompt=1 first_global=42 first_chunk=78 done=2400 rtf=0.31
```

这些数字直接喂到 DEVLOG 的"性能基线"表。

### § 6 关键决策（追加进 § 4 决策表）

| # | 决定 | 替代方案 | 理由 |
|---|---|---|---|
| 21 | MVP 不接 `AudioFocus`；来电中断后引导用户重试 | 接 AudioFocus 自动暂停/恢复 | 状态机复杂；demo 阶段不必要 |
| 22 | 锁定竖屏 + 跟系统主题；不做横屏适配 | 横竖屏自适应 | 一屏 demo App 没必要做布局自适应 |
| 23 | 下载并发数 = 2 | 全并行 9 / 严格串行 | 易控带宽 + 进度观察；INT8 165 MB 在 50 Mbps 下 ≈ 30 秒 |
| 24 | 模型托管 GitHub Releases `tag=onnx_model`（[link](https://github.com/AFun9/MOSS-TTS-Nano/releases/tag/onnx_model)），URL 模板写在 `DownloadManifest.kt` | HF / ModelScope / 自托管 | MVP 期零运维；V1.1 加镜像源切换 |
| 25 | MVP sha256 由 Android `DownloadManifest.kt` 硬编码；V1.1 改"读 manifest.json 字段" | 完全不校验 / 立刻改 export 端 | 避免侵入 Python；重导模型时手动同步 sha256 表 |

---

## § 7 / 8 — 测试计划

### 测试金字塔

```
                    ▲ EndToEnd Smoke (1 个)
                   ╱ ╲ androidTest, 真机/模拟器
                  ╱   ╲ "输入 → 听到声音"
                 ╱─────╲
                ╱ Inst  ╲ Instrumentation (5-10 个)
               ╱  test   ╲ androidTest, AudioPlayer / FileIO / Downloader
              ╱───────────╲
             ╱  JVM Unit   ╲ test/, 60-100 个
            ╱ Normalizer    ╲ Sampler / Sha256 / PromptBuilder
           ╱  Tokenizer      ╲
          ╱___________________╲
```

### JVM 单测（`src/test/`，跑在 `gradlew test`）

| 测试类 | 覆盖目标 | 大致用例数 |
|---|---|---:|
| `CommonNormalizerTest` | 控制字符 / 零宽 / 全半角 / 空格折叠 / 截断 | 10 |
| `IcuNumberNormalizerTest` | 整数 / 小数 / 负数 / 百分号 / 序数；中英日俄各覆盖 | 30 (4 lang × 7-8 case) |
| `PromptBuilderTest` | continuation prompt 张量值与 Python `onnx_infer.py` 对齐（**金标准对比**） | 5 |
| `SamplerTest` | top_k / top_p 边界、温度 0 退化为 argmax、repetition penalty 生效 | 10 |
| `Sha256VerifierTest` | 已知输入向已知 hash | 3 |
| `ManifestParserTest` | 正常 / 缺字段 / 多余字段 / 版本不兼容 | 5 |
| `KvCacheTest` | StateSpec 解析 / 增量拼接形状正确 | 5 |
| `LangNormalizerStubTest` | V1.1 占位（接口存在，默认 no-op） | 1 |
| **合计** | | **~70** |

#### 金标准对比怎么做

`PromptBuilderTest` 这种"逻辑必须 1:1 与 Python 一致"的测试，**离线生成 fixture**：

```bash
python tools/dump_prompt_fixtures.py --texts "你好" "Hello world" "こんにちは" \
    --out android/app/src/test/resources/prompt_fixtures.json
```

每条形如：

```json
{
  "text": "你好",
  "input_ids_shape": [1, 38, 17],
  "input_ids_first_8": [220, 14471, ...],
  "input_ids_last_8": [...],
  "input_ids_sum": 12345,
  "input_ids_xor": "0xabcdef..."
}
```

Kotlin 测试加载 fixture，比对 shape + 关键采样位 + 整体 hash —— **不存全张量**（避免 fixture 太大）。

固定 5-10 条 fixture 文本，覆盖 4 个语种 + 边界（极短 / 极长 / 含数字 / emoji）。

### 仪器测试（`src/androidTest/`，需要 emulator/device）

| 测试类 | 覆盖目标 |
|---|---|
| `AudioPlayerTest` | AudioTrack 创建 → write 1 秒 sin 波 → flush；只验状态机不崩 |
| `WavWriterTest` | 写 100 ms PCM → 读回 → 头部 RIFF/fmt/data 字段对得上 |
| `Sha256OnFileTest` | 从 cacheDir 读真实文件 → hash 与 stdin 算出来一致 |
| `ModelManagerLoadTest` | （需要 push 模型）从 `getExternalFilesDir/onnx_export` 加载 → 5 session 创建成功 |
| `EndToEndSmokeTest` | （需要 push 模型）输入"你好" → 调 `TtsEngine.synthesize` → 收到 ≥ 1 个 PcmChunk → AudioTrack 不报错 |

仪器测试**只在本地真机**手动跑；CI 只跑 JVM 单测。

### 性能基准（独立的 `BenchmarkActivity`，可选）

不引入 `Benchmark` 框架。直接：

```
BenchmarkActivity:
  · 加载模型（warmup 一次）
  · 跑 10 段固定文本（中英日俄各 2-3 段）
  · 每段记录 firstChunkMs / totalMs / rtf
  · 写到 Logcat + .csv 落到 Downloads/
```

每次重大优化后人工跑一次，把数字粘进 DEVLOG 性能基线表。

### 手动测试矩阵（每次发版前过一遍）

| 场景 | 期望 |
|---|---|
| 全新装 → 进 DownloadScreen → 下完 → 进主屏 → 输入"你好" → 合成 | 30 秒内能听到"你好" |
| 已下载 → 杀进程重开 → 输入英文 → 合成 | 跳过下载，直接进主屏 |
| 合成中切语种 / 编辑文本 | 当前播放不打断 |
| 合成中按"再合成" | 当前停 → 立即开始新合成 |
| 合成中按"停止" | 当前停 → 按钮回到合成 |
| 输入 350 字 | 截断 + Toast |
| 输入空 | 按钮置灰 |
| 输入 emoji 序列 | 不崩 |
| 网络断开下载 → 恢复 | 续传，不重新从 0 |
| 同名文件已存在 + sha256 对得上 | 跳过该文件 |
| 同名文件已存在 + sha256 错 | 删掉重下 |
| 来电中 | 当前播放被系统抢；下次合成 `AudioDeviceBusy` 可恢复 |
| 锁屏 | 继续播完 |
| 切到后台再回来 | 状态保持；播放继续/已结束 |
| 卸载 → 重装 | 模型不在 `getExternalFilesDir`（被系统清），重新走 DownloadScreen |

### CI 计划

`.github/workflows/android.yml`（在 fork 仓库下）：

```yaml
jobs:
  unit-test:
    runs-on: ubuntu-latest
    steps:
      - actions/checkout
      - actions/setup-java (17)
      - cache gradle
      - working-directory: android
        ./gradlew testDebugUnitTest --no-daemon
  build-apk:
    needs: unit-test
    steps:
      - 同上
      - ./gradlew assembleDebug
      - upload-artifact: app/build/outputs/apk/debug/*.apk
```

不跑 instrumentation；不上签名；产出 debug APK 供 fork 维护者下载试装。

---

## § 8 / 8 — 里程碑、风险与文档同步

### 里程碑

| 里程碑 | 目标 | 交付物 | 估时 |
|---|---|---|---|
| **M0** Bootstrap | Gradle 工程 + Compose Hello World 跑起来 | 装包能开 | 0.5 d |
| **M1** Model & Tokenizer | ModelManager + Tokenizer + Sha256 + 单测 | JVM 单测绿 | 1 d |
| **M2** Inference Loop | PromptBuilder + Sampler + InferenceLoop（不接播放，dump PCM） | 与 `onnx_infer.py` 对齐生成 wav | 2 d |
| **M3** Audio Streaming | AudioPlayer + 反压 + 状态行 | 端上能听见 | 1 d |
| **M4** UI + Normalizer | MainScreen + 语种下拉 + L1+L2 normalizer + 示例库 | 完整可玩 | 1.5 d |
| **M5** Download | DownloadScreen + ModelDownloader + 9 文件并发 | 全新装能跑通 | 1.5 d |
| **M6** Polish & Bench | 错误处理收口 + 关于页 + 性能基线表填充 + README | v0.1.0 可用 | 1 d |
| **总计** | | **MVP** | **8.5 d** |

每完成一个里程碑：

1. DEVLOG `Done` 表加一行 + commit hash
2. 对应 Backlog 项划掉
3. 性能基线（M3 起）填新数字
4. push feature 分支到 fork
5. 给你看一次确认再开下一个

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 找不到合适的 sentencepiece-android AAR | 中 | 中 | 退路：NDK 编译官方 sentencepiece，加 1 d |
| `OrtSession` 创建在中端机 > 1.5 s（远超 600ms 估算） | 低 | 中 | 测出来再决定；① 接受（splash 转动）② 模型再瘦身 |
| INT8 在某些设备上 ConvInteger 缺实现报错 | 低 | 高 | 已有处理经验；fallback 到 FP32 子集（global_transformer 仍用 INT8） |
| 真流式 AudioTrack 在低端机 underrun 严重 | 中 | 中 | 改决策 #5 退到 200 ms 攒缓冲（C 方案）；UI 不变 |
| GitHub Releases 国内下载慢（~100 KB/s） | 高 | 中 | DEVLOG 已记 V1.1 加 ModelScope 镜像 |
| sentencepiece JNI 与 ORT JNI 同进程 ABI 冲突 | 低 | 高 | 第一次合成成功就排除；若失败考虑 onnxruntime-extensions 嵌入 |
| Compose 在 minSdk 26 旧设备渲染卡顿 | 低 | 低 | 不针对优化；README 注明推荐 Android 11+ |
| `OutOfMemoryError`：global_transformer ~111 MB + 运行时 buffer，2GB RAM 设备临界 | 中 | 高 | manifest.json 启动检查可用 RAM；不够给提示并退出 |

### 文档同步约定

每个里程碑结束时由我同步以下文档（作为该里程碑 commit 的一部分）：

| 文档 | 同步内容 |
|---|---|
| `docs/plans/2026-04-16-android-onnx-demo-design.md` | **不动**（定稿后冻结）；如有重大设计变更，新建 `-rev1.md` |
| `docs/plans/2026-04-16-android-onnx-demo-plan.md` | writing-plans 阶段产出后，每个任务完成时只标记状态（不改任务内容） |
| `android/DEVLOG.md` | Done / Backlog / Known Issues / 性能基线 / Changelog 持续更新 |
| `android/README.md` | M5 完成后写：build / push 模型 / 跑应用的 3 步引导 |

### 完成定义 (Definition of Done)

MVP 视为完成，需同时满足：

- [ ] 7 个里程碑（M0–M6）全部 ✓
- [ ] JVM 单测全绿
- [ ] 至少一台真机跑过手动测试矩阵全部场景
- [ ] DEVLOG 性能基线表至少有 3 行数据（不同语种）
- [ ] `android/README.md` 写完
- [ ] 设计文档 (`-design.md`) 顶部状态从 DRAFT 改为 FROZEN
- [ ] 在 fork 仓库 `feat/android-onnx-demo` 分支 push 完所有 commits（**不开 PR 到上游**，自用）

### § 7 + § 8 关键决策（追加进 § 4 决策表）

| # | 决定 | 替代方案 | 理由 |
|---|---|---|---|
| 26 | MVP 不开 PR 到上游；只在 fork `feat/android-onnx-demo` 分支自用 | 开 PR 到上游 / 单独的 mini repo | 自用为主；避免给上游引入 Android 维护负担 |

---

## § 8 / 8 — 里程碑与风险

> **TODO**
