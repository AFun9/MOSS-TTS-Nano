# Android Demo · DEVLOG

> 长期活文档。每次"小阶段跑通"就更新一次（不是每次 commit 都更新）。
> 设计文档见 `docs/plans/2026-04-16-android-onnx-demo-design.md`。
> 实现计划见 `docs/plans/2026-04-16-android-onnx-demo-plan.md`（writing-plans 后产出）。

最后更新：2026-04-17（**M1 全部完成** — Tokenizer 走纯 Kotlin G 路径，64 fixture 与 Python sentencepiece byte-equal 对齐 ✓，43 unit tests 全绿）

---

## 🟢 已完成 (Done)

| 日期 | 内容 | Commit |
|---|---|---|
| 2026-04-16 | **M0 Bootstrap** — Gradle 8.13 wrapper / `:app` module / Compose Hello World / Material3 theme / 纯 vector adaptive launcher icon。`./gradlew :app:assembleDebug` 53 秒过；APK 8.3 MB（裸 Compose，无 native lib）。**真机验证 ✓**（adb install + am start 启动正常）。 | `ed98ac2`..`8b1bcaa` |
| 2026-04-17 | **M1 第 1 阶段（5/7 task）** — 加 `onnxruntime-android:1.20.0`（APK +17.5 MB native so → 26 MB）；切到 `kotlinx.serialization` 绕开 Android `org.json` stub（决策 #29）；TDD 完成 4 个核心类：`Manifest` / `ModelConfig`（修正 plan 的 `sample_rate=24000` 错误，实际是 `audio_tokenizer_sample_rate=48000`）/ `Sha256Verifier` / `ModelManager`。**22 unit tests 全绿**。`./gradlew :app:assembleDebug` ✓ 28 MB。剩 sentencepiece 依赖（M1.1b）+ Tokenizer JNI wrapper（M1.6）待决策。 | `471d746`..`136b593` |
| 2026-04-17 | **M1 完整收官（M1.6 Tokenizer · 纯 Kotlin G 路径）** — 决策 #30：放弃 sentencepiece JNI（无 ARM64 prebuilt + 维护成本），自己重写 SP-BPE。`tools/export_tokenizer_for_kotlin.py` 把 `tokenizer.model` 导出成 `assets/tokenizer_kotlin.json`（700 KiB · 16384 piece）+ 64 多语种金标准 fixture。Kotlin 三层：`SpNormalizer`（NFKC + 控制字符 + ZWJ/ZWNJ/LRM/RLM/BOM 处理 + ▁ 转义 + dummy prefix · 10 tests）／`BpeEncoder`（greedy max-score adjacent merge + UTF-8 byte_fallback · 9 toy tests）／`Tokenizer`（USER_DEFINED 特殊 token 优先切片，type=4 共 11 个）。**`TokenizerGoldenTest` 64/64 fixture 与 Python sp byte-equal 对齐 ✓**（中英日韩俄西法德阿泰越 + emoji + 半全角 + ZWJ/ZWNJ + `<\|im_start\|>` 混排）。**全 app 43 unit tests 全绿**。零 native 依赖。 | `7a940ab`..（待 commit） |

---

## 🟡 进行中 (In Progress)

| 任务 | 负责人 | 起始日期 | 备注 |
|---|---|---|---|
| **M2** Inference Loop · 与 `onnx_infer.py` 对齐 | Cursor + AFun9 | 2026-04-17 | M1 全部 7/7 完成；下一步：5 个 ORT session 加载 + PromptBuilder Kotlin 移植（用 `tools/dump_prompt_fixtures.py` 做金标准） |

---

## 🔵 新增 / 待启动 (Backlog)

| 优先级 | 项 | 计划阶段 |
|---|---|---|
| ~~P1~~ | ~~Gradle 工程初始化 + Compose 模板~~ | MVP — done in M0 |
| P1 | ONNX bundle 加载与 5 个 ORT session 创建 | MVP |
| P1 | 文本 → ONNX 推理 → PCM chunk pipeline | MVP |
| P1 | 流式 AudioTrack STREAM 播放 | MVP |
| P1 | 模型下载页（GitHub Releases，并行 + 进度 + sha256 + 断点续传） | MVP |
| P1 | 文本规范化 L1（通用清理） | MVP |
| P1 | 文本规范化 L2（ICU 数字读法） | MVP |
| P1 | 主界面（文本框 + 语种下拉 + 合成 + 进度 + 状态行 + 保存/分享） | MVP |
| P1 | 20 语种示例文本词条 | MVP |
| P2 | L3 中英日俄符号/缩写规则 | V1.1 |
| P2 | HuggingFace 镜像下载源 | V1.1 |
| P2 | ModelScope 镜像下载源（国内） | V1.1 |
| P2 | `export_onnx.py` 写 sha256 到 manifest（配 Android 端去硬编码） | V1.1 |
| P2 | **长文本切句 + 音色锁定**：ICU `BreakIterator` 切句 → 第 1 句 continuation → 截前 1.0 s 当 voice_clone reference → 句 2..N 用此 reference → 拼 PCM；上限 300 → ~5000 字；流水线（句 N 播放时合成句 N+1）。详见 Decision #27。 | V1.1 |
| P3 | Voice clone（录音 + UX 引导） | V2 |
| P3 | L4 日期/时间/货币规范化 | V2 |
| P3 | 历史记录 / 收藏 / 通知栏后台朗读 | V2+ |

---

## 🟠 待优化 (TODO / Refactor)

| # | 项 | 计划 |
|---|---|---|
| T1 | 在 `export_onnx.py` 计算并写入每文件 sha256 到 `manifest.json` | V1.1（届时 Android 端 `DownloadManifest.kt` 改成读 manifest 字段，去掉硬编码） |

---

## 🔴 已知问题 (Known Issues)

| # | 严重度 | 描述 | 处理 |
|---|---|---|---|
| K1 | 中 | 文本超过 **300 字会被截断**，截断后内容**完全不发声**（不是只播前 300 字然后跳过——后续字根本不送模型）。原因：单段合成的 KV cache 随 prompt + 生成长度线性增长，300 字是防 OOM 与单次合成时长 ≤ 12 s 的硬闸。 | MVP：截断 + Toast 提示（决策 #10）。V1.1：上"切句 + 音色锁定"方案（决策 #27 / Backlog P2），上限放宽到 ~5000 字。 |

---

## 📝 决策记录 (Decisions)

> ADR 精简版。任何超过 1 个备选方案的取舍都进表，哪怕当时没纠结过也写一行。

| # | 日期 | 决定 | 替代方案 | 理由 |
|---|---|---|---|---|
| 1 | 2026-04-16 | tokenizer 用 sentencepiece-android JNI，不嵌入 ONNX | onnxruntime-extensions / 纯 Kotlin 重写 | AAR 体积更小、调试更易、导出端零新依赖 |
| 2 | 2026-04-16 | App 定位 A2（极简 Demo + 导出 wav + 语种快选） | A0/A1/A3/B/C/D | 配合 ONNX PR 故事线；voice clone 留 V2 |
| 3 | 2026-04-16 | 模型分发：adb push + GitHub Releases 下载 | HF / ModelScope / assets / AAB | MVP 期零运维；后续可加镜像源切换 |
| 4 | 2026-04-16 | 语种下拉只控示例 + 规范化 locale，不传给模型 | 砍下拉 / 示例按钮 / 全无 | 兼顾 20 语种卖点与"模型自动识别"事实 |
| 5 | 2026-04-16 | 真流式 AudioTrack STREAM 播放 | 整段 MediaPlayer / 攒 200 ms 再播 | 80 ms 首帧的优化必须前端兑现 |
| 6 | 2026-04-16 | Kotlin + Compose，MVI + StateFlow，不上 Hilt | XML+Java / Compose+Hilt | A2 体量没必要 DI 框架 |
| 7 | 2026-04-16 | minSdk 26 / targetSdk 34，arm64-v8a only | 双 ABI / 更低 minSdk | 体积/性能换覆盖率 |
| 8 | 2026-04-16 | 文本规范化 MVP 走 L1 + L2（ICU 数字） | WeTextProcessing 移植 / 全不做 | ICU 零依赖、覆盖 20+ 语言数字读法 |
| 9 | 2026-04-16 | 规范化默认开启 + 设置开关"原始文本" | 永远开 / 永远关 | 方便对照测试 |
| 10 | 2026-04-16 | 文本上限 300 字 + 截断提示 | 1000 字 / 不限 | 防 OOM 与单句 10+ 秒等待 |
| 11 | 2026-04-16 | `ModelDownloader` 独立成模块，不进 `TtsEngine` | 进 `TtsEngine` 统一调度 | 仅 `DownloadScreen` 用一次；解耦更清晰 |
| 12 | 2026-04-16 | 不上 Hilt；`TtsEngine` 由 `Application` 持有单例 | Hilt / Koin / 手写 ServiceLocator | A2 体量没必要 DI 框架 |
| 13 | 2026-04-16 | 5 个 ORT session 在首屏 splash 一次性全部加载，常驻内存 | 按需 / lazy | 启动后全程零延迟；MVP 重点是首帧 ms |
| 14 | 2026-04-16 | 切换语种、编辑文本均不打断当前播放；仅"▶ 合成"或"⏹ 停止"才打断 | 切语种自动重合成 / 编辑自动重合成 | 显式触发才动播放，避免切语种意外吞掉正在听的句子 |
| 15 | 2026-04-16 | PCM 仅在用户按"保存/分享"时落盘，不自动写文件 | 自动落盘到 cache | 避免试听多次产生大量临时 wav |
| 16 | 2026-04-16 | 启动后立即进主屏；preload 后台进行；合成按钮分阶段可用 | splash 等 preload / 进主屏后再 preload | 主屏 ≤ 100 ms 可见；模型加载与用户输入并行 |
| 17 | 2026-04-16 | 包名 `com.afun.mosstts` | `com.openmoss.tts` | fork 维护者命名空间，简洁 |
| 18 | 2026-04-16 | MVP 单 Gradle module（只 `:app`） | 多 module（`:app` + `:core` ...） | 2-3k 行 Kotlin，分模块增量编译收益 ≈ 0 |
| 19 | 2026-04-16 | 20 语种示例文本放 `core/i18n/Examples.kt`，不放 strings.xml | 全部 strings.xml | demo 内容非 UI 文案；集中管理 + 单测友好 |
| 20 | 2026-04-16 | sentencepiece-android AAR 优先；NDK 自建兜底 | onnxruntime-extensions / 纯 Kotlin 重写 | AAR 体积最小；最终选项实施时锁 |
| 21 | 2026-04-16 | MVP 不接 `AudioFocus`；来电中断后引导用户重试 | 接 AudioFocus 自动暂停/恢复 | 状态机复杂；demo 阶段不必要 |
| 22 | 2026-04-16 | 锁定竖屏 + 跟系统主题；不做横屏适配 | 横竖屏自适应 | 一屏 demo 没必要做布局自适应 |
| 23 | 2026-04-16 | 下载并发数 = 2 | 全并行 9 / 严格串行 | 易控带宽 + 进度观察；INT8 165 MB 在 50 Mbps 下 ≈ 30 秒 |
| 24 | 2026-04-16 | 模型源 GitHub Releases `tag=onnx_model`，URL 写在 `DownloadManifest.kt` | HF / ModelScope / 自托管 | MVP 期零运维；V1.1 加镜像源切换 |
| 25 | 2026-04-16 | MVP sha256 由 Android 端硬编码；V1.1 改"读 manifest.json 字段" | 完全不校验 / 立刻改 export 端 | 避免侵入 Python；重导模型时手动同步 sha256 表 |
| 26 | 2026-04-16 | MVP 不开 PR 到上游；只在 fork `feat/android-onnx-demo` 分支自用 | 开 PR 到上游 / 单独 mini repo | 自用为主；避免给上游引入 Android 维护负担 |
| 27 | 2026-04-16 | **V1.1** 长文本走"句子切分 + 音色锁定"：ICU `BreakIterator` 按 locale 切句；第 1 句 continuation 模式合成（首帧 ~80 ms）；截前 1.0 s 作为后续句的 voice_clone reference；句 2..N 用 voice_clone 模式 + 该 reference；流水线（句 N 播放时后台合成句 N+1）；句间 ~30 ms 静音。每句独立 KV，生成完即销毁，单次合成 KV 上限 ≈ 10-15 MB 恒定。文本上限 300 → ~5000 字。 | ① 仅放宽上限到 600 字（不解决根本问题）<br>② 切句 + seed 锁定（**已验证不可行**：不同句文本 → step 0 hidden 不同 → 即使 RNG 状态一致，第一帧 audio token 也不同 → 音色跳变） <br>③ 滑窗 KV cache（会"忘记"前文，韵律破坏） | seed 不能保证音色对齐，因为模型没有显式 speaker embedding，音色是"第一帧采样的副产品"；voice_clone 的 reference 才是真正的"音色锚"。MVP 维持 300 字 + 截断（决策 #10）已能演示卖点；V1.1 加该方案是真"产品级长文本"。 |
| 28 | 2026-04-16 | **推理路径用 Kotlin + ORT Java API**（不上 C++/JNI）。MVP 用 Kotlin；M6 benchmark 后按实测决定：① 首帧 < 150 ms 不动；② 150-300 ms 走廉价优化（IoBinding / OnnxTensor 池 / DirectByteBuffer）；③ > 300 ms 再把 `InferenceLoop` 翻 C++（接口可替换，UI 零改动）。 | C++ + JNI 全栈 / 推理走 NDK 其余 Kotlin | C++ 路线工期 +50%（8.5d → ~14d）、JVM 单测做不了 prompt 金标准对比、调试难一个数量级。Kotlin JNI 开销 ~25-30%（每帧 ~20 ms），但模型计算本身 50 ms 主导，预期首帧 ~90-110 ms 仍 ≤ 150 ms 目标。"自用、可维护"压倒"极致性能"。 |
| 30 | 2026-04-17 | **Tokenizer 走 G 路径：纯 Kotlin 重写 SP-BPE，零 native**。`assets/tokenizer_kotlin.json`(700 KiB) 由 `tools/export_tokenizer_for_kotlin.py` 从 `tokenizer.model` 生成。`SpNormalizer` 实现 nmt_nfkc 子集（NFKC + 控制字符丢弃 + 空白折叠 + ZWNJ/LRM/RLM/BOM/bidi 字符丢弃 + ▁ 转义 + dummy prefix），`BpeEncoder` 实现 greedy max-score adjacent merge + UTF-8 byte_fallback。`Tokenizer` 顶层做 USER_DEFINED 特殊 token 切片（type=4，11 个：`<\|im_start\|>` 等），切片段内独立 normalize+bpe。`TokenizerGoldenTest` 64 个多语种 fixture 与 Python sentencepiece **byte-equal** 全过（中/英/日/韩/俄/西/法/德/阿/泰/越 + emoji + 半全角 + ZWJ/ZWNJ + 长段 + special token 混排）。 | A 嵌 ONNX onnxruntime-extensions（+15 MB so）/ B sentencepiece-android JNI（无 ARM64 prebuilt）/ C 自编 NDK build（+1d 工期 + 维护 native） | 零 native 依赖、APK 增量小（700 KiB JSON + ~10 KB Kotlin）、JVM 单测可直接跑、64 fixture 字节对齐保证未来重训不漂移。已知 approximation gap：未实现 SP `precompiled_charsmap` 完整字符表（仅手工补常见 zero-width / bidi 控制字符），若上游后续切换 normalizer 名或文本含未覆盖 Cf 字符可能漂移，金标准测试会立即报错。 |
| 29 | 2026-04-17 | **JSON 解析用 `kotlinx.serialization-json:1.6.3`**（`implementation` 进 main，APK +~700 KB） | ① Android 自带 `org.json` ② Robolectric ③ `org.json:json` Maven artifact 加 testImpl ④ Moshi / Jackson | Android 在 JVM 单测里把 `org.json.*` 全 stub 成抛 `RuntimeException("not mocked")`，且 mockable-android.jar 在 classpath 顺序上压过 testImplementation，导致 `org.json:json` 也救不了；Robolectric 启动 ~5s 拖慢 TDD；Moshi 需要 KSP/codegen 配置更重。`kotlinx.serialization` 是 Kotlin 官方零额外坑、零反射、纯 JVM，APK 体积可接受。一次配好，`Manifest` / `ModelConfig` / 后续 `audio_decoder_state_spec.json` 都直接复用。 |

---

## 📊 性能基线 (Benchmarks)

> 每次跑出新数字都进表。趋势比单点更有意义。

| 日期 | 设备 | 量化 | 文本（语种 + 长度） | 首帧 ms | RTF | 备注 |
|---|---|---|---|---:|---:|---|
| – | – | – | – | – | – | 待 MVP 跑通 |

---

## 🗓 Changelog

### v0.1.0 · _未发布_

- _MVP 进行中_

---

## 模板片段

提交新 Decision 时复制：

```
| N | YYYY-MM-DD | <一句话决定> | <替代> | <理由> |
```

提交新 Benchmark 时复制：

```
| YYYY-MM-DD | <设备> | <FP32/INT8> | <语种 + 字数/词数> | <ms> | <rtf> | <备注> |
```
