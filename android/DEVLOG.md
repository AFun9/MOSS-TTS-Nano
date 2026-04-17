# Android Demo · DEVLOG

> 长期活文档。每次"小阶段跑通"就更新一次（不是每次 commit 都更新）。
> 设计文档见 `docs/plans/2026-04-16-android-onnx-demo-design.md`。
> 实现计划见 `docs/plans/2026-04-16-android-onnx-demo-plan.md`（writing-plans 后产出）。

最后更新：2026-04-16

---

## 🟢 已完成 (Done)

_暂无。项目尚未开始实现。_

| 日期 | 内容 | Commit |
|---|---|---|
| – | – | – |

---

## 🟡 进行中 (In Progress)

| 任务 | 负责人 | 起始日期 | 备注 |
|---|---|---|---|
| Brainstorming + 设计文档 | 模型 + AFun9 | 2026-04-16 | § 1 已定，§ 2-8 进行中 |

---

## 🔵 新增 / 待启动 (Backlog)

| 优先级 | 项 | 计划阶段 |
|---|---|---|
| P1 | Gradle 工程初始化 + Compose 模板 | MVP |
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

_暂无。待实施开始后填充。_

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
