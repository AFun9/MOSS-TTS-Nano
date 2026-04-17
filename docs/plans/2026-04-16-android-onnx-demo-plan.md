# Android ONNX Demo Implementation Plan

> **For executing agent:** Implement task-by-task in order. Each task has Files / Steps / Commit. Tasks marked **TDD** follow Red-Green 5-step; tasks marked **UX/Config** follow 3-step (implement → manual verify → commit). Update `android/DEVLOG.md` after every milestone (M0–M6) per the design doc § 8.

**Goal:** Ship a Kotlin + Jetpack Compose Android app that drives the existing ONNX bundle directly, demonstrating ~80 ms first-chunk latency on CPU with 20-language auto-detection.

**Architecture:** Single-Activity Compose UI + Application-singleton `TtsEngine` running 5 ONNX Runtime sessions. True streaming via `AudioTrack.STREAM` + `channelFlow` back-pressure. Model files downloaded on first launch from GitHub Releases or pushed via adb.

**Tech Stack:** Kotlin 1.9+, Jetpack Compose (Material3), ONNX Runtime 1.20+, sentencepiece-android (or NDK fallback), OkHttp 4, Coroutines, Android ICU. minSdk 26, targetSdk 34, arm64-v8a only.

**Reference docs:**
- Design (frozen): `docs/plans/2026-04-16-android-onnx-demo-design.md`
- DEVLOG (live): `android/DEVLOG.md`
- Python reference: `onnx_infer.py`, `onnx_tts_utils.py`, `export_onnx.py`
- Bundle layout: `onnx_export/manifest.json`, `onnx_export/audio_decoder_state_spec.json`, `onnx_export/config.json`

---

## Conventions

- Branch: `feat/android-onnx-demo` (already exists, current HEAD `e63916a`)
- Commit prefix: `android(<area>): <message>` — e.g. `android(text): add CommonNormalizer`
- Push to `origin/feat/android-onnx-demo` after each milestone (M0…M6); user does not require PR
- After each milestone: update `android/DEVLOG.md` Done table + Backlog strikeouts + Benchmarks (M3+)
- Test frameworks: JUnit 4 + Truth (`com.google.truth:truth`) for JVM unit tests; AndroidJUnit4 for instrumentation
- Run JVM tests: `cd android && ./gradlew :app:testDebugUnitTest`
- Run instrumentation: `cd android && ./gradlew :app:connectedDebugAndroidTest` (needs device/emulator)

---

# Milestone M0 — Bootstrap (≈ 0.5 d)

**Goal:** Empty Compose "Hello world" app installs and launches on device. Establishes Gradle wrapper, build files, package layout.

## Task M0.1 — Skeleton directories and Gradle wrapper [UX/Config]

**Files:**
- Create: `android/.gitignore`
- Create: `android/settings.gradle.kts`
- Create: `android/build.gradle.kts`
- Create: `android/gradle.properties`
- Create: `android/gradle/wrapper/gradle-wrapper.properties`
- Create: `android/gradle/wrapper/gradle-wrapper.jar` (run `gradle wrapper --gradle-version 8.5`)
- Create: `android/gradlew`, `android/gradlew.bat`

**Step 1 — Implement**

`android/.gitignore`:
```
/build
/app/build
/.gradle
/local.properties
.idea/
*.apk
*.aab
captures/
.kotlin/
```

`android/settings.gradle.kts`:
```kotlin
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "MossTtsNano"
include(":app")
```

`android/build.gradle.kts`:
```kotlin
plugins {
    id("com.android.application") version "8.2.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
}
```

`android/gradle.properties`:
```
org.gradle.jvmargs=-Xmx4g -Dfile.encoding=UTF-8
android.useAndroidX=true
kotlin.code.style=official
android.nonTransitiveRClass=true
```

Run once on host to materialize wrapper:
```bash
cd android
gradle wrapper --gradle-version 8.5 --distribution-type bin
```

**Step 2 — Verify**
```bash
cd android && ./gradlew --version
```
Expected: prints Gradle 8.5, JVM ≥ 17.

**Step 3 — Commit**
```bash
git add android/
git commit -m "android(bootstrap): gradle wrapper and root build files"
```

---

## Task M0.2 — `app` module and AndroidManifest [UX/Config]

**Files:**
- Create: `android/app/build.gradle.kts`
- Create: `android/app/proguard-rules.pro`
- Create: `android/app/src/main/AndroidManifest.xml`

**Step 1 — Implement**

`android/app/build.gradle.kts`:
```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.afun.mosstts"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.afun.mosstts"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += listOf("arm64-v8a") }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
    composeOptions { kotlinCompilerExtensionVersion = "1.5.8" }
    packaging {
        jniLibs.useLegacyPackaging = false
        resources.excludes += setOf("/META-INF/{AL2.0,LGPL2.1}")
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.02.00")
    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    testImplementation("junit:junit:4.13.2")
    testImplementation("com.google.truth:truth:1.4.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
}
```

`android/app/proguard-rules.pro`:
```
# Keep ONNX Runtime native bindings (added in M1)
-keep class ai.onnxruntime.** { *; }
```

`android/app/src/main/AndroidManifest.xml`:
```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

    <application
        android:name=".App"
        android:label="MOSS TTS"
        android:icon="@mipmap/ic_launcher"
        android:theme="@style/Theme.MossTts">
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:screenOrientation="portrait"
            android:configChanges="uiMode">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

**Step 2 — Verify**
```bash
cd android && ./gradlew :app:tasks --all | head -20
```
Expected: lists `assembleDebug`, no errors.

**Step 3 — Commit**
```bash
git add android/app/build.gradle.kts android/app/proguard-rules.pro android/app/src/main/AndroidManifest.xml
git commit -m "android(bootstrap): app module + manifest"
```

---

## Task M0.3 — Hello-World App + MainActivity [UX/Config]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/App.kt`
- Create: `android/app/src/main/kotlin/com/afun/mosstts/MainActivity.kt`
- Create: `android/app/src/main/kotlin/com/afun/mosstts/ui/theme/Color.kt`
- Create: `android/app/src/main/kotlin/com/afun/mosstts/ui/theme/Type.kt`
- Create: `android/app/src/main/kotlin/com/afun/mosstts/ui/theme/Theme.kt`
- Create: `android/app/src/main/res/values/strings.xml`
- Create: `android/app/src/main/res/values/themes.xml`
- Create: `android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml`
- Create: `android/app/src/main/res/values/ic_launcher_background.xml`
- Create: `android/app/src/main/res/drawable/ic_launcher_foreground.xml`

**Step 1 — Implement**

`App.kt`:
```kotlin
package com.afun.mosstts

import android.app.Application

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        // TtsEngine singleton wiring goes here in M1
    }
}
```

`MainActivity.kt`:
```kotlin
package com.afun.mosstts

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.afun.mosstts.ui.theme.MossTtsTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MossTtsTheme { Surface { HelloScreen() } } }
    }
}

@Composable
fun HelloScreen() {
    Box(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("MOSS TTS Nano · v0.1.0")
    }
}
```

`ui/theme/Color.kt`, `Type.kt`, `Theme.kt`: standard Material3 boilerplate (Compose Studio template). The minimal `Theme.kt`:
```kotlin
package com.afun.mosstts.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

@Composable
fun MossTtsTheme(useDark: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (useDark) darkColorScheme() else lightColorScheme(),
        content = content
    )
}
```

`res/values/strings.xml`:
```xml
<resources>
    <string name="app_name">MOSS TTS</string>
</resources>
```

`res/values/themes.xml`:
```xml
<resources>
    <style name="Theme.MossTts" parent="android:Theme.Material.Light.NoActionBar" />
</resources>
```

Launcher icon: use Android Studio's "Image Asset" wizard to create a stub from a 'M' letter, or copy a standard adaptive icon. Acceptable placeholder fine for MVP.

**Step 2 — Verify**
```bash
cd android && ./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.afun.mosstts/.MainActivity
```
Expected: app launches, shows "MOSS TTS Nano · v0.1.0".

**Step 3 — Commit**
```bash
git add android/app/src/main/
git commit -m "android(bootstrap): hello-world Compose activity"
```

---

## Task M0.4 — Update DEVLOG and finalize M0 [UX/Config]

**Files:**
- Modify: `android/DEVLOG.md`

**Step 1 — Implement**

Move from In Progress / Backlog to Done:

```markdown
## 🟢 已完成 (Done)
| 日期 | 内容 | Commit |
|---|---|---|
| 2026-04-XX | M0 Bootstrap (gradlew + Hello World) | <hash> |
```

Strikethrough M0 items in Backlog:
```markdown
| ~~P1~~ | ~~Gradle 工程初始化 + Compose 模板~~ | MVP — done in M0 |
```

**Step 2 — Verify** — Read DEVLOG, confirm it parses as valid Markdown.

**Step 3 — Commit & push milestone**
```bash
git add android/DEVLOG.md
git commit -m "android(M0): mark milestone complete in DEVLOG"
git push origin feat/android-onnx-demo
```

---

# Milestone M1 — Model & Tokenizer (≈ 1 d)

**Goal:** App can parse the ONNX bundle's `manifest.json`, validate file presence + sha256, and load the SentencePiece tokenizer. No ORT sessions yet.

## Task M1.1 — Add ONNX Runtime + SentencePiece dependencies [UX/Config]

**Files:**
- Modify: `android/app/build.gradle.kts`

**Step 1 — Implement**

Append to `dependencies { ... }`:
```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")

// SentencePiece Android JNI — try AAR first; if Maven copy fails to resolve,
// fall back to NDK build per design § 5 / decision #20. Working candidate:
implementation("com.github.eaglesakura:sentencepiece-android:0.1.0")
// If above fails: clone https://github.com/google/sentencepiece, build with
// android-ndk-r25c for arm64-v8a, drop libsentencepiece.so into
// app/src/main/jniLibs/arm64-v8a/, and ship a small JNI wrapper class.
```

**Step 2 — Verify**
```bash
cd android && ./gradlew :app:assembleDebug
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep -E "onnxruntime|sentencepiece"
```
Expected: both `.so` files present in `lib/arm64-v8a/`.

If sentencepiece AAR fails to resolve, lock decision #20 to NDK fallback in `android/DEVLOG.md` and add the sub-task `M1.1b — Build sentencepiece for arm64-v8a` (skipped here; document it inline if hit).

**Step 3 — Commit**
```bash
git add android/app/build.gradle.kts
git commit -m "android(deps): onnxruntime-android + sentencepiece-android"
```

---

## Task M1.2 — Manifest data class + parser [TDD]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/model/Manifest.kt`
- Create: `android/app/src/test/kotlin/com/afun/mosstts/core/model/ManifestTest.kt`
- Create: `android/app/src/test/resources/manifest_v2_sample.json`

**Step 1 — Failing test**

`src/test/resources/manifest_v2_sample.json`: copy actual `onnx_export/manifest.json` content from project root.

`ManifestTest.kt`:
```kotlin
package com.afun.mosstts.core.model

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class ManifestTest {
    private val sample = javaClass.getResource("/manifest_v2_sample.json")!!.readText()

    @Test fun `parses schema and quantization`() {
        val m = Manifest.parse(sample)
        assertThat(m.schema).isEqualTo("moss-tts-nano-onnx-bundle/v2")
        assertThat(m.quantization.weightType).isEqualTo("QInt8")
        assertThat(m.quantization.perChannel).isFalse()
    }

    @Test fun `lists all int8 onnx files`() {
        val m = Manifest.parse(sample)
        val names = m.files.map { it.name }
        assertThat(names).containsAtLeast(
            "audio_decoder_int8.onnx",
            "audio_encoder_int8.onnx",
            "global_transformer_int8.onnx",
            "local_decoder_text_int8.onnx",
            "local_decoder_audio_int8.onnx",
            "tokenizer.model",
            "config.json",
            "audio_decoder_state_spec.json",
        )
    }

    @Test fun `rejects malformed json`() {
        assertThat(runCatching { Manifest.parse("not json") }.isFailure).isTrue()
    }
}
```

**Step 2 — Run test, expect FAIL**
```bash
cd android && ./gradlew :app:testDebugUnitTest --tests "*.ManifestTest"
```
Expected: compile error (Manifest doesn't exist).

**Step 3 — Implement**

`Manifest.kt`:
```kotlin
package com.afun.mosstts.core.model

import org.json.JSONObject

data class Manifest(
    val schema: String,
    val quantization: Quantization,
    val files: List<FileEntry>,
) {
    data class Quantization(val weightType: String, val perChannel: Boolean, val reduceRange: Boolean)
    data class FileEntry(val name: String, val sizeBytes: Long)

    companion object {
        fun parse(json: String): Manifest {
            val o = JSONObject(json)
            val q = o.getJSONObject("quantization")
            val arr = o.getJSONArray("files")
            val files = (0 until arr.length()).map {
                val it = arr.getJSONObject(it)
                FileEntry(it.getString("name"), it.getLong("size_bytes"))
            }
            return Manifest(
                schema = o.getString("schema"),
                quantization = Quantization(
                    weightType = q.getString("weight_type"),
                    perChannel = q.getBoolean("per_channel"),
                    reduceRange = q.getBoolean("reduce_range"),
                ),
                files = files,
            )
        }
    }
}
```

**Step 4 — Run, expect PASS**
```bash
./gradlew :app:testDebugUnitTest --tests "*.ManifestTest"
```
Expected: 3 passed.

**Step 5 — Commit**
```bash
git add android/app/src/main/kotlin/com/afun/mosstts/core/model/Manifest.kt \
        android/app/src/test/kotlin/com/afun/mosstts/core/model/ManifestTest.kt \
        android/app/src/test/resources/manifest_v2_sample.json
git commit -m "android(model): Manifest data class + parser"
```

---

## Task M1.3 — `ModelConfig` parser [TDD]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/model/ModelConfig.kt`
- Create: `android/app/src/test/kotlin/com/afun/mosstts/core/model/ModelConfigTest.kt`
- Create: `android/app/src/test/resources/config_sample.json`

**Step 1 — Failing test**

Copy `onnx_export/config.json` into `src/test/resources/config_sample.json`.

```kotlin
class ModelConfigTest {
    @Test fun `parses tokens and audio params`() {
        val raw = javaClass.getResource("/config_sample.json")!!.readText()
        val c = ModelConfig.parse(raw)
        assertThat(c.nVq).isEqualTo(16)
        assertThat(c.audioPadTokenId).isAtLeast(0)
        assertThat(c.imStartTokenId).isAtLeast(0)
        assertThat(c.imEndTokenId).isAtLeast(0)
        assertThat(c.audioStartTokenId).isAtLeast(0)
        assertThat(c.sampleRate).isEqualTo(24000)
    }
}
```

**Step 2 — Verify FAIL** — `./gradlew :app:testDebugUnitTest --tests "*.ModelConfigTest"`

**Step 3 — Implement**

```kotlin
package com.afun.mosstts.core.model

import org.json.JSONObject

data class ModelConfig(
    val nVq: Int,
    val audioPadTokenId: Int,
    val imStartTokenId: Int,
    val imEndTokenId: Int,
    val audioStartTokenId: Int,
    val audioAssistantSlotTokenId: Int,
    val audioUserSlotTokenId: Int,
    val sampleRate: Int,
    val rawJson: String,  // kept for fields PromptBuilder may need later
) {
    companion object {
        fun parse(json: String): ModelConfig {
            val o = JSONObject(json)
            return ModelConfig(
                nVq = o.getInt("n_vq"),
                audioPadTokenId = o.getInt("audio_pad_token_id"),
                imStartTokenId = o.getInt("im_start_token_id"),
                imEndTokenId = o.getInt("im_end_token_id"),
                audioStartTokenId = o.getInt("audio_start_token_id"),
                audioAssistantSlotTokenId = o.getInt("audio_assistant_slot_token_id"),
                audioUserSlotTokenId = o.getInt("audio_user_slot_token_id"),
                sampleRate = o.optInt("sample_rate", 24000),
                rawJson = json,
            )
        }
    }
}
```

**Step 4 — Verify PASS**

**Step 5 — Commit** — `git commit -m "android(model): ModelConfig parser"`

---

## Task M1.4 — `Sha256Verifier` [TDD]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/download/Sha256Verifier.kt`
- Create: `android/app/src/test/kotlin/com/afun/mosstts/core/download/Sha256VerifierTest.kt`

**Step 1 — Failing test**
```kotlin
class Sha256VerifierTest {
    @Test fun `hex of empty string`() {
        assertThat(Sha256Verifier.hex(byteArrayOf()))
            .isEqualTo("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }
    @Test fun `hex of abc`() {
        assertThat(Sha256Verifier.hex("abc".toByteArray()))
            .isEqualTo("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }
    @Test fun `streamHex matches hex`(@TempDir dir: java.io.File? = null) {
        val tmp = java.io.File.createTempFile("sha", ".bin").apply { writeBytes(ByteArray(8192) { (it % 256).toByte() }) }
        try {
            val expected = Sha256Verifier.hex(tmp.readBytes())
            val streamed = tmp.inputStream().use { Sha256Verifier.streamHex(it) }
            assertThat(streamed).isEqualTo(expected)
        } finally { tmp.delete() }
    }
}
```

**Step 2 — FAIL**

**Step 3 — Implement**
```kotlin
package com.afun.mosstts.core.download

import java.io.InputStream
import java.security.MessageDigest

object Sha256Verifier {
    fun hex(bytes: ByteArray): String {
        val md = MessageDigest.getInstance("SHA-256").apply { update(bytes) }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
    fun streamHex(input: InputStream): String {
        val md = MessageDigest.getInstance("SHA-256")
        val buf = ByteArray(64 * 1024)
        while (true) {
            val n = input.read(buf)
            if (n <= 0) break
            md.update(buf, 0, n)
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
```

**Step 4 — PASS**

**Step 5 — Commit** — `git commit -m "android(download): Sha256Verifier"`

---

## Task M1.5 — `ModelManager` (manifest validation, no ORT yet) [TDD]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/model/ModelManager.kt`
- Create: `android/app/src/test/kotlin/com/afun/mosstts/core/model/ModelManagerTest.kt`

**Step 1 — Failing test**

```kotlin
class ModelManagerTest {
    @Test fun `validates a complete bundle`(@TempDir dir: java.io.File) {
        // Write a complete fake bundle (manifest + 9 zero-byte files)
        dir.resolve("manifest.json").writeText(SAMPLE_MANIFEST_TEXT)
        dir.resolve("config.json").writeText("{\"n_vq\":16, ...}")
        listOf("tokenizer.model", "audio_decoder_state_spec.json",
               "audio_decoder_int8.onnx", "audio_encoder_int8.onnx",
               "global_transformer_int8.onnx",
               "local_decoder_text_int8.onnx", "local_decoder_audio_int8.onnx")
            .forEach { dir.resolve(it).writeBytes(ByteArray(8)) }

        val report = ModelManager.validate(dir, expectedSha = mapOf())  // empty = skip
        assertThat(report.complete).isTrue()
    }

    @Test fun `reports missing files`(@TempDir dir: java.io.File) {
        dir.resolve("manifest.json").writeText(SAMPLE_MANIFEST_TEXT)
        val report = ModelManager.validate(dir, expectedSha = mapOf())
        assertThat(report.complete).isFalse()
        assertThat(report.missing).contains("config.json")
    }
}
```

**Step 2 — FAIL**

**Step 3 — Implement**

```kotlin
package com.afun.mosstts.core.model

import com.afun.mosstts.core.download.Sha256Verifier
import java.io.File

class ModelManager private constructor(val dir: File, val manifest: Manifest, val config: ModelConfig) {

    data class ValidationReport(
        val complete: Boolean,
        val missing: List<String>,
        val checksumFailed: List<String>,
    )

    companion object {
        const val MANIFEST_FILE = "manifest.json"
        const val CONFIG_FILE = "config.json"

        fun validate(dir: File, expectedSha: Map<String, String>): ValidationReport {
            val mFile = File(dir, MANIFEST_FILE)
            if (!mFile.exists()) return ValidationReport(false, listOf(MANIFEST_FILE), emptyList())
            val manifest = Manifest.parse(mFile.readText())
            val expected = manifest.files.map { it.name } + CONFIG_FILE
            val missing = expected.filter { !File(dir, it).exists() }
            val failed = expectedSha.mapNotNull { (name, sha) ->
                val f = File(dir, name)
                if (!f.exists()) return@mapNotNull null
                if (f.inputStream().use { Sha256Verifier.streamHex(it) } != sha) name else null
            }
            return ValidationReport(missing.isEmpty() && failed.isEmpty(), missing, failed)
        }

        fun open(dir: File, expectedSha: Map<String, String>): ModelManager {
            val r = validate(dir, expectedSha)
            require(r.complete) { "Bundle invalid: missing=${r.missing} bad_sha=${r.checksumFailed}" }
            val manifest = Manifest.parse(File(dir, MANIFEST_FILE).readText())
            val config = ModelConfig.parse(File(dir, CONFIG_FILE).readText())
            return ModelManager(dir, manifest, config)
        }
    }

    fun fileFor(name: String): File = File(dir, name)
}
```

(ORT session creation will be added in M1.6 / M2 once we know we need it.)

**Step 4 — PASS** — `./gradlew :app:testDebugUnitTest --tests "*.ModelManagerTest"`

**Step 5 — Commit** — `git commit -m "android(model): ModelManager validation"`

---

## Task M1.6 — Tokenizer JNI wrapper + smoke test [TDD]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/text/Tokenizer.kt`
- Create: `android/app/src/androidTest/kotlin/com/afun/mosstts/core/text/TokenizerSmokeTest.kt`

**Step 1 — Failing test (instrumented; needs `tokenizer.model` pushed to device cache)**

`TokenizerSmokeTest.kt`:
```kotlin
@RunWith(AndroidJUnit4::class)
class TokenizerSmokeTest {
    @Test fun encodeDecodeRoundtrip() {
        val ctx = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        val modelFile = java.io.File(ctx.getExternalFilesDir(null), "onnx_export/tokenizer.model")
        org.junit.Assume.assumeTrue("Push tokenizer.model first", modelFile.exists())
        val tk = Tokenizer.load(modelFile.absolutePath)
        val ids = tk.encode("Hello 你好")
        Truth.assertThat(ids).isNotEmpty()
        val text = tk.decode(ids)
        Truth.assertThat(text.replace(" ", "")).contains("Hello")
        tk.close()
    }
}
```

**Step 2 — FAIL** — `./gradlew :app:connectedDebugAndroidTest --tests "*.TokenizerSmokeTest"`

**Step 3 — Implement**

`Tokenizer.kt` — thin wrapper around chosen sentencepiece library. The actual class name from the AAR will dictate the inner code; the contract is:
```kotlin
package com.afun.mosstts.core.text

interface Tokenizer : AutoCloseable {
    fun encode(text: String): IntArray
    fun decode(ids: IntArray): String
    val bosId: Int
    val eosId: Int

    companion object {
        fun load(modelPath: String): Tokenizer = SentencePieceTokenizerImpl(modelPath)
    }
}

private class SentencePieceTokenizerImpl(modelPath: String) : Tokenizer {
    // Replace with actual SP-Android API. Example with Google's sentencepiece JNI:
    private val sp = SentencePieceProcessor().apply { load(modelPath) }
    override fun encode(text: String): IntArray = sp.encodeAsIds(text)
    override fun decode(ids: IntArray): String = sp.decodeIds(ids)
    override val bosId: Int get() = sp.bosId()
    override val eosId: Int get() = sp.eosId()
    override fun close() = sp.close()
}
```

**Step 4 — PASS** (after `adb push onnx_export/ /sdcard/Android/data/com.afun.mosstts/files/`)

**Step 5 — Commit** — `git commit -m "android(text): Tokenizer wrapper + smoke test"`

---

## Task M1.7 — M1 milestone close [UX/Config]

Update `android/DEVLOG.md`: mark M1 done, log JVM unit test count (should be ~8 tests passing). Push.

```bash
git push origin feat/android-onnx-demo
```

---

# Milestone M2 — Inference Loop (≈ 2 d)

**Goal:** Synthesize a complete WAV file from text via 5 ORT sessions; output byte-equal (or perceptually equal) to `onnx_infer.py`. No streaming UI yet — outputs PCM to a file.

## Task M2.1 — Python: dump prompt fixtures for golden test [UX/Config]

**Files:**
- Create: `tools/dump_prompt_fixtures.py`

**Step 1 — Implement**

```python
#!/usr/bin/env python3
"""Dump PromptBuilder outputs so the Kotlin port can be golden-tested."""
import argparse, hashlib, json, os
import numpy as np
from onnx_infer import PromptBuilder
from onnx_tts_utils import SPTokenizer

def fingerprint(arr: np.ndarray) -> dict:
    flat = arr.flatten().astype(np.int64)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "first_8": flat[:8].tolist(),
        "last_8":  flat[-8:].tolist(),
        "sum": int(flat.sum()),
        "sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default="onnx_export")
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.bundle_dir, "config.json")))
    tk = SPTokenizer(os.path.join(args.bundle_dir, "tokenizer.model"))
    pb = PromptBuilder(tk, cfg)
    out = []
    for t in args.texts:
        ids = list(tk.encode(t, add_special_tokens=False))
        input_ids, _ = pb.build_continuation_prompt(ids)
        out.append({"text": t, "input_ids": fingerprint(input_ids)})
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
```

**Step 2 — Verify**
```bash
python tools/dump_prompt_fixtures.py \
    --texts "你好" "Hello world" "こんにちは、世界" "Привет мир" "" "1+2=3" \
    --out android/app/src/test/resources/prompt_fixtures.json
```
Expected: writes a JSON file with 6 entries.

**Step 3 — Commit**
```bash
git add tools/dump_prompt_fixtures.py android/app/src/test/resources/prompt_fixtures.json
git commit -m "tools(android): dump prompt fixtures for Kotlin port"
```

---

## Task M2.2 — `PromptBuilder` Kotlin port + golden test [TDD]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/infer/PromptBuilder.kt`
- Create: `android/app/src/test/kotlin/com/afun/mosstts/core/infer/PromptBuilderTest.kt`

**Step 1 — Failing test**

```kotlin
class PromptBuilderTest {
    private val fixtures = JSONArray(javaClass.getResource("/prompt_fixtures.json")!!.readText())
    private val cfg = ModelConfig.parse(javaClass.getResource("/config_sample.json")!!.readText())
    private val tokenizer = StubTokenizer.fromFile("/tokenizer.model")  // load real sp from resources, see note

    @Test fun `matches python golden for each fixture`() {
        val pb = PromptBuilder(tokenizer, cfg)
        for (i in 0 until fixtures.length()) {
            val f = fixtures.getJSONObject(i)
            val text = f.getString("text")
            val ids = tokenizer.encode(text)
            val (inputIds, _) = pb.buildContinuationPrompt(ids, null)
            val expected = f.getJSONObject("input_ids")
            assertThat(inputIds.size).isEqualTo(expected.getJSONArray("shape").getInt(0)
                * expected.getJSONArray("shape").getInt(1)
                * expected.getJSONArray("shape").getInt(2))
            // Compare first/last 8 and sha256
            assertThat(sha256Hex(inputIds.toByteArrayLE()))
                .isEqualTo(expected.getString("sha256"))
        }
    }
}
```

> **Note** — embedding `tokenizer.model` (~470 KB) as a JVM test resource is fine; it's the same file shipped to the device. Drop a copy under `app/src/test/resources/tokenizer.model`.

> **Tokenizer in JVM tests** — The Android `Tokenizer` impl uses JNI; for JVM tests use a tiny pure-Kotlin SP loader OR run this test as instrumentation (`androidTest`). Recommended: keep this as `androidTest` to share the same JNI loader.

**Step 2 — FAIL**

**Step 3 — Implement**

`PromptBuilder.kt` — port of `onnx_infer.py:PromptBuilder` (lines 68-124). Key constants from Python:

```kotlin
package com.afun.mosstts.core.infer

import com.afun.mosstts.core.model.ModelConfig
import com.afun.mosstts.core.text.Tokenizer

private const val USER_ROLE_PREFIX = "user\n"
private const val USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
private const val USER_TEMPLATE_AFTER_REFERENCE =
    "\n- Instruction:\nNone\n" +
    "- Tokens:\nNone\n" +
    "- Quality:\nNone\n" +
    "- Sound Event:\nNone\n" +
    "- Ambient Sound:\nNone\n" +
    "- Language:\nNone\n" +
    "- Text:\n"
private const val USER_TEMPLATE_SUFFIX = "\n</user_inst>"
private const val ASSISTANT_TURN_PREFIX = "\n"
private const val ASSISTANT_ROLE_PREFIX = "assistant\n"

class PromptBuilder(private val tokenizer: Tokenizer, private val cfg: ModelConfig) {
    private val nVq = cfg.nVq
    private val audioPad = cfg.audioPadTokenId
    private val userPrefix: IntArray
    private val afterRef:   IntArray
    private val noneIds:    IntArray
    private val assistantPrefix: IntArray

    init {
        fun enc(t: String) = tokenizer.encode(t)
        userPrefix = intArrayOf(cfg.imStartTokenId) + enc(USER_ROLE_PREFIX) + enc(USER_TEMPLATE_REFERENCE_PREFIX)
        afterRef = enc(USER_TEMPLATE_AFTER_REFERENCE)
        noneIds = enc("None")
        assistantPrefix = enc(USER_TEMPLATE_SUFFIX) + intArrayOf(cfg.imEndTokenId) +
                enc(ASSISTANT_TURN_PREFIX) + intArrayOf(cfg.imStartTokenId) + enc(ASSISTANT_ROLE_PREFIX)
    }

    /** Returns input_ids LongArray flattened in [B=1, T, nVq+1] row-major, plus T. */
    fun buildContinuationPrompt(textIds: IntArray, promptAudioCodes: Array<IntArray>?): Pair<LongArray, Int> {
        val promptIds = userPrefix + noneIds + afterRef + textIds + assistantPrefix
        val sections = mutableListOf<LongArray>()  // each row is [nVq+1] length
        sections += textRows(promptIds)
        sections += textRows(intArrayOf(cfg.audioStartTokenId))
        if (promptAudioCodes != null) {
            sections += audioPrefixRows(promptAudioCodes, cfg.audioAssistantSlotTokenId)
        }
        val totalT = sections.sumOf { it.size / (nVq + 1) }
        val flat = LongArray(totalT * (nVq + 1))
        var off = 0
        for (s in sections) { s.copyInto(flat, off); off += s.size }
        return flat to totalT
    }

    private fun textRows(ids: IntArray): LongArray {
        val rows = LongArray(ids.size * (nVq + 1)) { audioPad.toLong() }
        for (i in ids.indices) rows[i * (nVq + 1)] = ids[i].toLong()
        return rows
    }

    private fun audioPrefixRows(codes: Array<IntArray>, slotTokenId: Int): LongArray {
        val T = codes.size
        val rows = LongArray(T * (nVq + 1)) { audioPad.toLong() }
        for (t in 0 until T) {
            rows[t * (nVq + 1)] = slotTokenId.toLong()
            for (q in 0 until nVq) rows[t * (nVq + 1) + 1 + q] = codes[t][q].toLong()
        }
        return rows
    }
}
```

**Step 4 — PASS** — `./gradlew :app:connectedDebugAndroidTest --tests "*.PromptBuilderTest"`

**Step 5 — Commit** — `git commit -m "android(infer): PromptBuilder Kotlin port + golden test"`

---

## Task M2.3 — `Sampler` (top-k / top-p / temperature / repetition penalty) [TDD]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/infer/Sampler.kt`
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/infer/SamplingConfig.kt`
- Create: `android/app/src/test/kotlin/com/afun/mosstts/core/infer/SamplerTest.kt`

**Step 1 — Failing test**

```kotlin
class SamplerTest {
    @Test fun `temperature zero is argmax`() {
        val logits = floatArrayOf(0.1f, 5.0f, 0.2f, 0.3f)
        val s = Sampler(SamplingConfig(temperature = 0f, topK = 0, topP = 1f, repetitionPenalty = 1f))
        repeat(50) { assertThat(s.sample(logits, prevIds = intArrayOf())).isEqualTo(1) }
    }
    @Test fun `top_k restricts to top-k`() {
        val logits = floatArrayOf(0.1f, 0.2f, 5.0f, 5.1f)
        val s = Sampler(SamplingConfig(temperature = 1f, topK = 2, topP = 1f, repetitionPenalty = 1f))
        repeat(100) {
            val id = s.sample(logits, prevIds = intArrayOf())
            assertThat(id).isAnyOf(2, 3)
        }
    }
    @Test fun `repetition penalty downweights prev ids`() {
        val logits = floatArrayOf(2f, 2f, 2f, 2f)
        val s = Sampler(SamplingConfig(temperature = 0f, topK = 0, topP = 1f, repetitionPenalty = 2f))
        repeat(50) { assertThat(s.sample(logits, prevIds = intArrayOf(0))).isNotEqualTo(0) }
    }
}
```

**Step 2 — FAIL**

**Step 3 — Implement**

```kotlin
data class SamplingConfig(val temperature: Float, val topK: Int, val topP: Float, val repetitionPenalty: Float)

class Sampler(private val cfg: SamplingConfig, private val rng: kotlin.random.Random = kotlin.random.Random.Default) {
    fun sample(logits: FloatArray, prevIds: IntArray): Int {
        val l = logits.copyOf()
        if (cfg.repetitionPenalty != 1f) {
            val seen = prevIds.toHashSet()
            for (i in l.indices) if (i in seen) l[i] /= cfg.repetitionPenalty
        }
        if (cfg.temperature == 0f) return l.indices.maxBy { l[it] }
        for (i in l.indices) l[i] /= cfg.temperature
        // top-k filter
        if (cfg.topK > 0 && cfg.topK < l.size) {
            val threshold = l.copyOf().also { it.sort() }.let { it[it.size - cfg.topK] }
            for (i in l.indices) if (l[i] < threshold) l[i] = Float.NEGATIVE_INFINITY
        }
        // softmax
        val mx = l.max()
        var sum = 0f
        for (i in l.indices) { l[i] = kotlin.math.exp(l[i] - mx); sum += l[i] }
        for (i in l.indices) l[i] /= sum
        // top-p
        if (cfg.topP < 1f) {
            val sorted = l.indices.sortedByDescending { l[it] }
            var cum = 0f; var keep = 0
            for ((k, idx) in sorted.withIndex()) { cum += l[idx]; keep = k + 1; if (cum >= cfg.topP) break }
            val keepSet = sorted.subList(0, keep).toHashSet()
            var renorm = 0f
            for (i in l.indices) { if (i !in keepSet) l[i] = 0f else renorm += l[i] }
            for (i in l.indices) l[i] /= renorm
        }
        // sample
        val r = rng.nextFloat(); var cum = 0f
        for (i in l.indices) { cum += l[i]; if (r <= cum) return i }
        return l.size - 1
    }
}
```

**Step 4 — PASS**

**Step 5 — Commit** — `git commit -m "android(infer): Sampler + SamplingConfig"`

---

## Task M2.4 — `KvCache` and StateSpec parser [TDD]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/infer/KvCache.kt`
- Create: `android/app/src/test/kotlin/com/afun/mosstts/core/infer/KvCacheTest.kt`
- Create: `android/app/src/test/resources/audio_decoder_state_spec_sample.json` (copy from `onnx_export/`)

**Step 1 — Failing test**

```kotlin
class KvCacheTest {
    @Test fun `parses spec yields correct layer count and shapes`() {
        val raw = javaClass.getResource("/audio_decoder_state_spec_sample.json")!!.readText()
        val spec = StateSpec.parse(raw)
        assertThat(spec.layers).hasSize(spec.numLayers)
        assertThat(spec.framesPerCall).isEqualTo(1)
        // Each layer has key+value with [B=1, H, T=0 initially, D]
        for (l in spec.layers) {
            assertThat(l.key.shape[0]).isEqualTo(1)
            assertThat(l.value.shape[0]).isEqualTo(1)
        }
    }

    @Test fun `empty cache has zero seq len`() {
        val spec = StateSpec.parse(javaClass.getResource("/audio_decoder_state_spec_sample.json")!!.readText())
        val cache = KvCache.empty(spec)
        for (l in cache.layers) assertThat(l.key.shape[2]).isEqualTo(0)
    }
}
```

**Step 2 — FAIL**

**Step 3 — Implement** — `StateSpec.parse(json)` reads names + shapes per layer; `KvCache` keeps `Pair<OnnxTensor, OnnxTensor>` per layer plus a method to feed-in / receive-out for `OrtSession.run`. Reference: `audio_decoder_state_spec.json` produced by `export_onnx.py`.

(Skeleton kept short here; full code is straightforward I/O on the JSON spec.)

**Step 4 — PASS**

**Step 5 — Commit** — `git commit -m "android(infer): KvCache + StateSpec parser"`

---

## Task M2.5 — `InferenceLoop` (no audio decoder yet → emit codes) [TDD-light]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/infer/InferenceLoop.kt`
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/tts/TtsEngine.kt`
- Create: `android/app/src/androidTest/kotlin/com/afun/mosstts/core/infer/InferenceLoopSmokeTest.kt`

**Step 1 — Smoke test (instrumented)**

```kotlin
@RunWith(AndroidJUnit4::class)
class InferenceLoopSmokeTest {
    @Test fun shortSentenceEmitsCodes() = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.getExternalFilesDir(null), "onnx_export")
        Assume.assumeTrue("Push bundle first", File(dir, "manifest.json").exists())
        val engine = TtsEngine.create(ctx, dir)
        engine.preload()
        val codes = mutableListOf<IntArray>()
        engine.synthesizeCodes("你好").collect { codes += it }
        Truth.assertThat(codes.size).isAtLeast(5)
        engine.close()
    }
}
```

**Step 2 — FAIL**

**Step 3 — Implement**

`InferenceLoop.kt`: port of `onnx_infer.py:OnnxTTSEngine.generate` (lines ~480-600). Five sessions: `audio_encoder`, `global_transformer`, `local_decoder_text`, `local_decoder_audio`, `audio_decoder` (audio_decoder still unused at this step — leave its session loaded but emit raw 16-codebook arrays from `local_decoder_audio` instead of PCM).

`TtsEngine.kt`: top-level façade. `preload()` opens 5 ORT sessions; `synthesizeCodes(text)` returns `Flow<IntArray>` of 16-element code arrays per frame.

```kotlin
class TtsEngine private constructor(
    private val manager: ModelManager,
    private val tokenizer: Tokenizer,
    private val sessions: Sessions,
) : AutoCloseable {
    data class Sessions(
        val audioEncoder: OrtSession,
        val globalTransformer: OrtSession,
        val localText: OrtSession,
        val localAudio: OrtSession,
        val audioDecoder: OrtSession,
    )
    /* preload(), synthesizeCodes(text) emitting Flow<IntArray>, close() */
}
```

**Step 4 — PASS** (push bundle first)

**Step 5 — Commit** — `git commit -m "android(infer): InferenceLoop emits 16-codebook codes"`

---

## Task M2.6 — Hook up `audio_decoder` → emit PCM chunks, write WAV bench [TDD-light]

**Files:**
- Modify: `android/app/src/main/kotlin/com/afun/mosstts/core/tts/TtsEngine.kt`
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/audio/PcmChunk.kt`
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/audio/WavWriter.kt`
- Create: `android/app/src/androidTest/kotlin/com/afun/mosstts/core/tts/TtsEngineWavBenchTest.kt`

**Step 1 — Smoke test (instrumented)**

```kotlin
@Test fun synthesizeShortSentenceProducesValidWav() = runBlocking {
    /* ... load engine ... */
    val pcm = mutableListOf<Float>()
    engine.synthesize("你好").collect { chunk -> pcm += chunk.samples.toList() }
    Truth.assertThat(pcm.size).isAtLeast(24000 / 4)  // ≥ 0.25 s
    val outFile = File(ctx.getExternalFilesDir(null), "smoke_zh_hello.wav")
    WavWriter.write(outFile, pcm.toFloatArray(), 24000)
    Truth.assertThat(outFile.length()).isGreaterThan(8000L)
}
```

**Step 2 — FAIL**

**Step 3 — Implement**

`PcmChunk`: `data class PcmChunk(val samples: FloatArray, val sampleRate: Int)`.

`WavWriter`: 44-byte RIFF header + interleaved 16-bit PCM (or 32-bit float).

`TtsEngine.synthesize(text)`: for each frame from M2.5 → call `audio_decoder` (with KvCache) → emit `PcmChunk`.

**Step 4 — PASS** — Pull the WAV via `adb pull` and listen by ear to confirm intelligible.

**Step 5 — Commit** — `git commit -m "android(tts): synthesize emits PcmChunk; bench writes wav"`

---

## Task M2.7 — M2 milestone close

Update DEVLOG: M2 done. First **Benchmarks** row: log first-chunk ms, total ms, RTF for "你好" on the test device.

```bash
git push origin feat/android-onnx-demo
```

---

# Milestone M3 — Audio Streaming (≈ 1 d)

**Goal:** Audible streaming on device — first PCM chunk audible within ~150 ms (target).

## Task M3.1 — `PcmAccumulator` [TDD]

```kotlin
class PcmAccumulator {
    private val buf = mutableListOf<FloatArray>()
    fun append(c: PcmChunk) { buf += c.samples }
    fun toFloatArray(): FloatArray = buf.flatMap { it.toList() }.toFloatArray()
    fun reset() { buf.clear() }
}
```
Tests: append-then-toArray equals concatenation; reset zeroes; thread-safe (use synchronized internally).

Commit: `android(audio): PcmAccumulator`

---

## Task M3.2 — `AudioPlayer` (AudioTrack STREAM) [TDD-light]

**Files:**
- Create: `android/app/src/main/kotlin/com/afun/mosstts/core/audio/AudioPlayer.kt`
- Create: `android/app/src/androidTest/kotlin/com/afun/mosstts/core/audio/AudioPlayerSmokeTest.kt`

**Step 1 — Smoke test** — instantiate, write 1 s of 440 Hz sine, ensure no exception, audible (manual).

**Step 2 — FAIL**

**Step 3 — Implement**

```kotlin
class AudioPlayer(sampleRate: Int = 24000) : AutoCloseable {
    private val track = AudioTrack.Builder()
        .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
        .setAudioFormat(AudioFormat.Builder()
            .setSampleRate(sampleRate)
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
        .setBufferSizeInBytes(AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT) * 2)
        .setTransferMode(AudioTrack.MODE_STREAM)
        .build()

    private var started = false

    fun start() { if (!started) { track.play(); started = true } }
    fun write(samples: FloatArray) {
        if (!started) start()
        track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
    }
    fun flush() { track.pause(); track.flush() }
    override fun close() { track.stop(); track.release() }
}
```

**Step 4 — PASS** — manually confirm sine wave audible.

**Step 5 — Commit** — `android(audio): AudioPlayer streaming via AudioTrack`

---

## Task M3.3 — Wire `TtsEngine.synthesize` → `AudioPlayer` end-to-end [TDD-light]

Extend `TtsEngine` with:
```kotlin
suspend fun synthesizeAndPlay(text: String, onFirstChunkMs: (Long) -> Unit, onDone: (rtf: Float) -> Unit)
```
that internally collects `synthesize(text).onEach { player.write(it.samples) }` plus the `t0/t_first/t_done` timestamps.

Commit: `android(tts): synthesizeAndPlay end-to-end on-device streaming`

---

## Task M3.4 — Tiny dev UI to trigger synth [UX/Config]

Modify `MainActivity` Hello screen to add a `TextField` + `Button("合成")` that calls `TtsEngine.synthesizeAndPlay`. No styling. Verify sound on device.

Commit: `android(ui): minimal trigger for streaming TTS`

---

## Task M3.5 — M3 milestone close + benchmark

Run synth on 3 short sentences (zh/en/ja). Log first-chunk-ms, RTF to DEVLOG benchmarks table. Push.

---

# Milestone M4 — UI + Normalizer (≈ 1.5 d)

## Task M4.1 — `CommonNormalizer` (L1) [TDD]

Tests: control chars stripped, zero-width stripped, full→half width, multiple spaces collapsed, length truncated to 300.
Implementation: regex-based pipeline.
Commit: `android(text): L1 CommonNormalizer`

## Task M4.2 — `IcuNumberNormalizer` (L2) [TDD]

Tests (4 locales × {123, 1.5, 50%, -7}):
- zh: "123" → "一百二十三"
- en: "1.5" → "one point five"
- ja: "50%" → "百分の五十"
- ru: "-7" → "минус семь"

Implementation: `android.icu.text.RuleBasedNumberFormat(locale, RuleBasedNumberFormat.SPELLOUT)`. Use a regex to extract numeric tokens and substitute.
Commit: `android(text): L2 IcuNumberNormalizer`

## Task M4.3 — `NormalizerPipeline` + `LangNormalizer` interface [TDD]

Tests: pipeline = L1 ∘ L2 (∘ L3 if present). L3 left as no-op stub for V1.1.
Commit: `android(text): NormalizerPipeline`

## Task M4.4 — `LanguageOption` + `Examples.kt` (20 langs) [UX/Config]

```kotlin
enum class LanguageOption(val locale: java.util.Locale, val display: String, val example: String) {
    ZH(Locale("zh","CN"), "中文",     "你好，欢迎使用 MOSS TTS Nano。今天天气不错。"),
    EN(Locale("en","US"), "English",  "Hello, welcome to MOSS TTS Nano. The weather is nice today."),
    JA(Locale("ja","JP"), "日本語",   "こんにちは、MOSS TTS Nanoへようこそ。"),
    KO(Locale("ko","KR"), "한국어",   "안녕하세요, MOSS TTS Nano에 오신 것을 환영합니다."),
    RU(Locale("ru","RU"), "Русский",  "Привет, добро пожаловать в MOSS TTS Nano."),
    FR(Locale("fr","FR"), "Français", "Bonjour, bienvenue dans MOSS TTS Nano."),
    DE(Locale("de","DE"), "Deutsch",  "Hallo, willkommen bei MOSS TTS Nano."),
    ES(Locale("es","ES"), "Español",  "Hola, bienvenido a MOSS TTS Nano."),
    IT(Locale("it","IT"), "Italiano", "Ciao, benvenuto in MOSS TTS Nano."),
    PT(Locale("pt","PT"), "Português","Olá, bem-vindo ao MOSS TTS Nano."),
    AR(Locale("ar"),       "العربية",  "مرحبًا، أهلًا بك في MOSS TTS Nano."),
    HI(Locale("hi","IN"), "हिन्दी",    "नमस्ते, MOSS TTS Nano में आपका स्वागत है."),
    ID(Locale("id","ID"), "Bahasa",   "Halo, selamat datang di MOSS TTS Nano."),
    VI(Locale("vi","VN"), "Tiếng Việt","Xin chào, chào mừng đến với MOSS TTS Nano."),
    TH(Locale("th","TH"), "ไทย",      "สวัสดี ยินดีต้อนรับสู่ MOSS TTS Nano"),
    TR(Locale("tr","TR"), "Türkçe",   "Merhaba, MOSS TTS Nano'ya hoş geldiniz."),
    PL(Locale("pl","PL"), "Polski",   "Cześć, witaj w MOSS TTS Nano."),
    NL(Locale("nl","NL"), "Nederlands","Hallo, welkom bij MOSS TTS Nano."),
    SV(Locale("sv","SE"), "Svenska",  "Hej, välkommen till MOSS TTS Nano."),
    MIX(Locale("zh","CN"),"中英混合", "今天去 Starbucks 喝杯咖啡，order 了一杯 latte。"),
}
```
Commit: `android(i18n): 20 language examples`

## Task M4.5 — Material3 theme polish [UX/Config]

Define brand colors in `Color.kt`, typography in `Type.kt`. Light + dark schemes.
Commit: `android(ui): theme polish`

## Task M4.6 — `MainState`/`MainIntent`/`MainEffect`/`MainViewModel` [TDD-light]

Tests: state transitions on Synthesize / Stop / SwitchLanguage / EditText / FirstChunk / Done.

```kotlin
data class MainState(
    val text: String = "",
    val language: LanguageOption = LanguageOption.ZH,
    val phase: Phase = Phase.Idle,
    val firstChunkMs: Long? = null,
    val rtf: Float? = null,
    val durationMs: Long = 0,
    val playedMs: Long = 0,
    val error: TtsError? = null,
    val modelReady: Boolean = false,
)
sealed interface MainIntent {
    data object Synthesize : MainIntent
    data object Stop : MainIntent
    data class SetText(val s: String) : MainIntent
    data class SetLanguage(val l: LanguageOption) : MainIntent
    data object SaveWav : MainIntent
    data object Share : MainIntent
}
sealed interface MainEffect { /* Toasts, Snackbars, navigation */ }
```

Commit: `android(ui): MainViewModel + MVI`

## Task M4.7 — `MainScreen` Compose UI [UX/Config]

Implement per design § 1 草图 + § 3 取消语义:
- Top app bar (title + about icon)
- Language dropdown
- Multi-line text field (auto-fills from `language.example` when empty or equals previous example)
- Two buttons row: `▶ 合成` (primary) + `⏹ 停止` (secondary, only enabled while playing)
- Status line: `首帧 X ms · RTF Y · INT8`
- Linear progress bar reflecting `playedMs / durationMs`
- Save / Share icons

Commit: `android(ui): MainScreen with language dropdown and streaming controls`

## Task M4.8 — `AppNavigation` + `AboutScreen` [UX/Config]

`AppNavigation` is a Compose function selecting between screens via a `MainState.modelReady` flag (no Navigation library; this is sufficient for 3 screens).

`AboutScreen`: version, upstream link, fork link, GitHub Releases tag link, ICU data version (from `RuleBasedNumberFormat`).

Commit: `android(ui): AppNavigation + About`

## Task M4.9 — End-to-end manual smoke

With model already on device: launch → switch to Japanese → tap example → 合成 → hear audio. Repeat for zh/en/ru. Update DEVLOG benchmarks (3 new rows).

Commit: `android(M4): end-to-end smoke logged`

## Task M4.10 — M4 milestone close

Push.

---

# Milestone M5 — Download (≈ 1.5 d)

## Task M5.1 — Add OkHttp + sha256 known table [UX/Config]

`build.gradle.kts`: `implementation("com.squareup.okhttp3:okhttp:4.12.0")`.

`DownloadManifest.kt`:
```kotlin
object DownloadManifest {
    const val BASE_URL = "https://github.com/AFun9/MOSS-TTS-Nano/releases/download/onnx_model/"
    data class FileSpec(val name: String, val sizeBytes: Long, val sha256: String)
    val FILES: List<FileSpec> = listOf(
        FileSpec("manifest.json",                   727L, "<TBD>"),
        FileSpec("config.json",                     727L, "<TBD>"),
        FileSpec("tokenizer.model",            470_897L, "<TBD>"),
        FileSpec("audio_decoder_state_spec.json", 6_381L, "<TBD>"),
        FileSpec("audio_encoder_int8.onnx",   15_698_156L, "<TBD>"),
        FileSpec("local_decoder_text_int8.onnx", 7_145_975L, "<TBD>"),
        FileSpec("local_decoder_audio_int8.onnx", 19_718_825L, "<TBD>"),
        FileSpec("audio_decoder_int8.onnx",  11_513_855L, "<TBD>"),
        FileSpec("global_transformer_int8.onnx", 110_956_202L, "<TBD>"),
    )
    val totalBytes = FILES.sumOf { it.sizeBytes }
}
```

To populate `<TBD>`: run on host
```bash
cd onnx_export && for f in manifest.json config.json tokenizer.model \
    audio_decoder_state_spec.json *_int8.onnx; do
  echo "$(sha256sum $f)"
done
```
Paste hashes into the Kotlin file.

Commit: `android(download): DownloadManifest with sha256 table`

## Task M5.2 — `ModelDownloader` (parallel=2 + Range resume + sha256 verify) [TDD-light]

Tests: a) verifies sha after download b) resumes from partial file c) skips if already correct.
Implementation: OkHttp + `Semaphore(2)` + `Range: bytes=N-` for resume.

Commit: `android(download): ModelDownloader with resume + sha`

## Task M5.3 — `DownloadViewModel` + `DownloadScreen` [UX/Config]

UI: header "下载模型 · 165 MB · 仅首次"，按文件 9 行进度条 + 总进度 + 取消/重试按钮。

Commit: `android(ui): DownloadScreen`

## Task M5.4 — `App.kt` startup wiring [UX/Config]

```kotlin
class App : Application() {
    val ttsEngine: TtsEngine by lazy {
        TtsEngine.create(this, File(getExternalFilesDir(null), "onnx_export"))
    }
    fun bundleReady(): Boolean {
        val dir = File(getExternalFilesDir(null), "onnx_export")
        return ModelManager.validate(dir, expectedSha = emptyMap()).complete
    }
}
```

`MainActivity`: read `App.bundleReady()` → if true call `ttsEngine.preload()` in background and show MainScreen with synthesize button disabled until preload finishes; else show DownloadScreen → on success switch to MainScreen.

Commit: `android(app): bootstrap wiring (download or main)`

## Task M5.5 — Manual full-flow test [UX/Config]

Uninstall the app, reinstall, observe DownloadScreen → 165 MB downloaded → MainScreen → first synth works. Document timings in DEVLOG.

Commit: `android(M5): full first-launch flow verified`

## Task M5.6 — M5 milestone close + push

---

# Milestone M6 — Polish & Bench (≈ 1 d)

## Task M6.1 — `TtsError` sealed class + Snackbar plumbing [TDD-light]

Implement per design § 6. Snackbar host in Scaffold. ErrorDialog for `error`-severity.

Commit: `android(ui): TtsError handling with Snackbar/Dialog`

## Task M6.2 — `BenchmarkActivity` (10 fixed sentences) [UX/Config]

`BenchmarkActivity` invoked via debug menu. Runs 10 fixed prompts (zh/en/ja/ru × 2-3), records first-chunk-ms / total-ms / rtf, writes a `bench_<device>_<date>.csv` to Downloads.

Commit: `android(bench): BenchmarkActivity`

## Task M6.3 — Run benchmarks; populate DEVLOG [UX/Config]

Run on the test device; copy 5+ rows into DEVLOG `Benchmarks` table.

Commit: `android(M6): benchmark numbers v0.1.0`

## Task M6.4 — `android/README.md` (build / push / run quickstart) [UX/Config]

Three sections:
1. Prereqs (JDK 17, Android Studio Hedgehog+, NDK 25, ADB connected device)
2. Build: `./gradlew :app:assembleDebug && adb install -r app/build/outputs/apk/debug/app-debug.apk`
3. Use: either `adb push onnx_export/ /sdcard/Android/data/com.afun.mosstts/files/onnx_export/` (skip download) or open the app and let it download

Plus a "Performance baseline" subsection that links to DEVLOG.

Commit: `android(docs): quickstart README`

## Task M6.5 — Bump version + tag v0.1.0

`app/build.gradle.kts`: `versionCode = 1`, `versionName = "0.1.0"` (already).
Update DEVLOG `Changelog → v0.1.0` with the actual completion date and feature list.

```bash
git tag -a v0.1.0-android -m "Android demo MVP"
git push origin feat/android-onnx-demo --tags
```

Commit: `android(release): v0.1.0 mvp`

---

# Definition of Done (recap from design § 8)

- [ ] M0–M6 all green
- [ ] JVM unit tests passing (~70 tests)
- [ ] Manual test matrix from design § 7 fully exercised on at least one real device
- [ ] DEVLOG benchmarks ≥ 3 device/lang rows
- [ ] `android/README.md` written
- [ ] Design doc still FROZEN (no edits)
- [ ] `feat/android-onnx-demo` pushed; **no** PR opened

---

# After v0.1.0 (V1.1 backlog → DEVLOG)

- L3 normalizers for zh/en/ja/ru (idiom & symbol expansion)
- HuggingFace + ModelScope mirror download sources
- Voice clone (record 10 s prompt; switch `synthesize` to `voice_clone` mode in `OnnxTTSEngine`)
- Dates/currency normalization (L4)
- `export_onnx.py` to write per-file sha256 into `manifest.json` (eliminates Android-side hardcoded table)
