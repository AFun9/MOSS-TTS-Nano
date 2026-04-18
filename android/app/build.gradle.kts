import java.io.ByteArrayOutputStream

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    kotlin("plugin.serialization") version "1.9.22"
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

    // Make assets/ visible to JVM unit tests as classpath resources so we
    // can load the bundled tokenizer_kotlin.json the same way the runtime does.
    sourceSets {
        getByName("test") {
            resources.srcDir("src/main/assets")
        }
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
    // Pure-JVM JSON; avoids Android's `org.json.*` stub during unit tests.
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")

    // Pinned to match the PC reference (PC ORT 1.24.x), since Android ORT
    // 1.20 mis-broadcasts a `/Add_2` node inside `local_decoder_text_int8`
    // (verified on Mi/Redmi 24122RKC7C: "axis == 1 || axis == largest was
    // false. Attempting to broadcast an axis by a dimension other than 1.
    // 64 by 768"). 1.24.3 is the latest stable Maven release as of writing.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")
    // SentencePiece tokenizer is added in M1.1b (separate task; the Maven
    // candidate referenced by the plan turned out to be a JitPack-style
    // group id, so we deal with it independently to avoid blocking M1.2-M1.5).

    testImplementation("junit:junit:4.13.2")
    testImplementation("com.google.truth:truth:1.4.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("com.google.truth:truth:1.4.0")
}

// ---------------------------------------------------------------------------
// pushModels — adb-push the v1.0.0 release bundle from android/models/ to
// `/data/local/tmp/mosstts_models/` on the connected device.
//
// Why /data/local/tmp instead of the app-private /sdcard/Android/data dir?
// On MIUI / HyperOS (verified on 24122RKC7C) the OS reaps that directory
// whenever a debug-test APK is reinstalled, so anything `pushModels` puts
// there is gone by the time `InferenceLoopTraceTest` actually runs. The
// `/data/local/tmp` path is always world-readable, survives APK reinstalls,
// and isn't touched by MIUI's storage cleaner. The on-device test reads
// the bundle from there via a hard-coded absolute path (see the matching
// `MODELS_DIR` const in `InferenceLoopTraceTest`).
//
// `--sync` makes the per-file push idempotent (skips bytes that already
// match), so re-running `pushModels` after a quantization tweak only sends
// the changed graphs. Stale files left over from earlier bundle layouts
// are scrubbed up front by an explicit wildcard `rm`.
//
// Implemented as a typed task so Gradle 9's `ExecOperations` injection
// works without the deprecated Project.exec(Action) closure.
// ---------------------------------------------------------------------------
abstract class PushModelsTask @javax.inject.Inject constructor(
    private val execOps: org.gradle.process.ExecOperations,
) : DefaultTask() {

    @get:org.gradle.api.tasks.InputDirectory
    @get:org.gradle.api.tasks.PathSensitive(org.gradle.api.tasks.PathSensitivity.RELATIVE)
    abstract val srcDir: DirectoryProperty

    @get:org.gradle.api.tasks.Input
    abstract val deviceDir: Property<String>

    @org.gradle.api.tasks.TaskAction
    fun pushAll() {
        val src = srcDir.get().asFile
        val target = deviceDir.get()
        val files = src.listFiles()
            ?.filter { it.isFile && !it.name.startsWith(".") && it.name != "README.md" }
            ?.sortedBy { it.name }
            .orEmpty()
        require(files.isNotEmpty()) {
            "$src contains no shippable files. Did you run scripts/sync-models-from-export.sh?"
        }

        val totalMb = files.sumOf { it.length() } / 1_000_000.0
        logger.lifecycle(
            "Pushing %d files (%.1f MB) to %s ...".format(files.size, totalMb, target),
        )

        // Make sure the target exists; nuke anything that's not in the
        // current shipping bundle (covers stale v0 / experimental graphs).
        execOps.exec { commandLine("adb", "shell", "mkdir", "-p", target) }
        val keep = files.map { it.name }.toSet()
        val findCmd = "ls -1 $target 2>/dev/null"
        val out = ByteArrayOutputStream()
        execOps.exec {
            commandLine("adb", "shell", findCmd)
            standardOutput = out
            isIgnoreExitValue = true
        }
        val onDevice: List<String> = out.toString().lineSequence()
            .map { line: String -> line.trim() }
            .filter { line: String -> line.isNotEmpty() && !line.startsWith("ls:") }
            .toList()
        val stale = onDevice.filter { it !in keep }
        if (stale.isNotEmpty()) {
            logger.lifecycle("Removing ${stale.size} stale file(s): $stale")
            for (s in stale) {
                execOps.exec { commandLine("adb", "shell", "rm", "-f", "$target/$s") }
            }
        }

        for (f in files) {
            val sizeMb = f.length() / 1_000_000.0
            logger.lifecycle("  %-44s (%6.2f MB)".format(f.name, sizeMb))
            execOps.exec {
                commandLine("adb", "push", "--sync", f.absolutePath, "$target/${f.name}")
            }
        }
        logger.lifecycle("Done. Verify with: adb shell ls -lh $target")
    }
}

tasks.register<PushModelsTask>("pushModels") {
    group = "tts-nano"
    description =
        "adb push the v1.0.0 release bundle (~169 MB) from android/models/ to " +
        "/data/local/tmp/mosstts_models/ on the connected device."
    srcDir.set(rootProject.layout.projectDirectory.dir("models"))
    deviceDir.set("/data/local/tmp/mosstts_models")
}
