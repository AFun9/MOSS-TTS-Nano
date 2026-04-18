package com.afun.mosstts.core.infer

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.afun.mosstts.core.model.ModelConfig
import com.afun.mosstts.core.model.ModelManager
import com.google.common.truth.Truth.assertThat
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * End-to-end on-device test: TTS generate -> codec decode -> WAV file.
 *
 * Cross-architecture byte-equality for autoregressive models is not
 * achievable (ARM64 NEON vs x86 AVX2 use different SIMD instructions
 * whose FP rounding cascades through the autoregressive loop). This
 * test validates the full pipeline produces a playable audio file that
 * a human can verify sounds correct.
 *
 * Re-uses scenario data from the `inference_traces_v2/trace.json`
 * fixture (same text, same prompt codes) so the inputs are deterministic.
 *
 * Output WAV: `/data/local/tmp/mosstts_models/e2e_<scenario_id>.wav`
 * Pull with:  `adb pull /data/local/tmp/mosstts_models/`
 */
@RunWith(AndroidJUnit4::class)
class E2EAudioGenerationTest {

    private val modelsDir = File("/data/local/tmp/mosstts_models")
    private val testCtx = InstrumentationRegistry.getInstrumentation().context

    @Test
    fun generateAndDecodeToWav() {
        assumeTrue(
            "Push the bundle first: `./gradlew :app:pushModels`",
            modelsDir.exists() && modelsDir.resolve("manifest.json").isFile,
        )

        val manifest = ModelManager.loadManifest(modelsDir)
        val cfg = ModelConfig.fromManifest(manifest)
        val trace = loadTrace()

        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

        val prefill = env.createSession(modelsDir.resolve("moss_tts_prefill.onnx").absolutePath, opts)
        val decodeStep = env.createSession(modelsDir.resolve("moss_tts_decode_step.onnx").absolutePath, opts)
        val localStep = env.createSession(modelsDir.resolve("moss_tts_local_cached_step.onnx").absolutePath, opts)
        val decoderFull = env.createSession(modelsDir.resolve("moss_audio_tokenizer_decode_full.onnx").absolutePath, opts)

        try {
            val loop = InferenceLoop(env, prefill, decodeStep, localStep, cfg)
            val codec = CodecRunner(env, cfg, encoder = null, decoderFull = decoderFull)
            val builder = PromptBuilder(cfg, manifest.promptTemplates)

            val maxFrames = 30
            val sampler = Sampler(temperature = 0f)

            for (scn in trace.scenarios) {
                val codes = AudioCodes(
                    frames = scn.prompt_audio_codes.frames,
                    nVq = scn.prompt_audio_codes.n_vq,
                    data = LongArray(scn.prompt_audio_codes.codes.size) {
                        scn.prompt_audio_codes.codes[it].toLong()
                    },
                )
                val prompt = builder.buildVoiceClonePrompt(
                    scn.text_token_ids.toIntArray(), codes,
                )

                println("E2E [${scn.id}]: generating $maxFrames frames ...")
                val t0 = System.currentTimeMillis()
                val res = loop.generate(prompt, sampler, sampler, maxFrames)
                val tGenMs = System.currentTimeMillis() - t0

                assertThat(res.nGenerated).isGreaterThan(0)
                println("E2E [${scn.id}]: ${res.nGenerated} frames, eos=${res.eosFrame}, " +
                        "gen=${tGenMs}ms")

                val audioCodes = AudioCodes(
                    frames = res.nGenerated,
                    nVq = cfg.nVq,
                    data = LongArray(res.nGenerated * cfg.nVq).also { flat ->
                        for (i in 0 until res.nGenerated)
                            System.arraycopy(res.codes[i], 0, flat, i * cfg.nVq, cfg.nVq)
                    },
                )

                val t1 = System.currentTimeMillis()
                val pcm = codec.decodeFull(audioCodes)
                val tDecMs = System.currentTimeMillis() - t1
                println("E2E [${scn.id}]: codec decode ${pcm[0].size} samples/ch, ${tDecMs}ms")

                val targetCtx = InstrumentationRegistry.getInstrumentation().targetContext
                val wavDir = File(targetCtx.filesDir, "e2e_wav").also { it.mkdirs() }
                val outFile = File(wavDir, "e2e_${scn.id}.wav")
                writeWav(outFile, pcm, cfg.codecSampleRate, cfg.codecChannels)
                println("E2E [${scn.id}]: ${outFile.absolutePath} (${outFile.length() / 1024} KB)")

                assertThat(pcm[0].size).isGreaterThan(0)
            }
        } finally {
            prefill.close()
            decodeStep.close()
            localStep.close()
            decoderFull.close()
            opts.close()
        }
    }

    private fun loadTrace(): TraceFile {
        val text = testCtx.assets.open("inference_traces_v2/trace.json")
            .bufferedReader().use { it.readText() }
        return Json { ignoreUnknownKeys = true }
            .decodeFromString(TraceFile.serializer(), text)
    }

    private fun writeWav(file: File, pcm: Array<FloatArray>, sampleRate: Int, channels: Int) {
        val nSamples = pcm[0].size
        val bitsPerSample = 16
        val bytesPerSample = bitsPerSample / 8
        val dataSize = nSamples * channels * bytesPerSample
        val buf = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)

        buf.put("RIFF".toByteArray(Charsets.US_ASCII))
        buf.putInt(36 + dataSize)
        buf.put("WAVE".toByteArray(Charsets.US_ASCII))
        buf.put("fmt ".toByteArray(Charsets.US_ASCII))
        buf.putInt(16)
        buf.putShort(1)
        buf.putShort(channels.toShort())
        buf.putInt(sampleRate)
        buf.putInt(sampleRate * channels * bytesPerSample)
        buf.putShort((channels * bytesPerSample).toShort())
        buf.putShort(bitsPerSample.toShort())
        buf.put("data".toByteArray(Charsets.US_ASCII))
        buf.putInt(dataSize)

        for (i in 0 until nSamples) {
            for (c in 0 until channels) {
                val clamped = pcm[c][i].coerceIn(-1f, 1f)
                val s16 = (clamped * 32767f).toInt().toShort()
                buf.putShort(s16)
            }
        }

        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(0)
            raf.write(buf.array())
        }
    }

    @Serializable
    private data class TraceFile(
        val scenarios: List<TraceScenario>,
    )

    @Serializable
    private data class TraceScenario(
        val id: String,
        val text_token_ids: List<Int>,
        val prompt_audio_codes: PromptCodes,
    )

    @Serializable
    private data class PromptCodes(
        val frames: Int,
        val n_vq: Int,
        val codes: List<Int>,
    )
}
