package com.afun.mosstts.core.engine

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.afun.mosstts.core.infer.AudioCodes
import com.afun.mosstts.core.infer.CodecRunner
import com.afun.mosstts.core.infer.InferenceLoop
import com.afun.mosstts.core.infer.PromptBuilder
import com.afun.mosstts.core.infer.Sampler
import com.afun.mosstts.core.model.Manifest
import com.afun.mosstts.core.model.ModelConfig
import com.afun.mosstts.core.model.ModelManager
import com.afun.mosstts.core.tokenizer.Tokenizer
import com.afun.mosstts.core.voice.BuiltinVoice
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

data class GenerationResult(
    val pcm: Array<FloatArray>,
    val sampleRate: Int,
    val channels: Int,
    val generationTimeMs: Long,
    val framesGenerated: Int,
)

sealed interface StreamEvent {
    data class Progress(val frame: Int) : StreamEvent
    data class AudioChunk(val pcmShorts: ShortArray, val sampleRate: Int, val channels: Int) : StreamEvent
    data class Done(val totalFrames: Int, val genTimeMs: Long, val durationSec: Float) : StreamEvent
}

/**
 * Facade over the full TTS pipeline: tokenize -> prompt -> generate -> decode.
 *
 * Call [initialize] once from a coroutine, then [generate] as many times as
 * needed. Call [release] when done (e.g. in `ViewModel.onCleared`).
 *
 * All heavy work runs on [Dispatchers.Default] via `withContext`.
 */
class TtsEngine(private val context: Context) {

    private var env: OrtEnvironment? = null
    private var prefill: OrtSession? = null
    private var decodeStep: OrtSession? = null
    private var localStep: OrtSession? = null
    private var decoderFull: OrtSession? = null

    private var manifest: Manifest? = null
    private var cfg: ModelConfig? = null
    private var tokenizer: Tokenizer? = null
    private var voices: List<BuiltinVoice> = emptyList()

    val isInitialized: Boolean get() = env != null

    suspend fun initialize() = withContext(Dispatchers.Default) {
        val modelsDir = File(MODELS_DIR)
        require(modelsDir.exists() && modelsDir.resolve("manifest.json").isFile) {
            "Model bundle not found at $MODELS_DIR. Run `./gradlew :app:pushModels` first."
        }

        val m = ModelManager.loadManifest(modelsDir)
        manifest = m
        cfg = ModelConfig.fromManifest(m)

        val tokJson = context.assets.open("tokenizer_kotlin.json")
            .bufferedReader().use { it.readText() }
        tokenizer = Tokenizer.load(tokJson)

        voices = BuiltinVoice.loadAll(context)

        val e = OrtEnvironment.getEnvironment()
        env = e
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

        prefill = e.createSession(modelsDir.resolve("moss_tts_prefill.onnx").absolutePath, opts)
        decodeStep = e.createSession(modelsDir.resolve("moss_tts_decode_step.onnx").absolutePath, opts)
        localStep = e.createSession(modelsDir.resolve("moss_tts_local_cached_step.onnx").absolutePath, opts)
        decoderFull = e.createSession(modelsDir.resolve("moss_audio_tokenizer_decode_full.onnx").absolutePath, opts)
    }

    fun getVoices(): List<BuiltinVoice> = voices

    suspend fun generate(text: String, voiceIdx: Int): GenerationResult = withContext(Dispatchers.Default) {
        val c = cfg ?: error("Engine not initialized")
        val m = manifest ?: error("Engine not initialized")
        val tok = tokenizer ?: error("Engine not initialized")
        val e = env ?: error("Engine not initialized")

        val voice = voices.getOrNull(voiceIdx)
            ?: error("Voice index $voiceIdx out of range (${voices.size} available)")
        val promptCodes = voice.toAudioCodes(c.nVq)

        val textIds = tok.encode(text)
        require(textIds.isNotEmpty()) { "Text produced no tokens" }

        val builder = PromptBuilder(c, m.promptTemplates)
        val prompt = builder.buildVoiceClonePrompt(textIds, promptCodes)

        val gen = m.generationDefaults
        val textSampler = Sampler(
            temperature = gen.textTemperature,
            topK = gen.textTopK,
            topP = gen.textTopP,
        )
        val audioSampler = Sampler(
            temperature = gen.audioTemperature,
            topK = gen.audioTopK,
            topP = gen.audioTopP,
            repetitionPenalty = gen.audioRepetitionPenalty,
        )

        val loop = InferenceLoop(e, prefill!!, decodeStep!!, localStep!!, c)

        val t0 = System.currentTimeMillis()
        val res = loop.generate(prompt, textSampler, audioSampler, gen.maxNewFrames)
        val genMs = System.currentTimeMillis() - t0

        require(res.nGenerated > 0) { "Model generated 0 frames" }

        val flat = LongArray(res.nGenerated * c.nVq)
        for (i in 0 until res.nGenerated) {
            System.arraycopy(res.codes[i], 0, flat, i * c.nVq, c.nVq)
        }
        val audioCodes = AudioCodes(frames = res.nGenerated, nVq = c.nVq, data = flat)

        val codec = CodecRunner(e, c, encoder = null, decoderFull = decoderFull)
        val pcm = codec.decodeFull(audioCodes)

        GenerationResult(
            pcm = pcm,
            sampleRate = c.codecSampleRate,
            channels = c.codecChannels,
            generationTimeMs = genMs,
            framesGenerated = res.nGenerated,
        )
    }

    /**
     * Streaming variant: emits [StreamEvent.AudioChunk] every [chunkFrames]
     * generated TTS frames via a producer-consumer pipeline, so inference
     * and codec decoding run on separate threads — inference never stalls.
     *
     * A 10 ms linear crossfade at chunk boundaries smooths the minor
     * sample drift caused by the codec's non-causal attention layers
     * when re-decoding a longer accumulated sequence.
     */
    suspend fun generateStreaming(
        text: String,
        voiceIdx: Int,
        chunkFrames: Int = STREAM_CHUNK_FRAMES,
        onEvent: (StreamEvent) -> Unit,
    ) = withContext(Dispatchers.Default) {
        val c = cfg ?: error("Engine not initialized")
        val m = manifest ?: error("Engine not initialized")
        val tok = tokenizer ?: error("Engine not initialized")
        val e = env ?: error("Engine not initialized")

        val voice = voices.getOrNull(voiceIdx)
            ?: error("Voice index $voiceIdx out of range (${voices.size} available)")
        val promptCodes = voice.toAudioCodes(c.nVq)

        val textIds = tok.encode(text)
        require(textIds.isNotEmpty()) { "Text produced no tokens" }

        val builder = PromptBuilder(c, m.promptTemplates)
        val prompt = builder.buildVoiceClonePrompt(textIds, promptCodes)

        val gen = m.generationDefaults
        val textSampler = Sampler(
            temperature = gen.textTemperature,
            topK = gen.textTopK,
            topP = gen.textTopP,
        )
        val audioSampler = Sampler(
            temperature = gen.audioTemperature,
            topK = gen.audioTopK,
            topP = gen.audioTopP,
            repetitionPenalty = gen.audioRepetitionPenalty,
        )

        val loop = InferenceLoop(e, prefill!!, decodeStep!!, localStep!!, c)
        val channels = c.codecChannels
        val fadeShorts = CROSSFADE_SAMPLES * channels
        val t0 = System.currentTimeMillis()

        val frameChannel = Channel<LongArray>(Channel.UNLIMITED)
        val progressCount = AtomicInteger(0)
        var totalGenerated = 0
        var finalSamplePos = 0

        coroutineScope {
            // ---- Producer: TTS inference → frames into channel ----
            launch(Dispatchers.Default) {
                val res = loop.generate(
                    prompt, textSampler, audioSampler, gen.maxNewFrames,
                ) { _, frameCodes ->
                    frameChannel.trySend(frameCodes.copyOf())
                    onEvent(StreamEvent.Progress(progressCount.incrementAndGet()))
                }
                totalGenerated = res.nGenerated
                frameChannel.close()
            }

            // ---- Consumer: channel → accumulate → decode → emit ----
            launch(Dispatchers.Default) {
                val accCodes = ArrayList<LongArray>(256)
                var lastDecodedPos = 0
                var holdback: ShortArray? = null

                fun emitChunk(pcm: Array<FloatArray>, isFinal: Boolean) {
                    val totalSamples = pcm[0].size
                    if (totalSamples <= lastDecodedPos) return

                    val parts = ArrayList<ShortArray>()

                    if (holdback != null) {
                        val hbSamples = holdback!!.size / channels
                        val overlapStart = maxOf(0, lastDecodedPos - hbSamples)
                        val overlapEnd = lastDecodedPos
                        if (overlapEnd > overlapStart) {
                            val overlapNew = interleaveRange(pcm, overlapStart, overlapEnd, channels)
                            parts.add(crossfade(holdback!!, overlapNew))
                        } else {
                            parts.add(holdback!!)
                        }
                        holdback = null
                    }

                    val newRaw = interleaveRange(pcm, lastDecodedPos, totalSamples, channels)
                    lastDecodedPos = totalSamples

                    if (!isFinal && newRaw.size > fadeShorts) {
                        val cutoff = newRaw.size - fadeShorts
                        parts.add(newRaw.copyOfRange(0, cutoff))
                        holdback = newRaw.copyOfRange(cutoff, newRaw.size)
                    } else if (!isFinal && newRaw.size <= fadeShorts) {
                        holdback = newRaw
                        if (parts.isNotEmpty()) {
                            onEvent(StreamEvent.AudioChunk(
                                mergeShorts(parts), c.codecSampleRate, channels,
                            ))
                        }
                        return
                    } else {
                        parts.add(newRaw)
                    }

                    if (parts.isNotEmpty()) {
                        onEvent(StreamEvent.AudioChunk(
                            mergeShorts(parts), c.codecSampleRate, channels,
                        ))
                    }
                }

                for (frame in frameChannel) {
                    accCodes.add(frame)
                    if (accCodes.size % chunkFrames == 0) {
                        val pcm = decodeAccumulatedCodes(accCodes)
                        emitChunk(pcm, isFinal = false)
                    }
                }

                if (accCodes.isNotEmpty()) {
                    val pcm = decodeAccumulatedCodes(accCodes)
                    emitChunk(pcm, isFinal = true)
                }
                if (holdback != null) {
                    onEvent(StreamEvent.AudioChunk(holdback!!, c.codecSampleRate, channels))
                    holdback = null
                }

                finalSamplePos = lastDecodedPos
            }
        }

        val genMs = System.currentTimeMillis() - t0
        val durationSec = if (finalSamplePos > 0)
            finalSamplePos.toFloat() / c.codecSampleRate else 0f
        onEvent(StreamEvent.Done(totalGenerated, genMs, durationSec))
    }

    private fun decodeAccumulatedCodes(codes: List<LongArray>): Array<FloatArray> {
        val c = cfg!!
        val e = env!!
        val nFrames = codes.size
        val flat = LongArray(nFrames * c.nVq)
        for (i in codes.indices) System.arraycopy(codes[i], 0, flat, i * c.nVq, c.nVq)
        val ac = AudioCodes(frames = nFrames, nVq = c.nVq, data = flat)
        val codec = CodecRunner(e, c, encoder = null, decoderFull = decoderFull)
        return codec.decodeFull(ac)
    }

    private fun interleaveRange(
        pcm: Array<FloatArray>,
        startSample: Int,
        endSample: Int,
        channels: Int,
    ): ShortArray {
        val n = endSample - startSample
        val out = ShortArray(n * channels)
        for (i in 0 until n) {
            for (c in 0 until channels) {
                val clamped = pcm[c][startSample + i].coerceIn(-1f, 1f)
                out[i * channels + c] = (clamped * 32767f).toInt().toShort()
            }
        }
        return out
    }

    private fun crossfade(a: ShortArray, b: ShortArray): ShortArray {
        val len = minOf(a.size, b.size)
        val out = ShortArray(len)
        for (i in 0 until len) {
            val t = i.toFloat() / len
            val mixed = a[i].toFloat() * (1f - t) + b[i].toFloat() * t
            out[i] = mixed.toInt().coerceIn(-32768, 32767).toShort()
        }
        return out
    }

    private fun mergeShorts(parts: List<ShortArray>): ShortArray {
        val total = parts.sumOf { it.size }
        val out = ShortArray(total)
        var pos = 0
        for (p in parts) {
            System.arraycopy(p, 0, out, pos, p.size)
            pos += p.size
        }
        return out
    }

    fun release() {
        prefill?.close(); prefill = null
        decodeStep?.close(); decodeStep = null
        localStep?.close(); localStep = null
        decoderFull?.close(); decoderFull = null
        env = null
        manifest = null
        cfg = null
        tokenizer = null
    }

    companion object {
        const val MODELS_DIR = "/data/local/tmp/mosstts_models"
        const val STREAM_CHUNK_FRAMES = 16
        /** 10 ms crossfade at 48 kHz to smooth chunk boundaries. */
        const val CROSSFADE_SAMPLES = 480
    }
}
