package com.afun.mosstts.core.engine

import android.content.Context
import android.util.Log
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
import com.afun.mosstts.core.text.TextNormalizer
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
    private var encoder: OrtSession? = null

    private var manifest: Manifest? = null
    private var cfg: ModelConfig? = null

    @Volatile private var cancelFlag = false

    /** Request cancellation of the current streaming generation. */
    fun cancelGeneration() { cancelFlag = true }
    private var tokenizer: Tokenizer? = null
    private var textNormalizer: TextNormalizer? = null
    private var voices: List<BuiltinVoice> = emptyList()

    val isInitialized: Boolean get() = env != null

    suspend fun initialize() = withContext(Dispatchers.Default) {
        val t0 = System.nanoTime()
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
        textNormalizer = TextNormalizer()

        voices = BuiltinVoice.loadAll(context)

        val e = OrtEnvironment.getEnvironment()
        env = e
        val cpuCount = Runtime.getRuntime().availableProcessors()
        val ttsThreads = maxOf(2, cpuCount - 1)
        val codecThreads = maxOf(2, cpuCount / 2)

        val ttsOpts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(ttsThreads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val codecOpts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(codecThreads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

        prefill = e.createSession(modelsDir.resolve("moss_tts_prefill.onnx").absolutePath, ttsOpts)
        decodeStep = e.createSession(modelsDir.resolve("moss_tts_decode_step.onnx").absolutePath, ttsOpts)
        localStep = e.createSession(modelsDir.resolve("moss_tts_local_cached_step.onnx").absolutePath, ttsOpts)
        decoderFull = e.createSession(modelsDir.resolve("moss_audio_tokenizer_decode_full.onnx").absolutePath, codecOpts)
        encoder = e.createSession(modelsDir.resolve("moss_audio_tokenizer_encode.onnx").absolutePath, codecOpts)

        Log.i(TAG, "initialized in ${(System.nanoTime() - t0) / 1_000_000}ms " +
                "(cpus=$cpuCount, ttsThreads=$ttsThreads, codecThreads=$codecThreads)")
    }

    fun getVoices(): List<BuiltinVoice> = voices

    /**
     * Encode a raw PCM reference audio into [AudioCodes] for voice cloning.
     * The PCM can be any sample rate / channel count; it will be resampled
     * to 48 kHz stereo and peak-normalised before encoding.
     */
    suspend fun encodeReferenceAudio(
        pcm: Array<FloatArray>,
        srcSampleRate: Int,
    ): AudioCodes = withContext(Dispatchers.Default) {
        val c = cfg ?: error("Engine not initialized")
        val e = env ?: error("Engine not initialized")
        val resampled = CodecRunner.resampleStereo(pcm, srcSampleRate)
        val normalised = CodecRunner.normalizePeak(resampled)
        val codec = CodecRunner(e, c, encoder = encoder, decoderFull = null)
        val t0 = System.nanoTime()
        val codes = codec.encode(normalised)
        Log.d(TAG, "encode reference: ${codes.frames} frames in ${(System.nanoTime() - t0) / 1_000_000}ms")
        codes
    }

    suspend fun generate(
        text: String,
        voiceIdx: Int,
        customCodes: AudioCodes? = null,
    ): GenerationResult = withContext(Dispatchers.Default) {
        val c = cfg ?: error("Engine not initialized")
        val m = manifest ?: error("Engine not initialized")
        val tok = tokenizer ?: error("Engine not initialized")
        val e = env ?: error("Engine not initialized")

        val normalizedText = textNormalizer?.normalize(text) ?: text

        val promptCodes = customCodes ?: run {
            val voice = voices.getOrNull(voiceIdx)
                ?: error("Voice index $voiceIdx out of range (${voices.size} available)")
            voice.toAudioCodes(c.nVq)
        }

        val textIds = tok.encode(normalizedText)
        require(textIds.isNotEmpty()) { "Text produced no tokens" }

        val builder = PromptBuilder(c, m.promptTemplates)
        val prompt = builder.buildVoiceClonePrompt(textIds, promptCodes)

        val gen = m.generationDefaults
        val textSampler = Sampler(
            temperature = TEXT_TEMPERATURE,
            topK = TEXT_TOP_K,
            topP = TEXT_TOP_P,
        )
        val audioSampler = Sampler(
            temperature = AUDIO_TEMPERATURE,
            topK = AUDIO_TOP_K,
            topP = AUDIO_TOP_P,
            repetitionPenalty = AUDIO_REPETITION_PENALTY,
        )

        val loop = InferenceLoop(e, prefill!!, decodeStep!!, localStep!!, c)

        val noStop = maxOf(MIN_FRAMES_NO_STOP, textIds.size * MIN_FRAMES_PER_TOKEN)
        val eosStart = maxOf(noStop, textIds.size * EXPECTED_FRAMES_PER_TOKEN)
        val t0 = System.currentTimeMillis()
        val res = loop.generate(
            prompt, textSampler, audioSampler, gen.maxNewFrames,
            eosNoStopBefore = noStop,
            eosBoostStart = eosStart,
            eosBoostPerFrame = EOS_BOOST_PER_FRAME,
        )
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
        customCodes: AudioCodes? = null,
        chunkFrames: Int = STREAM_CHUNK_FRAMES,
        onEvent: (StreamEvent) -> Unit,
    ) = withContext(Dispatchers.Default) {
        val c = cfg ?: error("Engine not initialized")
        val m = manifest ?: error("Engine not initialized")
        val tok = tokenizer ?: error("Engine not initialized")
        val e = env ?: error("Engine not initialized")

        val normalizedText = textNormalizer?.normalize(text) ?: text

        val promptCodes = customCodes ?: run {
            val voice = voices.getOrNull(voiceIdx)
                ?: error("Voice index $voiceIdx out of range (${voices.size} available)")
            voice.toAudioCodes(c.nVq)
        }

        val textIds = tok.encode(normalizedText)
        require(textIds.isNotEmpty()) { "Text produced no tokens" }

        val builder = PromptBuilder(c, m.promptTemplates)
        val prompt = builder.buildVoiceClonePrompt(textIds, promptCodes)

        val gen = m.generationDefaults
        val textSampler = Sampler(
            temperature = TEXT_TEMPERATURE,
            topK = TEXT_TOP_K,
            topP = TEXT_TOP_P,
        )
        val audioSampler = Sampler(
            temperature = AUDIO_TEMPERATURE,
            topK = AUDIO_TOP_K,
            topP = AUDIO_TOP_P,
            repetitionPenalty = AUDIO_REPETITION_PENALTY,
        )

        val loop = InferenceLoop(e, prefill!!, decodeStep!!, localStep!!, c)
        val channels = c.codecChannels
        val fadeShorts = CROSSFADE_SAMPLES * channels
        val t0 = System.currentTimeMillis()

        val noStop = maxOf(MIN_FRAMES_NO_STOP, textIds.size * MIN_FRAMES_PER_TOKEN)
        val eosStart = maxOf(noStop, textIds.size * EXPECTED_FRAMES_PER_TOKEN)
        val frameCap = gen.maxNewFrames

        val frameChannel = Channel<LongArray>(Channel.UNLIMITED)
        val progressCount = AtomicInteger(0)
        cancelFlag = false
        var totalGenerated = 0
        var finalSamplePos = 0

        coroutineScope {
            // ---- Producer: TTS inference → frames into channel ----
            launch(Dispatchers.Default) {
                val res = loop.generate(
                    prompt, textSampler, audioSampler, frameCap,
                    eosNoStopBefore = noStop,
                    eosBoostStart = eosStart,
                    eosBoostPerFrame = EOS_BOOST_PER_FRAME,
                ) { _, frameCodes ->
                    frameChannel.trySend(frameCodes.copyOf())
                    onEvent(StreamEvent.Progress(progressCount.incrementAndGet()))
                    !cancelFlag
                }
                totalGenerated = res.nGenerated
                frameChannel.close()
            }

            // ---- Consumer: channel → decode chunk → crossfade → emit ----
            // O(n) decoding: each chunk is decoded independently with a small
            // leading overlap for context. Total work = O(totalFrames) instead
            // of O(totalFrames^2) from re-decoding all accumulated frames.
            launch(Dispatchers.Default) {
                val accCodes = ArrayList<LongArray>(256)
                var emittedFrames = 0
                var holdback: ShortArray? = null

                fun decodeAndEmit(isFinal: Boolean) {
                    val totalFrames = accCodes.size
                    if (totalFrames <= emittedFrames) return

                    // Decode only a window: [contextStart .. totalFrames-1]
                    // Leading overlap gives the codec's attention left-context.
                    val contextStart = maxOf(0, emittedFrames - CONTEXT_OVERLAP_FRAMES)
                    val window = accCodes.subList(contextStart, totalFrames)
                    val pcm = decodeCodeWindow(window)

                    val totalSamples = pcm[0].size
                    // Estimate where emittedFrames falls in this window's sample space.
                    // frames decoded = totalFrames - contextStart
                    // samples per frame (approximate) = totalSamples / (totalFrames - contextStart)
                    val framesInWindow = totalFrames - contextStart
                    val newFrameOffset = emittedFrames - contextStart
                    val sampleOffset = if (framesInWindow > 0)
                        (totalSamples.toLong() * newFrameOffset / framesInWindow).toInt()
                    else 0

                    if (sampleOffset >= totalSamples) return

                    val newRaw = interleaveRange(pcm, sampleOffset, totalSamples, channels)
                    emittedFrames = totalFrames

                    val parts = ArrayList<ShortArray>()

                    if (holdback != null) {
                        val hbLen = minOf(holdback!!.size, newRaw.size, fadeShorts)
                        if (hbLen > 0) {
                            val fadeA = holdback!!.copyOfRange(holdback!!.size - hbLen, holdback!!.size)
                            val fadeB = newRaw.copyOfRange(0, hbLen)
                            parts.add(crossfade(fadeA, fadeB))
                            // Emit the non-overlapping portion of newRaw
                            if (newRaw.size > hbLen) {
                                val tail = newRaw.copyOfRange(hbLen, newRaw.size)
                                if (!isFinal && tail.size > fadeShorts) {
                                    parts.add(tail.copyOfRange(0, tail.size - fadeShorts))
                                    holdback = tail.copyOfRange(tail.size - fadeShorts, tail.size)
                                } else if (!isFinal) {
                                    holdback = tail
                                } else {
                                    parts.add(tail)
                                    holdback = null
                                }
                            } else {
                                holdback = null
                            }
                        } else {
                            parts.add(holdback!!)
                            holdback = null
                            if (!isFinal && newRaw.size > fadeShorts) {
                                parts.add(newRaw.copyOfRange(0, newRaw.size - fadeShorts))
                                holdback = newRaw.copyOfRange(newRaw.size - fadeShorts, newRaw.size)
                            } else if (!isFinal) {
                                holdback = newRaw
                            } else {
                                parts.add(newRaw)
                            }
                        }
                    } else {
                        if (!isFinal && newRaw.size > fadeShorts) {
                            parts.add(newRaw.copyOfRange(0, newRaw.size - fadeShorts))
                            holdback = newRaw.copyOfRange(newRaw.size - fadeShorts, newRaw.size)
                        } else if (!isFinal) {
                            holdback = newRaw
                        } else {
                            parts.add(newRaw)
                        }
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
                        decodeAndEmit(isFinal = false)
                    }
                }

                if (accCodes.size > emittedFrames) {
                    decodeAndEmit(isFinal = true)
                }
                if (holdback != null) {
                    onEvent(StreamEvent.AudioChunk(holdback!!, c.codecSampleRate, channels))
                    holdback = null
                }

                // Approximate duration: ~3840 samples per frame at 48kHz stereo codec
                finalSamplePos = accCodes.size * SAMPLES_PER_FRAME_APPROX
            }
        }

        val genMs = System.currentTimeMillis() - t0
        val durationSec = if (finalSamplePos > 0)
            finalSamplePos.toFloat() / c.codecSampleRate else 0f
        onEvent(StreamEvent.Done(totalGenerated, genMs, durationSec))
    }

    private fun decodeCodeWindow(codes: List<LongArray>): Array<FloatArray> {
        val c = cfg!!
        val e = env!!
        val nFrames = codes.size
        val flat = LongArray(nFrames * c.nVq)
        for (i in codes.indices) System.arraycopy(codes[i], 0, flat, i * c.nVq, c.nVq)
        val ac = AudioCodes(frames = nFrames, nVq = c.nVq, data = flat)
        val codec = CodecRunner(e, c, encoder = null, decoderFull = decoderFull)
        val t0 = System.nanoTime()
        val pcm = codec.decodeFull(ac)
        Log.d(TAG, "codec decode: ${nFrames} frames in ${(System.nanoTime() - t0) / 1_000_000}ms")
        return pcm
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
        encoder?.close(); encoder = null
        env = null
        manifest = null
        cfg = null
        tokenizer = null
    }

    companion object {
        private const val TAG = "TtsEngine"
        const val MODELS_DIR = "/data/local/tmp/mosstts_models"
        const val STREAM_CHUNK_FRAMES = 16
        /** 10 ms crossfade at 48 kHz to smooth chunk boundaries. */
        const val CROSSFADE_SAMPLES = 480
        /** Leading context frames for O(n) chunk decoding. Provides left-context
         *  to the codec's non-causal attention without re-decoding everything. */
        const val CONTEXT_OVERLAP_FRAMES = 4
        /** Approximate PCM samples per codec frame (48kHz / frame rate). */
        const val SAMPLES_PER_FRAME_APPROX = 3840

        // ---- EOS control: no-stop window + boost ----
        /** Minimum frames per token that MUST be generated before model
         *  is allowed to stop. Prevents premature stopping (漏字). */
        const val MIN_FRAMES_PER_TOKEN = 4
        /** Absolute floor for no-stop window. */
        const val MIN_FRAMES_NO_STOP = 8
        /** Estimated frames per text token for boost-start calculation. */
        const val EXPECTED_FRAMES_PER_TOKEN = 6
        /** Logit bonus added to `audio_end` each frame past the boost start.
         *  At +3.0/frame, 4 frames past start overwhelms even Δ≈12 (Junhao). */
        const val EOS_BOOST_PER_FRAME = 3.0f

        // ---- Text sampler: stochastic (model relies on sampling to emit EOS) ----
        const val TEXT_TEMPERATURE = 1.0f
        const val TEXT_TOP_K = 50
        const val TEXT_TOP_P = 1.0f

        // ---- Audio sampler parameters (official defaults — EOS boost handles
        //      termination, so these are optimized for quality not loop prevention) ----
        const val AUDIO_TEMPERATURE = 0.8f
        const val AUDIO_TOP_K = 25
        const val AUDIO_TOP_P = 0.95f
        const val AUDIO_REPETITION_PENALTY = 1.2f
    }
}
