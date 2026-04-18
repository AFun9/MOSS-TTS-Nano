package com.afun.mosstts.core.infer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.afun.mosstts.core.model.ModelConfig

/**
 * Wrapper around the 3 codec ORT sessions in the v1.0.0 release:
 *
 *   - **encode** (`moss_audio_tokenizer_encode.onnx`):
 *         48 kHz stereo float PCM `[1, 2, T]` -> audio_codes `[F, 16]` int64.
 *         Used by the voice-clone path to derive reference codes from a
 *         user-supplied wav.
 *   - **decodeFull** (`moss_audio_tokenizer_decode_full.onnx`):
 *         audio_codes `[1, F, 16]` -> 48 kHz stereo float PCM `[2, T]`.
 *         Used at the end of generation to materialise the wav file in
 *         one shot (~2-12 s utterances fit in a few hundred MB peak,
 *         well under any reasonable budget).
 *   - **decodeStep** (`moss_audio_tokenizer_decode_step.onnx`):
 *         streaming 1-frame-at-a-time variant with 12 attention layers'
 *         worth of explicit KV state on the inputs/outputs. Used by the
 *         M2.6 streaming player. NOT implemented here yet - left as a
 *         documented stub so the dependency graph is in place when we
 *         pick up the streaming work.
 *
 * Lifetime: `encode` / `decodeFull` sessions are instantiated by the
 * caller and held for the app's lifetime; passing them in (rather than
 * loading them here) keeps this class JVM-testable without ORT.
 *
 * All callers must close their own [OrtSession]s; this class does not
 * own them.
 */
class CodecRunner(
    private val env: OrtEnvironment,
    private val cfg: ModelConfig,
    private val encoder: OrtSession?,
    private val decoderFull: OrtSession?,
) {
    /**
     * Encode a stereo float PCM buffer at 48 kHz into [AudioCodes].
     *
     * Input layout: `[2, samples]`, channel-major, values in `[-1, 1]`.
     * If your source is mono / different sample rate, resample/expand
     * first (e.g. with [resampleStereo]).
     *
     * Returns the codes truncated to the encoder-reported `audio_code_lengths`
     * (the model pads up to the next stride internally).
     *
     * Caps the input to [maxSeconds] of audio - mirrors `demo_generate.py`'s
     * 20 s default for clone references; longer clips bloat prefill without
     * giving more timbre information.
     *
     * @throws IllegalStateException if the encoder session was not provided.
     */
    fun encode(pcm: Array<FloatArray>, maxSeconds: Float = 20f): AudioCodes {
        val enc = encoder ?: error("CodecRunner: encoder session not provided")
        require(pcm.size == cfg.codecChannels) {
            "expected $cfg.codecChannels channels, got ${pcm.size}"
        }
        val targetN = (maxSeconds * cfg.codecSampleRate).toInt()
        val nIn = pcm[0].size
        val n = if (nIn > targetN) targetN else nIn
        require(pcm.all { it.size >= n }) { "channels must be equally sized" }

        // Flatten to [1, 2, n] in channel-major order: c0[0..n], c1[0..n].
        val flat = FloatArray(cfg.codecChannels * n)
        for (c in 0 until cfg.codecChannels) {
            System.arraycopy(pcm[c], 0, flat, c * n, n)
        }
        val tWave = OrtTensors.floatTensor(
            env, flat, longArrayOf(1L, cfg.codecChannels.toLong(), n.toLong()),
        )
        val tLen = OrtTensors.intTensor(env, intArrayOf(n), longArrayOf(1L))

        val res = enc.run(linkedMapOf("waveform" to tWave, "input_lengths" to tLen))
        try {
            // audio_codes [1, F_padded, 16]  int32; audio_code_lengths [1] int32.
            val codesT = res.get(0) as OnnxTensor
            val lenT = res.get(1) as OnnxTensor

            @Suppress("UNCHECKED_CAST")
            val codes3D = codesT.value as Array<Array<IntArray>>
            check(codes3D.size == 1) { "expected leading dim 1" }
            val rowsPadded = codes3D[0]
            val codeLen = (lenT.value as IntArray)[0]

            val data = LongArray(codeLen * cfg.nVq)
            for (i in 0 until codeLen) {
                for (k in 0 until cfg.nVq) data[i * cfg.nVq + k] = rowsPadded[i][k].toLong()
            }
            return AudioCodes(frames = codeLen, nVq = cfg.nVq, data = data)
        } finally {
            res.close()
            tWave.close(); tLen.close()
        }
    }

    /**
     * Decode the full code grid in one pass into a stereo PCM buffer.
     *
     * Returns `[2, samples]` float arrays in `[-1, 1]`. Sample count is
     * read from the model's `audio_lengths` output; rows beyond it are
     * dropped automatically.
     *
     * @throws IllegalStateException if the decoder session was not provided.
     */
    fun decodeFull(codes: AudioCodes): Array<FloatArray> {
        val dec = decoderFull ?: error("CodecRunner: decoderFull session not provided")
        require(codes.nVq == cfg.nVq) { "codes.nVq=${codes.nVq} != cfg.nVq=${cfg.nVq}" }
        require(codes.frames > 0) { "no codes to decode" }

        val flatI32 = IntArray(codes.frames * cfg.nVq) { codes.data[it].toInt() }
        val tCodes = OrtTensors.intTensor(
            env, flatI32, longArrayOf(1L, codes.frames.toLong(), cfg.nVq.toLong()),
        )
        val tLen = OrtTensors.intTensor(env, intArrayOf(codes.frames), longArrayOf(1L))

        val res = dec.run(linkedMapOf("audio_codes" to tCodes, "audio_code_lengths" to tLen))
        try {
            val audioT = res.get(0) as OnnxTensor
            val lenT = res.get(1) as OnnxTensor

            @Suppress("UNCHECKED_CAST")
            val audio3D = audioT.value as Array<Array<FloatArray>>
            check(audio3D.size == 1) { "expected leading dim 1" }
            val channelsActual = audio3D[0].size
            val padded = audio3D[0]
            val nSamples = (lenT.value as IntArray)[0]
            val out = Array(channelsActual) { c ->
                val chSrc = padded[c]
                FloatArray(nSamples) { i -> chSrc[i] }
            }
            return out
        } finally {
            res.close()
            tCodes.close(); tLen.close()
        }
    }

    companion object {
        /**
         * Linear-interpolation resampler + mono->stereo expander.
         * Mirrors `demo_generate.py:_read_wav_for_codec` so a wav read
         * with `AudioFileSource` lines up byte-for-byte with the Python
         * reference clone path.
         *
         * Input is `[channels, T]`; output is `[2, T_out]` at [targetSr].
         */
        fun resampleStereo(pcm: Array<FloatArray>, srcSr: Int, targetSr: Int = ModelConfig.CODEC_SR): Array<FloatArray> {
            val channelsIn = pcm.size
            require(channelsIn >= 1) { "pcm must have at least 1 channel" }
            val nIn = pcm[0].size

            // First, fix channel count to 2 (duplicate mono / drop extras).
            val twoCh: Array<FloatArray> = when {
                channelsIn == ModelConfig.CODEC_CHANNELS -> pcm
                channelsIn == 1 -> arrayOf(pcm[0], pcm[0].copyOf())
                channelsIn > ModelConfig.CODEC_CHANNELS -> arrayOf(pcm[0], pcm[1])
                else -> error("unreachable: channelsIn=$channelsIn")
            }

            // Then, resample if needed (linear interpolation).
            if (srcSr == targetSr) return twoCh
            val nOut = (nIn.toLong() * targetSr / srcSr).toInt().coerceAtLeast(1)
            val out = Array(ModelConfig.CODEC_CHANNELS) { FloatArray(nOut) }
            // Maps i_out (0..nOut-1) -> i_in_float (0..nIn-1)
            val scale = if (nOut <= 1) 0.0 else (nIn - 1).toDouble() / (nOut - 1).toDouble()
            for (c in 0 until ModelConfig.CODEC_CHANNELS) {
                val src = twoCh[c]
                val dst = out[c]
                for (i in 0 until nOut) {
                    val pos = i * scale
                    val idx = pos.toInt()
                    val frac = (pos - idx).toFloat()
                    dst[i] = if (idx + 1 >= nIn) src[nIn - 1]
                             else src[idx] * (1f - frac) + src[idx + 1] * frac
                }
            }
            return out
        }

        /** Peak-normalise so abs(max) <= 1.0 (no-op if already in range). */
        fun normalizePeak(pcm: Array<FloatArray>): Array<FloatArray> {
            var peak = 0f
            for (ch in pcm) for (v in ch) {
                val a = if (v < 0f) -v else v
                if (a > peak) peak = a
            }
            if (peak <= 1f || peak == 0f) return pcm
            val inv = 1f / peak
            return Array(pcm.size) { c -> FloatArray(pcm[c].size) { i -> pcm[c][i] * inv } }
        }
    }
}
