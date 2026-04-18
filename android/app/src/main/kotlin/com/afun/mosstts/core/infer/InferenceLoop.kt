package com.afun.mosstts.core.infer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.afun.mosstts.core.model.ModelConfig

/**
 * Autoregressive code generator - Kotlin port of
 * `export_v2/scripts/demo_generate.py:generate` (the v1.0.0 release).
 *
 * Three ORT sessions drive the loop, mirroring the official 4-graph
 * protocol:
 *
 *   1. **prefill** (1x per utterance): full prompt -> last hidden + 12-layer
 *      global KV cache (`present_key/value_0..11`).
 *   2. **decode_step** (1x per generated frame): single-row append; consumes
 *      the previous global KV and `past_valid_lengths`, emits the new
 *      hidden + grown KV.
 *   3. **local_cached_step** (17x per generated frame): one text-head
 *      decision (step_type=0) followed by 16 audio-head channels
 *      (step_type=1 then 15x step_type=2), each step using a **per-frame**
 *      local KV cache (1 layer, reset to empty at the start of every
 *      frame). The 16 sampled codes plus the assistant_slot token become
 *      the next decode_step input row.
 *
 * The local_decoder.onnx graph is loaded by `make_release.py` for
 * completeness / debug parity, but is *not* used by this loop.
 *
 * Per-utterance lifecycle:
 *
 * ```
 * val loop = InferenceLoop(env, prefill, decodeStep, localCachedStep, config)
 * val res  = loop.generate(prompt, textSampler, audioSampler, maxFrames)
 * // res.codes is [n_generated, n_vq] long, ready for codec decode
 * ```
 *
 * `OnnxTensor` lifetime: each iteration appends every tensor it owns into
 * `iterScope` and unconditionally closes them in `finally`. Tensors that
 * survive into the next iteration (the global KV cache held in
 * `globalKv`, plus the per-frame local KV swapped in `localKv`) are owned
 * outside `iterScope` and rotated explicitly.
 *
 * Sampling: text head decides only between `audio_assistant_slot` and
 * `audio_end` - it samples from the 2-element subset of the 16384-vocab
 * logits (matches `demo_generate.py:sample_token` over `cand_logits`).
 * Audio head samples each of the 16 codebooks independently with
 * repetition penalty over the *channel-local* history of previously
 * sampled codes for that codebook.
 *
 * Stateless across [generate] calls; the underlying [OrtSession]s may be
 * shared across loops but should be serialised externally if your
 * `SessionOptions` allow concurrent runs.
 */
class InferenceLoop(
    private val env: OrtEnvironment,
    private val prefill: OrtSession,
    private val decodeStep: OrtSession,
    private val localCachedStep: OrtSession,
    private val cfg: ModelConfig,
) {
    // Global `past_X` / `present_X` names, one pair per layer, in declared order.
    private val gtPastKeyNames: List<String> =
        (0 until cfg.numGlobalLayers).map { "past_key_$it" }
    private val gtPastValueNames: List<String> =
        (0 until cfg.numGlobalLayers).map { "past_value_$it" }

    // Local `past_X` / `present_X` names, one pair per layer (1 in v1.0.0).
    private val lcPastKeyNames: List<String> =
        (0 until cfg.numLocalLayers).map { "local_past_key_$it" }
    private val lcPastValueNames: List<String> =
        (0 until cfg.numLocalLayers).map { "local_past_value_$it" }

    fun generate(
        prompt: Prompt,
        textSampler: Sampler,
        audioSampler: Sampler,
        maxFrames: Int,
        onFrameCommitted: ((frameIdx: Int, frame: LongArray) -> Unit)? = null,
    ): GenResult {
        require(maxFrames > 0) { "maxFrames must be positive, got $maxFrames" }
        require(prompt.rowStride == cfg.rowStride) {
            "prompt.rowStride=${prompt.rowStride} does not match cfg.rowStride=${cfg.rowStride}"
        }

        val nVq = cfg.nVq
        val audioSlot = cfg.audioAssistantSlotTokenId
        val audioEnd = cfg.audioEndTokenId

        val historyBuf = Array(maxFrames) { LongArray(nVq) }
        val frameTokens = LongArray(nVq)
        var nGenerated = 0
        var eosFrame: Int? = null

        // KV cache lifetime contract:
        //   ORT-Java's `OrtSession.Result.close()` releases EVERY `OnnxValue`
        //   it owns (verified in v1.24.3 source). The KV tensors we feed into
        //   the *next* `run()` are still backed by the *previous* `Result`'s
        //   native memory, so we must hold that `Result` open until the next
        //   step has consumed it. We package each `(Result + List<KvTensor>)`
        //   pair into a [KvHolder] and rotate holders the same way we used to
        //   rotate raw lists - just close the previous holder, never the
        //   tensors directly.
        //
        // Initial empty local KV is allocated by us (NOT from a Result), so
        // it's wrapped in a holder with `res = null`.
        var globalKv: KvHolder = KvHolder(null, emptyList())
        var lastHidden: FloatArray
        var pastValidLength: Int

        try {
            // ---- 1) PREFILL --------------------------------------------------
            val (firstHidden, firstKv, prefillSeq) = runPrefill(prompt)
            lastHidden = firstHidden            // [hidden]
            globalKv = firstKv                  // KvHolder owning prefill Result
            pastValidLength = prefillSeq

            // ---- 2) Generate frames ----------------------------------------
            frameLoop@ for (frameIdx in 0 until maxFrames) {
                // Local KV resets to empty at the start of every frame.
                var localKv: KvHolder = freshEmptyLocalKv()

                try {
                    // ---- 2a) text head (step_type=0) -----------------------
                    val (textLogits, _, textKv) = runLocalStep(
                        globalHidden = lastHidden,
                        textTokenId = 0,
                        audioTokenId = 0,
                        channelIndex = 0,
                        stepType = 0,
                        localPastKv = localKv.tensors,
                    )
                    localKv.close(); localKv = textKv

                    // text head must decide between {assistant_slot, audio_end}.
                    val candIds = intArrayOf(audioSlot, audioEnd)
                    val candLogits = FloatArray(candIds.size) { textLogits[candIds[it]] }
                    val sampledIdx = textSampler.sample(candLogits, previousIds = null)
                    val nextTextToken = candIds[sampledIdx]
                    closeFloatLogits(textLogits)

                    if (nextTextToken == audioEnd) {
                        eosFrame = frameIdx
                        localKv.close()
                        break@frameLoop
                    }

                    // ---- 2b) 16 audio channels (step_type=1, then 15x step_type=2) ---
                    for (ch in 0 until nVq) {
                        val stepType: Int
                        val textArg: Int
                        val audioArg: Int
                        val chanArg: Int
                        if (ch == 0) {
                            stepType = 1
                            textArg = nextTextToken
                            audioArg = 0
                            chanArg = 0
                        } else {
                            stepType = 2
                            textArg = 0
                            audioArg = frameTokens[ch - 1].toInt()
                            chanArg = ch - 1
                        }

                        val (_, audioLogits3D, newKv) = runLocalStep(
                            globalHidden = lastHidden,
                            textTokenId = textArg,
                            audioTokenId = audioArg,
                            channelIndex = chanArg,
                            stepType = stepType,
                            localPastKv = localKv.tensors,
                        )
                        localKv.close(); localKv = newKv

                        // audioLogits3D is [1, 16, 1024]; we take channel ch.
                        val chLogits = audioLogits3D[ch]
                        val previousIds: IntArray? = if (nGenerated > 0) {
                            IntArray(nGenerated) { historyBuf[it][ch].toInt() }
                        } else null
                        val tok = audioSampler.sample(chLogits, previousIds = previousIds)
                        frameTokens[ch] = tok.toLong()
                    }

                    // ---- 2c) commit frame ---------------------------------
                    System.arraycopy(frameTokens, 0, historyBuf[nGenerated], 0, nVq)
                    nGenerated += 1
                    onFrameCommitted?.invoke(nGenerated - 1, historyBuf[nGenerated - 1])

                    // ---- 2d) decode_step: feed [audio_slot, frame_codes...] ---
                    val nextRow = IntArray(nVq + 1).also {
                        it[0] = audioSlot
                        for (k in 0 until nVq) it[k + 1] = frameTokens[k].toInt()
                    }
                    val (newHidden, newGlobalKv, newPastLen) = runDecodeStep(
                        nextRow = nextRow,
                        pastValidLength = pastValidLength,
                        pastKv = globalKv.tensors,
                    )
                    globalKv.close()
                    globalKv = newGlobalKv
                    lastHidden = newHidden
                    pastValidLength = newPastLen
                } finally {
                    localKv.close()
                }
            }
        } finally {
            globalKv.close()
        }

        val codes = Array(nGenerated) { i -> historyBuf[i].copyOf() }
        return GenResult(codes = codes, nGenerated = nGenerated, eosFrame = eosFrame)
    }

    // ---- session wrappers ---------------------------------------------

    /**
     * Lifetime guard for KV cache tensors.
     *
     * `OrtSession.Result.close()` releases every `OnnxValue` it owns
     * (verified in ORT-Java v1.24.3 source - see [generate]'s contract
     * comment). So the moment we close the [Result] that produced our
     * KV tensors, those tensors' native pointers become invalid; feeding
     * them into the *next* `Session.run()` corrupts the input table and
     * causes a SIGSEGV that surfaces while iterating input names
     * (fault addr looks like an ASCII chunk of "past_valid_lengths" /
     * "local_past_key_0" - this was the exact symptom on Mi/Redmi 24122RKC7C
     * with libonnxruntime.so 1.24.3 arm64).
     *
     * A holder pins both together: the tensor list lives as long as the
     * holder; closing the holder closes the underlying [Result] (which
     * cascades to all KV tensors). For the very-first empty local KV
     * (allocated by us, not from a `Result`), `res = null` and we close
     * the tensors directly.
     */
    private class KvHolder(
        private val res: OrtSession.Result?,
        val tensors: List<OnnxTensor>,
    ) : AutoCloseable {
        private var closed = false
        override fun close() {
            if (closed) return
            closed = true
            if (res != null) {
                res.close() // releases all OnnxValues it owns, including our KV
            } else {
                for (t in tensors) t.close()
            }
        }
    }

    private data class PrefillOut(
        val lastHidden: FloatArray,
        val kv: KvHolder,
        val seqLen: Int,
    )

    private fun runPrefill(prompt: Prompt): PrefillOut {
        val seqLen = prompt.seqLen
        val inputIdsI32 = IntArray(prompt.inputIds.size) { prompt.inputIds[it].toInt() }
        val attnI32 = IntArray(seqLen) { 1 }

        val tInputIds = OrtTensors.intTensor(
            env, inputIdsI32, longArrayOf(1L, seqLen.toLong(), cfg.rowStride.toLong()),
        )
        val tAttn = OrtTensors.intTensor(env, attnI32, longArrayOf(1L, seqLen.toLong()))

        val feed = LinkedHashMap<String, OnnxTensor>(2).apply {
            put("input_ids", tInputIds)
            put("attention_mask", tAttn)
        }
        val res = prefill.run(feed)
        try {
            // Output 0 = global_hidden [1, T, 768]; rest are KV pairs.
            // Materialise the hidden into a host FloatArray now so we can
            // hand back a holder that only owns the KV outputs.
            val hiddenT = res.get(0) as OnnxTensor
            val lastHidden = OrtTensors.readFloat3DLastRow(hiddenT)
            val kvList = ArrayList<OnnxTensor>(2 * cfg.numGlobalLayers)
            for (i in 1 until res.size()) kvList += res.get(i) as OnnxTensor
            // Holder takes over `res` ownership; do NOT close res here.
            return PrefillOut(lastHidden = lastHidden, kv = KvHolder(res, kvList), seqLen = seqLen)
        } finally {
            tInputIds.close(); tAttn.close()
        }
    }

    private data class DecodeStepOut(
        val lastHidden: FloatArray,
        val kv: KvHolder,
        val newPastLen: Int,
    )

    private fun runDecodeStep(
        nextRow: IntArray,
        pastValidLength: Int,
        pastKv: List<OnnxTensor>,
    ): DecodeStepOut {
        require(nextRow.size == cfg.rowStride) { "nextRow size=${nextRow.size} != rowStride=${cfg.rowStride}" }
        require(pastKv.size == 2 * cfg.numGlobalLayers) {
            "pastKv size=${pastKv.size} != 2*numGlobalLayers=${2 * cfg.numGlobalLayers}"
        }
        val tInputIds = OrtTensors.intTensor(env, nextRow, longArrayOf(1L, 1L, cfg.rowStride.toLong()))
        val tPastLen = OrtTensors.intTensor(env, intArrayOf(pastValidLength), longArrayOf(1L))

        val feed = LinkedHashMap<String, OnnxTensor>(2 + 2 * cfg.numGlobalLayers).apply {
            put("input_ids", tInputIds)
            put("past_valid_lengths", tPastLen)
            for (i in 0 until cfg.numGlobalLayers) {
                put(gtPastKeyNames[i], pastKv[2 * i])
                put(gtPastValueNames[i], pastKv[2 * i + 1])
            }
        }
        val res = decodeStep.run(feed)
        try {
            val hiddenT = res.get(0) as OnnxTensor
            val lastHidden = OrtTensors.readFloat3DLastRow(hiddenT)
            val kvList = ArrayList<OnnxTensor>(2 * cfg.numGlobalLayers)
            for (i in 1 until res.size()) kvList += res.get(i) as OnnxTensor
            return DecodeStepOut(
                lastHidden = lastHidden,
                kv = KvHolder(res, kvList),
                newPastLen = pastValidLength + 1,
            )
        } finally {
            tInputIds.close(); tPastLen.close()
        }
    }

    /**
     * One sub-step of the local decoder. Returns the materialised text
     * logits (FloatArray[16384]), the **3-D audio logits as [16][1024]**
     * (FloatArray[16][1024], indexed by codebook channel), and a [KvHolder]
     * that owns the new local KV tensors (and the underlying [Result]).
     */
    private data class LocalStepOut(
        val textLogits: FloatArray,
        val audioLogits: Array<FloatArray>,
        val kv: KvHolder,
    )

    private fun runLocalStep(
        globalHidden: FloatArray,
        textTokenId: Int,
        audioTokenId: Int,
        channelIndex: Int,
        stepType: Int,
        localPastKv: List<OnnxTensor>,
    ): LocalStepOut {
        require(globalHidden.size == cfg.hiddenSize) {
            "globalHidden size=${globalHidden.size} != hiddenSize=${cfg.hiddenSize}"
        }
        require(localPastKv.size == 2 * cfg.numLocalLayers) {
            "localPastKv size=${localPastKv.size} != 2*numLocalLayers=${2 * cfg.numLocalLayers}"
        }

        // Past length is read from the [1, seq, 12, 64] tensor's seq dim.
        val pastSeq = if (localPastKv.isEmpty()) 0
                      else localPastKv[0].info.shape[1].toInt()

        val tHidden = OrtTensors.floatTensor(
            env, globalHidden, longArrayOf(1L, cfg.hiddenSize.toLong()),
        )
        val tTextTok = OrtTensors.intTensor(env, intArrayOf(textTokenId), longArrayOf(1L))
        val tAudioTok = OrtTensors.intTensor(env, intArrayOf(audioTokenId), longArrayOf(1L))
        val tChan = OrtTensors.intTensor(env, intArrayOf(channelIndex), longArrayOf(1L))
        val tStep = OrtTensors.intTensor(env, intArrayOf(stepType), longArrayOf(1L))
        val tPastLen = OrtTensors.intTensor(env, intArrayOf(pastSeq), longArrayOf(1L))

        val feed = LinkedHashMap<String, OnnxTensor>(6 + 2 * cfg.numLocalLayers).apply {
            put("global_hidden", tHidden)
            put("text_token_id", tTextTok)
            put("audio_token_id", tAudioTok)
            put("channel_index", tChan)
            put("step_type", tStep)
            put("past_valid_lengths", tPastLen)
            for (i in 0 until cfg.numLocalLayers) {
                put(lcPastKeyNames[i], localPastKv[2 * i])
                put(lcPastValueNames[i], localPastKv[2 * i + 1])
            }
        }

        val res = localCachedStep.run(feed)
        try {
            val textLogitsT = res.get(0) as OnnxTensor
            val audioLogitsT = res.get(1) as OnnxTensor

            val textLogits = OrtTensors.readFloat1D(textLogitsT)
            // audio_logits is [1, 16, 1024] -> Array<FloatArray>(16 of size 1024).
            @Suppress("UNCHECKED_CAST")
            val audioOuter = audioLogitsT.value as Array<Array<FloatArray>>
            check(audioOuter.size == 1) { "audio_logits leading dim != 1" }
            val audioRows = audioOuter[0]

            val kvList = ArrayList<OnnxTensor>(2 * cfg.numLocalLayers)
            for (i in 2 until res.size()) kvList += res.get(i) as OnnxTensor

            return LocalStepOut(
                textLogits = textLogits,
                audioLogits = audioRows,
                kv = KvHolder(res, kvList),
            )
        } finally {
            tHidden.close(); tTextTok.close(); tAudioTok.close()
            tChan.close(); tStep.close(); tPastLen.close()
        }
    }

    // ---- helpers ------------------------------------------------------

    private fun freshEmptyLocalKv(): KvHolder {
        val shape = longArrayOf(1L, 0L, cfg.numHeads.toLong(), cfg.headDim.toLong())
        val tensors = List(2 * cfg.numLocalLayers) { OrtTensors.emptyFloatTensor(env, shape) }
        return KvHolder(res = null, tensors = tensors)
    }

    @Suppress("UNUSED_PARAMETER")
    private fun closeFloatLogits(arr: FloatArray) {
        // FloatArray was already materialised host-side; nothing to release.
        // Marker exists to keep the call sites symmetric and signal intent.
    }

    /** Final output of a [generate] call. `codes` is `[n_generated, n_vq]`. */
    data class GenResult(
        val codes: Array<LongArray>,
        val nGenerated: Int,
        /** Frame index at which the text head sampled `audio_end`, or null if max reached. */
        val eosFrame: Int?,
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is GenResult) return false
            if (nGenerated != other.nGenerated || eosFrame != other.eosFrame) return false
            if (codes.size != other.codes.size) return false
            for (i in codes.indices) {
                if (!codes[i].contentEquals(other.codes[i])) return false
            }
            return true
        }
        override fun hashCode(): Int {
            var h = (nGenerated * 31 + (eosFrame ?: -1))
            for (row in codes) h = h * 31 + row.contentHashCode()
            return h
        }
    }
}
