package com.afun.mosstts.core.infer

import com.afun.mosstts.core.tokenizer.Tokenizer

/**
 * Kotlin port of `onnx_infer.py:PromptBuilder`.
 *
 * Produces the `[1, T, n_vq+1]` int64 prompt tensor (text token in
 * column 0, audio_pad in columns 1..n_vq for text rows; slot-token in
 * column 0, codec codes in columns 1..n_vq for audio rows). Two prompt
 * shapes are supported:
 *
 *   - `buildContinuationPrompt(textIds, codes? = null)`:
 *         user_prefix + "None" + after_ref + textIds + assistant_prefix +
 *         <audio_start> [+ codes-as-assistant-slot if provided]
 *
 *   - `buildVoiceClonePrompt(textIds, codes)`:
 *         user_prefix + <audio_start> + codes-as-user-slot + <audio_end> +
 *         after_ref + textIds + assistant_prefix + <audio_start>
 *
 * Static prefix tokens are computed once in the constructor (matches
 * Python `__init__`), so per-call work is O(textIds + codeFrames).
 *
 * Byte-equality with the reference is enforced by `PromptBuilderTest`
 * across 17 multi-language fixtures including the empty-text and
 * special-token-literal edge cases.
 */
class PromptBuilder(
    private val tokenizer: Tokenizer,
    val nVq: Int,
    private val audioPadTokenId: Int,
    private val imStartTokenId: Int,
    private val imEndTokenId: Int,
    private val audioStartTokenId: Int,
    private val audioEndTokenId: Int,
    private val audioUserSlotTokenId: Int,
    private val audioAssistantSlotTokenId: Int,
) {
    /** `n_vq + 1` — total columns per prompt row (text/slot col + n_vq code cols). */
    val rowStride: Int = nVq + 1

    private val userPrefix: IntArray
    private val afterRef: IntArray
    private val assistantPrefix: IntArray
    private val noneIds: IntArray

    init {
        val imStart = imStartTokenId
        val imEnd = imEndTokenId
        userPrefix = concat(
            intArrayOf(imStart),
            encode(USER_ROLE_PREFIX),
            encode(USER_TEMPLATE_REFERENCE_PREFIX),
        )
        afterRef = encode(USER_TEMPLATE_AFTER_REFERENCE)
        assistantPrefix = concat(
            encode(USER_TEMPLATE_SUFFIX),
            intArrayOf(imEnd),
            encode(ASSISTANT_TURN_PREFIX),
            intArrayOf(imStart),
            encode(ASSISTANT_ROLE_PREFIX),
        )
        noneIds = encode("None")
    }

    fun buildContinuationPrompt(textIds: IntArray, promptAudioCodes: AudioCodes? = null): Prompt {
        val promptIds = concat(userPrefix, noneIds, afterRef, textIds, assistantPrefix)
        val sections = ArrayList<Section>(3).apply {
            add(Section.Text(promptIds))
            add(Section.Text(intArrayOf(audioStartTokenId)))
            if (promptAudioCodes != null) {
                add(Section.Audio(promptAudioCodes, slotTokenId = audioAssistantSlotTokenId))
            }
        }
        return assemble(sections)
    }

    fun buildVoiceClonePrompt(textIds: IntArray, promptAudioCodes: AudioCodes): Prompt {
        require(promptAudioCodes.nVq == nVq) { "audio prompt nVq=${promptAudioCodes.nVq} != $nVq" }
        val prefixIds = concat(userPrefix, intArrayOf(audioStartTokenId))
        val suffixIds = concat(
            intArrayOf(audioEndTokenId),
            afterRef,
            textIds,
            assistantPrefix,
            intArrayOf(audioStartTokenId),
        )
        return assemble(
            listOf(
                Section.Text(prefixIds),
                Section.Audio(promptAudioCodes, slotTokenId = audioUserSlotTokenId),
                Section.Text(suffixIds),
            ),
        )
    }

    // ---- internals -------------------------------------------------

    private fun encode(text: String): IntArray = tokenizer.encode(text)

    private sealed class Section {
        abstract val rows: Int
        class Text(val ids: IntArray) : Section() { override val rows: Int = ids.size }
        class Audio(val codes: AudioCodes, val slotTokenId: Int) : Section() {
            override val rows: Int = codes.frames
        }
    }

    private fun assemble(sections: List<Section>): Prompt {
        val totalRows = sections.sumOf { it.rows }
        val flat = LongArray(totalRows * rowStride) { audioPadTokenId.toLong() }
        var rowOffset = 0
        for (sec in sections) {
            when (sec) {
                is Section.Text -> for (i in sec.ids.indices) {
                    flat[(rowOffset + i) * rowStride] = sec.ids[i].toLong()
                }
                is Section.Audio -> {
                    val slot = sec.slotTokenId.toLong()
                    val codes = sec.codes.data
                    for (i in 0 until sec.codes.frames) {
                        val rowBase = (rowOffset + i) * rowStride
                        flat[rowBase] = slot
                        val codeBase = i * sec.codes.nVq
                        for (k in 0 until sec.codes.nVq) {
                            flat[rowBase + 1 + k] = codes[codeBase + k]
                        }
                    }
                }
            }
            rowOffset += sec.rows
        }
        val attention = LongArray(totalRows) { 1L }
        return Prompt(inputIds = flat, attentionMask = attention, seqLen = totalRows, rowStride = rowStride)
    }

    private fun concat(vararg parts: IntArray): IntArray {
        var n = 0
        for (p in parts) n += p.size
        val out = IntArray(n)
        var off = 0
        for (p in parts) {
            System.arraycopy(p, 0, out, off, p.size)
            off += p.size
        }
        return out
    }

    companion object {
        // Must mirror onnx_infer.py:USER_TEMPLATE_* exactly. Any whitespace
        // change here breaks byte-equality with the reference.
        private const val USER_ROLE_PREFIX = "user\n"
        private const val USER_TEMPLATE_REFERENCE_PREFIX =
            "<user_inst>\n- Reference(s):\n"
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
    }
}

/** Reference codec codes for voice-clone or continuation prompts. */
data class AudioCodes(
    val frames: Int,
    val nVq: Int,
    /** Row-major flattened, length = frames * nVq. */
    val data: LongArray,
) {
    init {
        require(data.size == frames * nVq) {
            "data.size=${data.size} != frames*nVq=${frames * nVq}"
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is AudioCodes) return false
        return frames == other.frames && nVq == other.nVq && data.contentEquals(other.data)
    }
    override fun hashCode(): Int =
        (frames * 31 + nVq) * 31 + data.contentHashCode()
}

/** Output of `PromptBuilder`. `inputIds` is row-major `[1, seqLen, rowStride]`. */
data class Prompt(
    val inputIds: LongArray,
    val attentionMask: LongArray,
    val seqLen: Int,
    val rowStride: Int,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Prompt) return false
        return seqLen == other.seqLen && rowStride == other.rowStride &&
            inputIds.contentEquals(other.inputIds) &&
            attentionMask.contentEquals(other.attentionMask)
    }
    override fun hashCode(): Int =
        ((seqLen * 31 + rowStride) * 31 + inputIds.contentHashCode()) * 31 +
            attentionMask.contentHashCode()
}
