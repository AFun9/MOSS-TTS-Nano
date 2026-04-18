package com.afun.mosstts.core.infer

import com.afun.mosstts.core.model.Manifest
import com.afun.mosstts.core.model.ModelConfig

/**
 * Kotlin port of `export_v2/scripts/demo_generate.py:build_input_ids`.
 *
 * Produces the `[1, T, n_vq+1]` int64 prompt tensor used by `prefill`:
 *   - column 0 holds either a text token id (for "text rows") or one of
 *     the slot tokens (`audio_user_slot` / `audio_assistant_slot`) for
 *     "audio rows";
 *   - columns 1..n_vq hold `audio_pad_token_id` for text rows, or the
 *     16 codec codebook ids for audio rows.
 *
 * The official voice_clone layout (the only generation mode the v1.0.0
 * release exposes) is, top-to-bottom:
 *
 * ```
 * user_prompt_prefix_token_ids                      [text rows]
 * <audio_start>                                     [single text row]
 * prompt_audio_codes (F rows, slot=audio_user_slot) [F audio rows]
 * <audio_end>                                       [single text row]
 * user_prompt_after_reference_token_ids             [text rows]
 * text_token_ids (target text)                      [text rows]
 * assistant_prompt_prefix_token_ids                 [text rows]
 * <audio_start>                                     [single text row, kicks off generation]
 * ```
 *
 * **Why we no longer encode template strings via the tokenizer**: the
 * v1.0.0 release ships the three boilerplate sections as pre-tokenized
 * id arrays inside `manifest.prompt_templates`. Driving them through
 * `Tokenizer.encode(...)` had been a recurring source of off-by-one drift
 * (the user/role/template strings include `<|im_start|>` and ZWJ-adjacent
 * unicode that the SP normalizer is touchy about). Since the manifest is
 * the single source of truth for the on-device protocol, we now consume
 * those id arrays verbatim — guaranteed byte-equal to Python.
 *
 * The user-supplied target text is still tokenized at runtime; that's
 * locked down by the 64-fixture `TokenizerGoldenTest`.
 */
class PromptBuilder(
    private val cfg: ModelConfig,
    private val templates: Manifest.PromptTemplates,
) {
    /** `n_vq + 1` — total columns per prompt row. */
    val rowStride: Int = cfg.rowStride

    /**
     * Build the full input_ids for the official voice_clone generation
     * mode. [textIds] are the tokenizer-encoded user text; [promptCodes]
     * are the codec codes for the reference audio (either from a builtin
     * voice or from a runtime-cloned wav via [com.afun.mosstts.core.infer.CodecRunner]).
     */
    fun buildVoiceClonePrompt(textIds: IntArray, promptCodes: AudioCodes): Prompt {
        require(promptCodes.nVq == cfg.nVq) {
            "audio prompt nVq=${promptCodes.nVq} != cfg.nVq=${cfg.nVq}"
        }
        val sections = listOf(
            Section.Text(templates.userPromptPrefixTokenIds.toIntArray()),
            Section.Text(intArrayOf(cfg.audioStartTokenId)),
            Section.Audio(promptCodes, slotTokenId = cfg.audioUserSlotTokenId),
            Section.Text(intArrayOf(cfg.audioEndTokenId)),
            Section.Text(templates.userPromptAfterReferenceTokenIds.toIntArray()),
            Section.Text(textIds),
            Section.Text(templates.assistantPromptPrefixTokenIds.toIntArray()),
            Section.Text(intArrayOf(cfg.audioStartTokenId)),
        )
        return assemble(sections)
    }

    // ---- internals -------------------------------------------------

    private sealed class Section {
        abstract val rows: Int
        class Text(val ids: IntArray) : Section() { override val rows: Int = ids.size }
        class Audio(val codes: AudioCodes, val slotTokenId: Int) : Section() {
            override val rows: Int = codes.frames
        }
    }

    private fun assemble(sections: List<Section>): Prompt {
        val totalRows = sections.sumOf { it.rows }
        val pad = cfg.audioPadTokenId.toLong()
        val flat = LongArray(totalRows * rowStride) { pad }
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
        return Prompt(
            inputIds = flat,
            attentionMask = attention,
            seqLen = totalRows,
            rowStride = rowStride,
        )
    }
}

/** Codec codes for a reference clip (16 codebooks, F frames). */
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

/** Output of `PromptBuilder`. `inputIds` is row-major `[1, seqLen, rowStride]` int64. */
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
