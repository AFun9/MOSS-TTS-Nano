package com.afun.mosstts.core.infer

import com.afun.mosstts.core.model.Manifest
import com.afun.mosstts.core.model.ModelConfig
import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * Structural correctness for the v1.0.0 voice_clone prompt layout.
 *
 * The new builder no longer encodes template strings via the tokenizer
 * (the manifest ships pre-tokenized prefix/suffix arrays), so there is
 * no Python-side fixture file to compare against here. Instead we lock
 * down the **layout invariants**:
 *
 *   1. Total row count is the sum of: prefix + 1 + F + 1 + after_ref +
 *      text + asst_prefix + 1.
 *   2. Each row has stride n_vq+1 = 17 for MOSS-TTS-Nano.
 *   3. Text rows have audio_pad in columns 1..n_vq.
 *   4. Reference-audio rows have audio_user_slot in column 0 and the
 *      verbatim codes in columns 1..n_vq.
 *   5. The single `audio_start` row right before the audio block holds
 *      `audio_start_token_id` in column 0; the single `audio_end` row
 *      right after holds `audio_end_token_id`; the final row that kicks
 *      off generation holds `audio_start_token_id` again.
 *
 * Once we have a Kotlin-side dump tool we'll add a byte-equal fixture
 * test against `demo_generate.py:build_input_ids`.
 */
class PromptBuilderTest {
    private val manifest by lazy {
        Manifest.parse(javaClass.getResource("/release_manifest_v1.json")!!.readText())
    }
    private val cfg by lazy { ModelConfig.fromManifest(manifest) }
    private val pb by lazy { PromptBuilder(cfg, manifest.promptTemplates) }

    private fun fakeCodes(frames: Int, seed: Int = 0): AudioCodes {
        val data = LongArray(frames * cfg.nVq)
        // Deterministic, codebook-bounded fake codes.
        for (i in data.indices) data[i] = ((i * 31 + seed) % cfg.audioCodebookSize).toLong()
        return AudioCodes(frames = frames, nVq = cfg.nVq, data = data)
    }

    @Test
    fun `voice_clone prompt total rows equals expected sum`() {
        val frames = 5
        val codes = fakeCodes(frames)
        val textIds = intArrayOf(100, 200, 300, 400)
        val prompt = pb.buildVoiceClonePrompt(textIds, codes)

        val pt = manifest.promptTemplates
        val expected = pt.userPromptPrefixTokenIds.size + 1 + frames + 1 +
            pt.userPromptAfterReferenceTokenIds.size + textIds.size +
            pt.assistantPromptPrefixTokenIds.size + 1

        assertThat(prompt.seqLen).isEqualTo(expected)
        assertThat(prompt.rowStride).isEqualTo(cfg.rowStride)
        assertThat(prompt.inputIds.size).isEqualTo(expected * cfg.rowStride)
        assertThat(prompt.attentionMask.size).isEqualTo(expected)
        assertThat(prompt.attentionMask.toList()).containsExactlyElementsIn(List(expected) { 1L })
    }

    @Test
    fun `text rows fill columns 1 to n_vq with audio_pad`() {
        val codes = fakeCodes(3)
        val textIds = intArrayOf(42)
        val prompt = pb.buildVoiceClonePrompt(textIds, codes)

        // Row 0 is the first prefix token; check its layout.
        val pad = cfg.audioPadTokenId.toLong()
        for (k in 1..cfg.nVq) {
            assertThat(prompt.inputIds[k]).isEqualTo(pad)
        }
    }

    @Test
    fun `audio rows hold user_slot in col 0 and verbatim codes in cols 1+`() {
        val frames = 4
        val codes = fakeCodes(frames, seed = 7)
        val prompt = pb.buildVoiceClonePrompt(intArrayOf(1, 2), codes)

        val pt = manifest.promptTemplates
        val audioStartRow = pt.userPromptPrefixTokenIds.size  // single row
        val audioBaseRow = audioStartRow + 1                  // start of audio block

        for (i in 0 until frames) {
            val rowBase = (audioBaseRow + i) * cfg.rowStride
            assertThat(prompt.inputIds[rowBase])
                .isEqualTo(cfg.audioUserSlotTokenId.toLong())
            for (k in 0 until cfg.nVq) {
                assertThat(prompt.inputIds[rowBase + 1 + k])
                    .isEqualTo(codes.data[i * cfg.nVq + k])
            }
        }
    }

    @Test
    fun `audio_start sandwiches the audio block, audio_end follows, final audio_start kicks off generation`() {
        val frames = 2
        val codes = fakeCodes(frames)
        val prompt = pb.buildVoiceClonePrompt(intArrayOf(7, 8, 9), codes)

        val pt = manifest.promptTemplates
        val rowAudioStart = pt.userPromptPrefixTokenIds.size
        val rowAudioEnd = rowAudioStart + 1 + frames
        val rowFinalAudioStart = prompt.seqLen - 1

        val rs = cfg.rowStride
        assertThat(prompt.inputIds[rowAudioStart * rs])
            .isEqualTo(cfg.audioStartTokenId.toLong())
        assertThat(prompt.inputIds[rowAudioEnd * rs])
            .isEqualTo(cfg.audioEndTokenId.toLong())
        assertThat(prompt.inputIds[rowFinalAudioStart * rs])
            .isEqualTo(cfg.audioStartTokenId.toLong())
    }

    @Test
    fun `mismatched n_vq throws`() {
        val bogus = AudioCodes(frames = 2, nVq = 8, data = LongArray(16) { 0L })
        val ex = runCatching { pb.buildVoiceClonePrompt(intArrayOf(1), bogus) }
        assertThat(ex.isFailure).isTrue()
    }
}
