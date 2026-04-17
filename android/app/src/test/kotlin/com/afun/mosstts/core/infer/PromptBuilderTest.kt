package com.afun.mosstts.core.infer

import com.afun.mosstts.core.tokenizer.Tokenizer
import com.google.common.truth.Truth.assertThat
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.junit.Test
import java.security.MessageDigest

/**
 * End-to-end byte-equality of `PromptBuilder` against
 * `onnx_infer.py:PromptBuilder` over 17 fixtures captured by
 * `tools/dump_prompt_fixtures.py`.
 *
 * Failing fingerprints (shape / length / first 16 / last 16 / sum /
 * sha256) catch every plausible Kotlin↔Python divergence:
 *   - tokenizer drift (already covered by the 109-fixture tokenizer
 *     test, but PromptBuilder is the next-most-likely place to break),
 *   - special-token IDs read from the wrong meta key,
 *   - row stride / `audio_pad_token_id` filling wrong cells,
 *   - audio slot-token wiring (user vs assistant slot),
 *   - section concatenation order in `build_voice_clone_prompt`.
 */
class PromptBuilderTest {
    private val payload by lazy {
        val raw = javaClass.getResource("/prompt_fixtures.json")
            ?: error("prompt_fixtures.json not found on test classpath")
        Json { ignoreUnknownKeys = true }
            .decodeFromString(FixtureFile.serializer(), raw.readText())
    }

    private val tokenizer by lazy {
        val raw = javaClass.getResource("/tokenizer_kotlin.json")!!.readText()
        Tokenizer.load(raw)
    }

    private val pb by lazy {
        PromptBuilder(
            tokenizer = tokenizer,
            nVq = payload.meta.nVq,
            audioPadTokenId = payload.meta.audioPadTokenId,
            imStartTokenId = payload.meta.imStartTokenId,
            imEndTokenId = payload.meta.imEndTokenId,
            audioStartTokenId = payload.meta.audioStartTokenId,
            audioEndTokenId = payload.meta.audioEndTokenId,
            audioUserSlotTokenId = payload.meta.audioUserSlotTokenId,
            audioAssistantSlotTokenId = payload.meta.audioAssistantSlotTokenId,
        )
    }

    @Test
    fun `fixture file is the one currently shipped by the python exporter`() {
        assertThat(payload.fixtures).hasSize(17)
        assertThat(payload.meta.nVq).isEqualTo(16)
        assertThat(payload.meta.audioPadTokenId).isEqualTo(1024)
    }

    @Test
    fun `every prompt fixture matches Python PromptBuilder byte-for-byte`() {
        val mismatches = mutableListOf<String>()
        for ((idx, fx) in payload.fixtures.withIndex()) {
            val ids = tokenizer.encode(fx.text).map { it }.toIntArray()
            assertThat(ids.size).isEqualTo(fx.textIdsLen)  // tokenizer-side sanity

            val codes = fx.audioPrompt?.let {
                AudioCodes(frames = it.frames, nVq = it.nVq, data = it.codes.map { v -> v.toLong() }.toLongArray())
            }
            val prompt = when (fx.mode) {
                "continuation" -> pb.buildContinuationPrompt(ids, codes)
                "voice_clone" -> pb.buildVoiceClonePrompt(ids, codes ?: error("voice_clone needs codes"))
                else -> error("unknown mode ${fx.mode}")
            }

            val actualFp = fingerprint(prompt.inputIds)
            val expectedFp = fx.inputIds
            if (actualFp != expectedFp) {
                mismatches += buildString {
                    appendLine("[#$idx] mode=${fx.mode} text=${fx.text.take(40)!!}")
                    appendLine("    expected shape=${expectedFp.shape} length=${expectedFp.length} sum=${expectedFp.sum}")
                    appendLine("    actual   shape=${actualFp.shape}   length=${actualFp.length}   sum=${actualFp.sum}")
                    appendLine("    expected first16=${expectedFp.first16}")
                    appendLine("    actual   first16=${actualFp.first16}")
                    appendLine("    expected last16=${expectedFp.last16}")
                    appendLine("    actual   last16=${actualFp.last16}")
                    appendLine("    expected sha=${expectedFp.sha256}")
                    appendLine("    actual   sha=${actualFp.sha256}")
                }
                if (mismatches.size >= 17) break
            }
        }
        if (mismatches.isNotEmpty()) {
            error(
                "PromptBuilder diverged from Python on ${mismatches.size}/${payload.fixtures.size} " +
                    "fixtures (showing up to 3):\n\n" + mismatches.joinToString("\n"),
            )
        }
    }

    // ---- helpers ----------------------------------------------------

    private fun fingerprint(longs: LongArray): InputIdsFingerprint {
        val md = MessageDigest.getInstance("SHA-256")
        val buf = java.nio.ByteBuffer.allocate(longs.size * 8).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (v in longs) buf.putLong(v)
        md.update(buf.array())
        val sha = md.digest().joinToString("") { "%02x".format(it) }
        val first16 = LongArray(minOf(16, longs.size)) { longs[it] }.toList().map { it.toInt() }
        val last16 = LongArray(minOf(16, longs.size)) { longs[longs.size - minOf(16, longs.size) + it] }
            .toList().map { it.toInt() }
        var sum = 0L
        for (v in longs) sum += v
        // Recovered shape: PromptBuilder always emits [1, T, nVq+1] flattened.
        val rowStride = payload.meta.nVq + 1
        val seqLen = longs.size / rowStride
        return InputIdsFingerprint(
            shape = listOf(1, seqLen, rowStride),
            dtype = "int64",
            length = longs.size,
            first16 = first16,
            last16 = last16,
            sum = sum,
            sha256 = sha,
        )
    }

    @Serializable
    data class FixtureFile(
        val meta: Meta,
        val fixtures: List<Fixture>,
    )

    @Serializable
    data class Meta(
        @SerialName("n_vq") val nVq: Int,
        @SerialName("im_start_token_id") val imStartTokenId: Int,
        @SerialName("im_end_token_id") val imEndTokenId: Int,
        @SerialName("audio_start_token_id") val audioStartTokenId: Int,
        @SerialName("audio_end_token_id") val audioEndTokenId: Int,
        @SerialName("audio_pad_token_id") val audioPadTokenId: Int,
        @SerialName("audio_user_slot_token_id") val audioUserSlotTokenId: Int,
        @SerialName("audio_assistant_slot_token_id") val audioAssistantSlotTokenId: Int,
    )

    @Serializable
    data class Fixture(
        val mode: String,
        val text: String,
        @SerialName("text_ids_len") val textIdsLen: Int,
        @SerialName("has_audio_prompt") val hasAudioPrompt: Boolean,
        @SerialName("audio_prompt") val audioPrompt: AudioPromptDto? = null,
        @SerialName("input_ids") val inputIds: InputIdsFingerprint,
    )

    @Serializable
    data class AudioPromptDto(
        val frames: Int,
        val seed: Int,
        @SerialName("n_vq") val nVq: Int,
        val sha256: String,
        val codes: List<Int>,
    )

    @Serializable
    data class InputIdsFingerprint(
        val shape: List<Int>,
        val dtype: String,
        val length: Int,
        @SerialName("first_16") val first16: List<Int>,
        @SerialName("last_16") val last16: List<Int>,
        val sum: Long,
        val sha256: String,
    )
}
