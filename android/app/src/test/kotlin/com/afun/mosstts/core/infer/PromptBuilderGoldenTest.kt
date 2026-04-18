package com.afun.mosstts.core.infer

import com.afun.mosstts.core.model.Manifest
import com.afun.mosstts.core.model.ModelConfig
import com.google.common.truth.Truth.assertThat
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * Byte-equal golden test against `tools/dump_prompt_fixtures_v2.py`,
 * which calls the canonical `demo_generate.py:build_input_ids` from the
 * v1.0.0 release.
 *
 * Each fixture carries the full int64 input_ids tensor; we compare
 *   1. the per-element sequence (catches any single-cell drift), and
 *   2. the SHA-256 fingerprint of the LE int64 byte buffer (last-line-of
 *      defence + makes failure messages compact when the array is huge).
 *
 * Drift detection contract: if `make_release.py` ever changes the
 * `prompt_templates` block in the manifest, this test starts failing
 * **immediately** (the manifest is the same input on both sides; a
 * Python <-> Kotlin divergence here means the Kotlin assembly logic
 * itself drifted).
 *
 * Re-generate fixtures with:
 *     python tools/dump_prompt_fixtures_v2.py
 */
class PromptBuilderGoldenTest {
    private val manifest by lazy {
        Manifest.parse(javaClass.getResource("/release_manifest_v1.json")!!.readText())
    }
    private val cfg by lazy { ModelConfig.fromManifest(manifest) }
    private val pb by lazy { PromptBuilder(cfg, manifest.promptTemplates) }

    private val payload by lazy {
        val raw = javaClass.getResource("/prompt_fixtures_v2.json")
            ?: error("prompt_fixtures_v2.json missing - run tools/dump_prompt_fixtures_v2.py")
        Json { ignoreUnknownKeys = true }.decodeFromString(FixtureFile.serializer(), raw.readText())
    }

    @Test
    fun `fixture file matches the on-device manifest version and codebook`() {
        assertThat(payload.nVq).isEqualTo(cfg.nVq)
        assertThat(payload.audioCodebookSize).isEqualTo(cfg.audioCodebookSize)
        // Optional: same release_version - guards against accidentally shipping
        // mis-paired manifest+fixtures during a release rev.
        assertThat(payload.releaseVersion).isEqualTo(manifest.releaseVersion)
    }

    @Test
    fun `every fixture matches Python build_input_ids byte-for-byte`() {
        val mismatches = mutableListOf<String>()

        for (fx in payload.fixtures) {
            val codes = AudioCodes(
                frames = fx.input.audioPrompt.frames,
                nVq = fx.input.audioPrompt.nVq,
                data = fx.input.audioPrompt.codes.map { it.toLong() }.toLongArray(),
            )
            val textIds = fx.input.textTokenIds.toIntArray()

            val prompt = pb.buildVoiceClonePrompt(textIds, codes)

            val expectedFlat = fx.inputIdsFull.map { it.toLong() }.toLongArray()
            val expectedFp = fx.fingerprint
            val actualSha = sha256OfLongs(prompt.inputIds)

            val seqOk = prompt.inputIds.contentEquals(expectedFlat)
            val shaOk = actualSha.equals(expectedFp.sha256, ignoreCase = true)

            if (!seqOk || !shaOk) {
                mismatches += buildString {
                    appendLine("[${fx.id}] PromptBuilder diverged from Python")
                    appendLine("    expected shape=${expectedFp.shape} len=${expectedFp.length} sha=${expectedFp.sha256}")
                    appendLine("    actual   len=${prompt.inputIds.size} sha=$actualSha")
                    if (prompt.seqLen * prompt.rowStride != expectedFp.length) {
                        appendLine("    seq_len*stride mismatch: ${prompt.seqLen}*${prompt.rowStride} vs ${expectedFp.length}")
                    }
                    val firstDiff = (0 until minOf(prompt.inputIds.size, expectedFlat.size))
                        .firstOrNull { prompt.inputIds[it] != expectedFlat[it] }
                    if (firstDiff != null) {
                        val rowIdx = firstDiff / prompt.rowStride
                        val colIdx = firstDiff % prompt.rowStride
                        appendLine("    first diff at flat=$firstDiff (row=$rowIdx col=$colIdx):")
                        appendLine("      expected=${expectedFlat[firstDiff]}  actual=${prompt.inputIds[firstDiff]}")
                    }
                }
            }
        }
        if (mismatches.isNotEmpty()) {
            error(
                "PromptBuilder diverged on ${mismatches.size}/${payload.fixtures.size} fixtures:\n\n" +
                    mismatches.joinToString("\n"),
            )
        }
    }

    @Test
    fun `attention mask is all ones with length matching seq_len`() {
        for (fx in payload.fixtures) {
            val codes = AudioCodes(
                frames = fx.input.audioPrompt.frames,
                nVq = fx.input.audioPrompt.nVq,
                data = fx.input.audioPrompt.codes.map { it.toLong() }.toLongArray(),
            )
            val textIds = fx.input.textTokenIds.toIntArray()
            val prompt = pb.buildVoiceClonePrompt(textIds, codes)

            assertThat(prompt.attentionMask.size).isEqualTo(prompt.seqLen)
            assertThat(prompt.attentionMask.all { it == 1L }).isTrue()
        }
    }

    private fun sha256OfLongs(longs: LongArray): String {
        val md = MessageDigest.getInstance("SHA-256")
        val bb = ByteBuffer.allocate(longs.size * java.lang.Long.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        for (v in longs) bb.putLong(v)
        md.update(bb.array())
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    @Serializable
    data class FixtureFile(
        val schema: String,
        @SerialName("release_version") val releaseVersion: String,
        @SerialName("n_vq") val nVq: Int,
        @SerialName("audio_codebook_size") val audioCodebookSize: Int,
        val fixtures: List<Fixture>,
    )

    @Serializable
    data class Fixture(
        val id: String,
        val input: FixtureInput,
        @SerialName("input_ids_full") val inputIdsFull: List<Long>,
        val fingerprint: Fingerprint,
    )

    @Serializable
    data class FixtureInput(
        @SerialName("text_token_ids") val textTokenIds: List<Int>,
        @SerialName("audio_prompt") val audioPrompt: AudioPromptDto,
    )

    @Serializable
    data class AudioPromptDto(
        val frames: Int,
        @SerialName("n_vq") val nVq: Int,
        val codes: List<Int>,
    )

    @Serializable
    data class Fingerprint(
        val shape: List<Int>,
        val length: Int,
        @SerialName("first_16") val first16: List<Int>,
        @SerialName("last_16") val last16: List<Int>,
        val sum: Long,
        val sha256: String,
    )
}
