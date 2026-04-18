package com.afun.mosstts.core.model

import com.afun.mosstts.core.download.Sha256Verifier
import com.google.common.truth.Truth.assertThat
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.security.MessageDigest

/**
 * v1.0.0 release: the manifest is **self-validating** (every entry in
 * `manifest.files` carries a SHA-256). `ModelManager` checks the on-disk
 * SHA against the manifest's, and optionally against an additional
 * trusted-source map (e.g. GitHub release notes) to defend against a
 * tampered manifest.
 */
class ModelManagerTest {
    @get:Rule val tmp = TemporaryFolder()

    private val manifestText by lazy {
        javaClass.getResource("/release_manifest_v1.json")!!.readText()
    }

    /**
     * Lay down a fake bundle with files whose contents intentionally do
     * NOT match the manifest's SHA-256 (we use random bytes), so the
     * test can assert the SHA-mismatch detection path. SHA verification
     * is opt-in via `verifySha`; tests that don't care can pass
     * `verifySha = false`.
     */
    private fun layFakeBundle(): File {
        val dir = tmp.newFolder("bundle")
        dir.resolve("manifest.json").writeText(manifestText)
        val m = Manifest.parse(manifestText)
        for (name in m.files.keys) {
            dir.resolve(name).writeBytes(ByteArray(8) { (it + 1).toByte() })
        }
        return dir
    }

    @Test
    fun `validate without sha check accepts a complete bundle`() {
        val dir = layFakeBundle()
        val report = ModelManager.validate(dir, verifySha = false)
        assertThat(report.complete).isTrue()
        assertThat(report.missing).isEmpty()
        assertThat(report.shaMismatch).isEmpty()
    }

    @Test
    fun `validate reports missing files`() {
        val dir = layFakeBundle()
        dir.resolve("tokenizer.model").delete()
        dir.resolve("moss_tts_prefill.onnx").delete()

        val report = ModelManager.validate(dir, verifySha = false)
        assertThat(report.complete).isFalse()
        assertThat(report.missing).containsAtLeast("tokenizer.model", "moss_tts_prefill.onnx")
    }

    @Test
    fun `validate detects manifest-internal sha mismatch when verifySha is on`() {
        val dir = layFakeBundle()
        // Files have random bytes; manifest's SHA-256 entries point at the
        // real release contents -> every file should fail SHA verification.
        val report = ModelManager.validate(dir, verifySha = true)
        assertThat(report.complete).isFalse()
        assertThat(report.shaMismatch).isNotEmpty()
        // tokenizer.model is in the manifest with a real sha; its random
        // payload here cannot match.
        assertThat(report.shaMismatch).contains("tokenizer.model")
    }

    @Test
    fun `validate accepts files whose payload matches both manifest and external sha`() {
        val dir = tmp.newFolder("bundle")
        dir.resolve("manifest.json").writeText(buildSyntheticManifest("hello\n", "world\n"))
        dir.resolve("a.bin").writeText("hello\n")
        dir.resolve("b.bin").writeText("world\n")

        val external = mapOf("a.bin" to sha256("hello\n"))
        val report = ModelManager.validate(dir, expectedSha = external, verifySha = true)
        assertThat(report.complete).isTrue()
        assertThat(report.ok).containsAtLeast("a.bin", "b.bin")
    }

    @Test
    fun `validate flags external sha mismatch even if manifest sha is fine`() {
        val dir = tmp.newFolder("bundle")
        dir.resolve("manifest.json").writeText(buildSyntheticManifest("hello\n", "world\n"))
        dir.resolve("a.bin").writeText("hello\n")
        dir.resolve("b.bin").writeText("world\n")

        val wrongExternal = mapOf(
            "b.bin" to "0".repeat(64),
        )
        val report = ModelManager.validate(dir, expectedSha = wrongExternal, verifySha = true)
        assertThat(report.complete).isFalse()
        assertThat(report.shaMismatch).contains("b.bin")
    }

    @Test
    fun `validate throws when manifest is missing`() {
        val dir = tmp.newFolder("empty")
        val ex = runCatching { ModelManager.validate(dir, verifySha = false) }
        assertThat(ex.isFailure).isTrue()
    }

    @Test
    fun `loadConfig produces a runtime ModelConfig`() {
        val dir = layFakeBundle()
        val cfg = ModelManager.loadConfig(dir)
        assertThat(cfg.nVq).isEqualTo(16)
        assertThat(cfg.audioStartTokenId).isEqualTo(6)
    }

    @Test
    fun `loadManifest round-trips the schema fields`() {
        val dir = layFakeBundle()
        val m = ModelManager.loadManifest(dir)
        assertThat(m.formatVersion).isEqualTo(1)
        assertThat(m.tts.graphs).hasSize(4)
    }

    // ---- helpers ----

    private fun sha256(s: String): String =
        MessageDigest.getInstance("SHA-256").digest(s.toByteArray()).joinToString("") { "%02x".format(it) }

    /** Tiny well-formed manifest with two synthetic files for SHA tests. */
    private fun buildSyntheticManifest(aContent: String, bContent: String): String {
        val aSha = sha256(aContent)
        val bSha = sha256(bContent)
        return """
        {
          "format_version": 1,
          "release_version": "0.0.0-test",
          "tts": {
            "graphs": [],
            "shared_blob": "missing.data",
            "external_data_files": {},
            "io": {}
          },
          "codec": {
            "graphs": [],
            "shared_blob": "missing.data",
            "external_data_files": {},
            "io": {}
          },
          "tts_config": {
            "n_vq": 16,
            "audio_pad_token_id": 1024,
            "pad_token_id": 3,
            "im_start_token_id": 4,
            "im_end_token_id": 5,
            "audio_start_token_id": 6,
            "audio_end_token_id": 7,
            "audio_user_slot_token_id": 8,
            "audio_assistant_slot_token_id": 9,
            "audio_codebook_sizes": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                      1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
            "vocab_size": 16384
          },
          "prompt_templates": {
            "user_prompt_prefix_token_ids": [1, 2, 3],
            "user_prompt_after_reference_token_ids": [4, 5],
            "assistant_prompt_prefix_token_ids": [6, 7]
          },
          "generation_defaults": {
            "audio_temperature": 0.8,
            "audio_top_p": 0.95,
            "audio_top_k": 25
          },
          "tokenizer": { "type": "sentencepiece", "file": "tokenizer.model" },
          "files": {
            "a.bin": { "size": ${aContent.toByteArray().size}, "sha256": "$aSha" },
            "b.bin": { "size": ${bContent.toByteArray().size}, "sha256": "$bSha" }
          }
        }
        """.trimIndent()
    }
}
