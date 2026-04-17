package com.afun.mosstts.core.model

import com.afun.mosstts.core.download.Sha256Verifier
import com.google.common.truth.Truth.assertThat
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ModelManagerTest {
    @get:Rule val tmp = TemporaryFolder()

    private val realManifestText by lazy {
        javaClass.getResource("/manifest_v2_sample.json")!!.readText()
    }
    private val realConfigText by lazy {
        javaClass.getResource("/config_sample.json")!!.readText()
    }

    /** Lays down everything `manifest.files` lists with random non-empty bytes. */
    private fun layCompleteBundle(): File {
        val dir = tmp.newFolder("bundle")
        dir.resolve("manifest.json").writeText(realManifestText)
        val manifest = Manifest.parse(realManifestText)
        manifest.files.forEach { entry ->
            val payload = when (entry.name) {
                "config.json" -> realConfigText.toByteArray()
                else -> ByteArray(8) { (it + 1).toByte() }
            }
            dir.resolve(entry.name).writeBytes(payload)
        }
        return dir
    }

    @Test
    fun `validate reports complete=true on a fully-laid bundle (no sha check)`() {
        val dir = layCompleteBundle()
        val report = ModelManager.validate(dir, expectedSha = emptyMap())
        assertThat(report.complete).isTrue()
        assertThat(report.missing).isEmpty()
        assertThat(report.shaMismatch).isEmpty()
    }

    @Test
    fun `validate reports missing files`() {
        val dir = layCompleteBundle()
        dir.resolve("config.json").delete()
        dir.resolve("tokenizer.model").delete()

        val report = ModelManager.validate(dir, expectedSha = emptyMap())
        assertThat(report.complete).isFalse()
        assertThat(report.missing).containsExactly("config.json", "tokenizer.model")
    }

    @Test
    fun `validate flags sha mismatch when expectedSha is provided`() {
        val dir = layCompleteBundle()
        val expected = mapOf(
            "tokenizer.model" to "0000000000000000000000000000000000000000000000000000000000000000",
        )

        val report = ModelManager.validate(dir, expectedSha = expected)
        assertThat(report.complete).isFalse()
        assertThat(report.shaMismatch).containsExactly("tokenizer.model")
        assertThat(report.missing).isEmpty()
    }

    @Test
    fun `validate is silent on entries not present in expectedSha`() {
        val dir = layCompleteBundle()
        // Only one entry expected; everything else should still be considered fine.
        val tokenizer = dir.resolve("tokenizer.model")
        val correctSha = tokenizer.inputStream().use { Sha256Verifier.streamHex(it) }

        val report = ModelManager.validate(
            dir,
            expectedSha = mapOf("tokenizer.model" to correctSha),
        )
        assertThat(report.complete).isTrue()
        assertThat(report.shaMismatch).isEmpty()
    }

    @Test
    fun `validate throws when manifest is missing`() {
        val dir = tmp.newFolder("empty")
        val ex = runCatching { ModelManager.validate(dir, expectedSha = emptyMap()) }
        assertThat(ex.isFailure).isTrue()
    }

    @Test
    fun `loadConfig returns the parsed ModelConfig`() {
        val dir = layCompleteBundle()
        val cfg = ModelManager.loadConfig(dir)
        assertThat(cfg.nVq).isEqualTo(16)
        assertThat(cfg.audioTokenizerSampleRate).isEqualTo(48000)
    }
}
