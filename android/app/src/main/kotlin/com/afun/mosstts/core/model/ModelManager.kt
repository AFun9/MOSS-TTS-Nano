package com.afun.mosstts.core.model

import com.afun.mosstts.core.download.Sha256Verifier
import java.io.File

/**
 * Validates an on-disk ONNX bundle and parses its `config.json`.
 *
 * Stays free of `android.*` and ORT imports so it can be unit-tested on the
 * JVM. The actual ORT session loading lives in `core.engine` (M2).
 *
 * Usage:
 * ```
 * val report = ModelManager.validate(bundleDir, expectedSha)
 * if (report.complete) {
 *     val cfg = ModelManager.loadConfig(bundleDir)
 *     // hand off to InferenceEngine
 * }
 * ```
 */
object ModelManager {
    const val MANIFEST_NAME = "manifest.json"
    const val CONFIG_NAME = "config.json"

    /** Outcome of [validate]; [complete] is true iff `missing` and `shaMismatch` are both empty. */
    data class ValidationReport(
        val missing: List<String>,
        val shaMismatch: List<String>,
    ) {
        val complete: Boolean get() = missing.isEmpty() && shaMismatch.isEmpty()
    }

    /**
     * Confirms every file declared in `manifest.json` exists in [bundleDir]
     * and (when [expectedSha] supplies an entry for it) its SHA-256 matches.
     *
     * @throws IllegalArgumentException if [bundleDir] has no manifest.
     */
    fun validate(bundleDir: File, expectedSha: Map<String, String>): ValidationReport {
        val manifestFile = bundleDir.resolve(MANIFEST_NAME)
        require(manifestFile.isFile) { "manifest.json not found in ${bundleDir.absolutePath}" }
        val manifest = Manifest.parse(manifestFile.readText())

        val missing = mutableListOf<String>()
        val shaMismatch = mutableListOf<String>()

        for (entry in manifest.files) {
            val f = bundleDir.resolve(entry.name)
            if (!f.isFile) {
                missing += entry.name
                continue
            }
            val expected = expectedSha[entry.name] ?: continue
            if (!Sha256Verifier.verifyFile(f, expected)) {
                shaMismatch += entry.name
            }
        }
        return ValidationReport(missing = missing, shaMismatch = shaMismatch)
    }

    /** Reads and parses `config.json` from [bundleDir]. */
    fun loadConfig(bundleDir: File): ModelConfig {
        val cfgFile = bundleDir.resolve(CONFIG_NAME)
        require(cfgFile.isFile) { "config.json not found in ${bundleDir.absolutePath}" }
        return ModelConfig.parse(cfgFile.readText())
    }
}
