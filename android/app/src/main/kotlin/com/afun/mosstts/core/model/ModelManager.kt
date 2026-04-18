package com.afun.mosstts.core.model

import com.afun.mosstts.core.download.Sha256Verifier
import java.io.File

/**
 * Validates an on-disk release bundle and parses its `manifest.json`.
 *
 * Stays free of `android.*` and ORT imports so it can be unit-tested on the
 * JVM. The actual ORT session loading lives in `core.engine` (M2.x).
 *
 * The bundle is **self-validating**: every shipped file appears in
 * `manifest.files` with both a byte size and a SHA-256, so the runtime can
 * verify the bundle without an external sha256 sidecar. Callers may still
 * pass a separately-trusted `expectedSha` map (e.g. from the GitHub release
 * notes) to defend against a tampered manifest.
 *
 * Usage:
 * ```
 * val report = ModelManager.validate(bundleDir)
 * if (report.complete) {
 *     val manifest = ModelManager.loadManifest(bundleDir)
 *     val cfg = ModelConfig.fromManifest(manifest)
 *     // hand off to InferenceEngine
 * }
 * ```
 */
object ModelManager {
    const val MANIFEST_NAME = "manifest.json"

    /** Outcome of [validate]; [complete] iff `missing` and `shaMismatch` are both empty. */
    data class ValidationReport(
        /** Files declared by the manifest (or in `expectedSha`) that are not on disk. */
        val missing: List<String>,
        /** Files whose actual SHA-256 disagrees with manifest or `expectedSha`. */
        val shaMismatch: List<String>,
        /** Files present + SHA-OK. Useful for status UIs. */
        val ok: List<String>,
    ) {
        val complete: Boolean get() = missing.isEmpty() && shaMismatch.isEmpty()
    }

    /**
     * Confirm that every file declared in `manifest.json` exists in
     * [bundleDir], and that its on-disk SHA-256 matches both:
     *   1. the SHA in the manifest itself (always checked), AND
     *   2. the entry in [expectedSha], if one is provided for that file.
     *
     * Set [verifySha] = false to skip the (slow) hashing pass — useful for
     * a quick "is the bundle laid out at all" probe before kicking off
     * downloads. Default is `true` because correctness > a few seconds.
     *
     * @throws IllegalArgumentException if [bundleDir] has no manifest.
     */
    fun validate(
        bundleDir: File,
        expectedSha: Map<String, String> = emptyMap(),
        verifySha: Boolean = true,
    ): ValidationReport {
        val manifestFile = bundleDir.resolve(MANIFEST_NAME)
        require(manifestFile.isFile) { "manifest.json not found in ${bundleDir.absolutePath}" }
        val manifest = Manifest.parse(manifestFile.readText())

        val toCheck = (manifest.files.keys + expectedSha.keys).distinct()

        val missing = mutableListOf<String>()
        val shaMismatch = mutableListOf<String>()
        val ok = mutableListOf<String>()

        for (name in toCheck) {
            val f = bundleDir.resolve(name)
            if (!f.isFile) {
                missing += name
                continue
            }
            if (!verifySha) { ok += name; continue }

            val actual = f.inputStream().use { Sha256Verifier.streamHex(it) }
            val manifestSha = manifest.fileEntryOf(name)?.sha256
            val externalSha = expectedSha[name]

            val manifestOk = manifestSha?.equals(actual, ignoreCase = true) ?: true
            val externalOk = externalSha?.equals(actual, ignoreCase = true) ?: true

            if (manifestOk && externalOk) ok += name else shaMismatch += name
        }
        return ValidationReport(missing = missing, shaMismatch = shaMismatch, ok = ok)
    }

    /** Reads and parses `manifest.json` from [bundleDir]. */
    fun loadManifest(bundleDir: File): Manifest {
        val mf = bundleDir.resolve(MANIFEST_NAME)
        require(mf.isFile) { "manifest.json not found in ${bundleDir.absolutePath}" }
        return Manifest.parse(mf.readText())
    }

    /** Convenience: parse manifest then extract a [ModelConfig]. */
    fun loadConfig(bundleDir: File): ModelConfig =
        ModelConfig.fromManifest(loadManifest(bundleDir))
}
