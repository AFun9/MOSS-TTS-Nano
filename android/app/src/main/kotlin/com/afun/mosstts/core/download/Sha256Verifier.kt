package com.afun.mosstts.core.download

import java.io.File
import java.io.InputStream
import java.security.MessageDigest

/**
 * SHA-256 helpers used by [com.afun.mosstts.core.model.ModelManager] and the
 * download path. Pure JVM, no Android dependencies, so it runs in JUnit
 * unit tests without Robolectric.
 */
object Sha256Verifier {
    private const val BUFFER_BYTES = 64 * 1024

    fun hex(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .toHex()

    /** Streams [input] through SHA-256, closing it via the caller's `use { }`. */
    fun streamHex(input: InputStream): String {
        val md = MessageDigest.getInstance("SHA-256")
        val buf = ByteArray(BUFFER_BYTES)
        while (true) {
            val n = input.read(buf)
            if (n <= 0) break
            md.update(buf, 0, n)
        }
        return md.digest().toHex()
    }

    /** Convenience wrapper that streams [file] and compares to [expectedHex] (case-insensitive). */
    fun verifyFile(file: File, expectedHex: String): Boolean {
        val actual = file.inputStream().use { streamHex(it) }
        return actual.equals(expectedHex, ignoreCase = true)
    }

    private fun ByteArray.toHex(): String {
        val out = CharArray(size * 2)
        val table = "0123456789abcdef"
        for (i in indices) {
            val v = this[i].toInt() and 0xff
            out[i * 2] = table[v ushr 4]
            out[i * 2 + 1] = table[v and 0x0f]
        }
        return String(out)
    }
}
