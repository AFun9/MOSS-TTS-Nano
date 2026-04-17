package com.afun.mosstts.core.download

import com.google.common.truth.Truth.assertThat
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class Sha256VerifierTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test
    fun `hex of empty input matches RFC test vector`() {
        assertThat(Sha256Verifier.hex(byteArrayOf()))
            .isEqualTo("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }

    @Test
    fun `hex of abc matches RFC test vector`() {
        assertThat(Sha256Verifier.hex("abc".toByteArray()))
            .isEqualTo("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    @Test
    fun `streamHex matches hex on a multi-buffer file`() {
        // > 64 KiB so we exercise the buffered read loop.
        val bytes = ByteArray(200_000) { (it % 251).toByte() }
        val file = tmp.newFile("payload.bin").apply { writeBytes(bytes) }

        val expected = Sha256Verifier.hex(bytes)
        val streamed = file.inputStream().use { Sha256Verifier.streamHex(it) }

        assertThat(streamed).isEqualTo(expected)
    }

    @Test
    fun `verifyFile returns true on match and false on mismatch`() {
        val file = tmp.newFile("a.bin").apply { writeText("abc") }
        val correct = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assertThat(Sha256Verifier.verifyFile(file, correct)).isTrue()
        assertThat(Sha256Verifier.verifyFile(file, "deadbeef")).isFalse()
    }

    @Test
    fun `verifyFile is case-insensitive on the expected hex`() {
        val file = tmp.newFile("a.bin").apply { writeText("abc") }
        val upper = "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
        assertThat(Sha256Verifier.verifyFile(file, upper)).isTrue()
    }
}
