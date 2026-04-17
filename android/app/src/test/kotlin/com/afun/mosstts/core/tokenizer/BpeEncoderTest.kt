package com.afun.mosstts.core.tokenizer

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class BpeEncoderTest {
    /**
     * Tiny hand-built vocab so we can reason about merges by hand.
     *
     * id 0..2:  byte fallback for [<0x00>..<0x02>] (we only need 256 of them
     *           in production; here we map all 256 bytes to consecutive ids
     *           starting at BYTE_FALLBACK_START).
     */
    private val pieces = mapOf(
        "<unk>" to 0,
        "a" to 1,
        "b" to 2,
        "c" to 3,
        "ab" to 4,
        "bc" to 5,
        "abc" to 6,
        "あ" to 7,  // single non-ASCII codepoint in vocab
    )
    private val scoresArr = floatArrayOf(
        0.0f,    // <unk>
        -3.0f,   // a
        -3.0f,   // b
        -3.0f,   // c
        -2.0f,   // ab   higher score → preferred
        -2.5f,   // bc
        -1.0f,   // abc  highest → most preferred
        -3.0f,   // あ
    ).copyOf(BYTE_FALLBACK_START + 256).also { full ->
        // Pad scores for the byte_fallback range with -inf so they're never
        // picked as merges (matches what SP does for type=BYTE pieces).
        for (i in BYTE_FALLBACK_START until full.size) full[i] = Float.NEGATIVE_INFINITY
    }

    private val vocabWithBytes: Map<String, Int> = buildMap {
        putAll(pieces)
        for (b in 0..255) put(byteToken(b), BYTE_FALLBACK_START + b)
    }

    private val enc = BpeEncoder(
        vocab = vocabWithBytes,
        scores = scoresArr,
        byteFallbackStart = BYTE_FALLBACK_START,
        unkId = 0,
    )

    @Test
    fun `empty input returns empty array`() {
        assertThat(enc.encode("").toList()).isEmpty()
    }

    @Test
    fun `single in-vocab codepoint returns its id`() {
        assertThat(enc.encode("a").toList()).containsExactly(1).inOrder()
    }

    @Test
    fun `merge picks highest score (abc beats ab beats a-b-c)`() {
        // Greedy: 'abc' has score -1.0, the best.
        assertThat(enc.encode("abc").toList()).containsExactly(6).inOrder()
    }

    @Test
    fun `partial merges fall back to leftover pieces`() {
        // "ab" -> [ab(4)]  (no abc match because only 2 chars)
        assertThat(enc.encode("ab").toList()).containsExactly(4).inOrder()
        // "abca" -> first merges "abc"(6), then "a"(1) is left alone
        assertThat(enc.encode("abca").toList()).containsExactly(6, 1).inOrder()
    }

    @Test
    fun `OOV ascii codepoint expands via byte_fallback (UTF-8 = 1 byte)`() {
        // 'z' is not in vocab; UTF-8 of 'z' = 0x7A
        val expected = listOf(BYTE_FALLBACK_START + 0x7A)
        assertThat(enc.encode("z").toList()).isEqualTo(expected)
    }

    @Test
    fun `OOV multi-byte codepoint expands per UTF-8 byte`() {
        // 'い' (U+3044) UTF-8 = E3 81 84
        val expected = listOf(
            BYTE_FALLBACK_START + 0xE3,
            BYTE_FALLBACK_START + 0x81,
            BYTE_FALLBACK_START + 0x84,
        )
        assertThat(enc.encode("い").toList()).isEqualTo(expected)
    }

    @Test
    fun `in-vocab non-ascii codepoint stays as one id (no byte_fallback)`() {
        // 'あ' is in vocab → id 7
        assertThat(enc.encode("あ").toList()).containsExactly(7).inOrder()
    }

    @Test
    fun `mix of in-vocab and OOV chars`() {
        // 'a' -> 1; 'z' -> byte_fallback 0x7A; 'b' -> 2
        val expected = listOf(1, BYTE_FALLBACK_START + 0x7A, 2)
        assertThat(enc.encode("azb").toList()).isEqualTo(expected)
    }

    @Test
    fun `astral plane codepoint (emoji) goes through byte_fallback`() {
        // '🙂' (U+1F642) UTF-8 = F0 9F 99 82  (4 bytes; surrogate pair in UTF-16)
        val expected = listOf(
            BYTE_FALLBACK_START + 0xF0,
            BYTE_FALLBACK_START + 0x9F,
            BYTE_FALLBACK_START + 0x99,
            BYTE_FALLBACK_START + 0x82,
        )
        assertThat(enc.encode("🙂").toList()).isEqualTo(expected)
    }

    companion object {
        private const val BYTE_FALLBACK_START = 8
        private fun byteToken(b: Int) = "<0x%02X>".format(b)
    }
}
