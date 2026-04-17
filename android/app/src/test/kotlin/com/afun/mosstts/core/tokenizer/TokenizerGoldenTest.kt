package com.afun.mosstts.core.tokenizer

import com.google.common.truth.Truth.assertThat
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.junit.Test

/**
 * End-to-end byte-equality test against Python sentencepiece.
 *
 * Loads the runtime vocab (`tokenizer_kotlin.json`, the same file the app
 * uses on device) and the golden fixtures (`tokenizer_golden.json`,
 * captured from `sp.encode(text)` on the same `tokenizer.model`).
 *
 * Every fixture's ids must match exactly. Any divergence fails the build
 * with the offending input + expected/actual ids printed.
 */
class TokenizerGoldenTest {
    private val tokenizer by lazy {
        val raw = javaClass.getResource("/tokenizer_kotlin.json")
            ?: error("tokenizer_kotlin.json not found on test classpath; check Gradle test sourceSet")
        Tokenizer.load(raw.readText())
    }
    private val fixtures by lazy {
        val raw = javaClass.getResource("/tokenizer_golden.json")!!.readText()
        Json.decodeFromString<List<GoldenFixture>>(raw)
    }

    @Test
    fun `loads the bundled tokenizer json`() {
        assertThat(tokenizer.vocabSize).isEqualTo(16384)
        assertThat(tokenizer.unkId).isEqualTo(0)
        assertThat(tokenizer.bosId).isEqualTo(1)
        assertThat(tokenizer.eosId).isEqualTo(2)
        assertThat(tokenizer.padId).isEqualTo(3)
        // im_start / im_end / audio_* used by PromptBuilder must resolve.
        assertThat(tokenizer.specialTokenId("<|im_start|>")).isEqualTo(4)
        assertThat(tokenizer.specialTokenId("<|im_end|>")).isEqualTo(5)
        assertThat(tokenizer.specialTokenId("<|audio_start|>")).isEqualTo(6)
    }

    @Test
    fun `golden fixture file is the one currently shipped by the python exporter`() {
        // Pin the count so an out-of-date golden file (e.g. exporter ran but
        // test classpath cached an older copy) fails loud instead of silently
        // matching a smaller, weaker set.
        assertThat(fixtures.size).isAtLeast(100)
    }

    @Test
    fun `every golden fixture matches Python sentencepiece byte-for-byte`() {
        val mismatches = mutableListOf<String>()
        for ((idx, fx) in fixtures.withIndex()) {
            val actual = tokenizer.encode(fx.text).toList()
            if (actual != fx.ids) {
                mismatches += buildString {
                    append("[#%d] %s\n".format(idx, repr(fx.text)))
                    append("    expected: ${fx.ids}\n")
                    append("    actual:   $actual\n")
                    append("    expected pieces: ${fx.pieces}")
                }
                if (mismatches.size >= 5) break  // cap output noise
            }
        }
        if (mismatches.isNotEmpty()) {
            error(
                "Tokenizer diverged from Python sentencepiece on " +
                    "${mismatches.size}/${fixtures.size} fixtures (showing up to 5):\n\n" +
                    mismatches.joinToString("\n\n"),
            )
        }
    }

    private fun repr(s: String): String = buildString {
        append('"')
        for (c in s) when {
            c == '"' -> append("\\\"")
            c == '\\' -> append("\\\\")
            c.code in 0x20..0x7E -> append(c)
            c == '\n' -> append("\\n")
            c == '\t' -> append("\\t")
            else -> append("\\u%04x".format(c.code))
        }
        append('"')
    }

    @Serializable
    private data class GoldenFixture(
        val text: String,
        val ids: List<Int>,
        val pieces: List<String>,
    )
}
