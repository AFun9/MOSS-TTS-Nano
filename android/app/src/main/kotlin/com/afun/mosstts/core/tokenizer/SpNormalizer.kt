package com.afun.mosstts.core.tokenizer

import java.text.Normalizer

/**
 * Approximation of SentencePiece's `nmt_nfkc` normalizer.
 *
 * SP applies, in order:
 *  1. NFKC + a small set of NMT-specific rules (mostly: drop ASCII control
 *     characters, fold every kind of whitespace into a single space).
 *  2. `remove_extra_whitespaces`: collapse runs of spaces and trim ends.
 *  3. `escape_whitespaces`: every space → '▁' (U+2581).
 *  4. `add_dummy_prefix`: if the result is non-empty, prepend '▁'.
 *
 * We don't unpack SP's `precompiled_charsmap` blob; instead, NFKC is the JVM
 * `java.text.Normalizer` and the NMT subset is hand-written here. The
 * golden-fixture test in M1.6d will fail loudly on any locale where this
 * approximation diverges from the Python tokenizer; the fixtures are also
 * wide enough (CJK / RU / AR / TH / VI / Latin-with-diacritics / emoji /
 * smart quotes / NBSP / ZWJ etc.) to make divergences visible early.
 */
class SpNormalizer(
    val name: String,
    val addDummyPrefix: Boolean,
    val removeExtraWhitespaces: Boolean,
    val escapeWhitespaces: Boolean,
) {
    fun normalize(input: String, addDummyPrefix: Boolean = this.addDummyPrefix): String {
        var s = if (name == "nmt_nfkc" || name == "nfkc") nfkcNmt(input) else input
        if (removeExtraWhitespaces) s = collapseWhitespace(s)
        if (escapeWhitespaces) s = s.replace(' ', WHITESPACE_BLOCK)
        if (addDummyPrefix && s.isNotEmpty()) {
            s = WHITESPACE_BLOCK + s
        }
        return s
    }

    private fun nfkcNmt(input: String): String {
        val n = Normalizer.normalize(input, Normalizer.Form.NFKC)
        val sb = StringBuilder(n.length)
        for (c in n) {
            val code = c.code
            when {
                // Drop C0/C1 control chars (matches SP's nmt rule). Note that
                // Java treats some of these as whitespace (e.g. \t, \n) — we
                // first turn whitespace into ' ', so dropping the rest is safe.
                code <= 0x1F || code in 0x7F..0x9F -> {
                    if (c.isWhitespace()) sb.append(' ')  // \t \n \r etc → ' '
                    // else: drop
                }
                // Anything Unicode considers whitespace folds to ' '.
                // This covers NBSP (U+00A0), ideographic space (U+3000),
                // narrow no-break (U+202F), zero-width space (U+200B), etc.
                c.isWhitespace() || code == 0x200B -> sb.append(' ')
                // SP's precompiled `nmt_nfkc` charsmap drops a handful of
                // zero-width / bidi format chars while keeping ZWJ (U+200D),
                // which is needed for many Indic / Arabic ligatures and
                // appears as a real piece in our vocab. Without these drops
                // the byte_fallback path emits 3-byte sequences that diverge
                // from Python sp on inputs that contain them.
                code == 0x200C ||  // ZWNJ
                    code == 0x200E || code == 0x200F ||  // LRM / RLM
                    code in 0x202A..0x202E ||  // bidi explicit overrides
                    code == 0xFEFF  // BOM / ZWNBSP
                    -> Unit  // drop
                else -> sb.append(c)
            }
        }
        return sb.toString()
    }

    private fun collapseWhitespace(s: String): String {
        if (s.isEmpty()) return s
        val sb = StringBuilder(s.length)
        var prevSpace = false
        var startedNonSpace = false
        for (c in s) {
            if (c == ' ') {
                if (startedNonSpace) prevSpace = true
                // else: leading space, skip
            } else {
                if (prevSpace) sb.append(' ')
                sb.append(c)
                startedNonSpace = true
                prevSpace = false
            }
        }
        return sb.toString()
    }

    companion object {
        const val WHITESPACE_BLOCK = '\u2581'  // '▁'

        /** Defaults matching `tokenizer.model` for MOSS-TTS-Nano. */
        val NMT_NFKC = SpNormalizer(
            name = "nmt_nfkc",
            addDummyPrefix = true,
            removeExtraWhitespaces = true,
            escapeWhitespaces = true,
        )
    }
}
