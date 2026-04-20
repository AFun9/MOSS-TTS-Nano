package com.afun.mosstts.core.text

/**
 * L1 text normalizer: general-purpose cleanup applied before tokenization.
 *
 * Pipeline (order matters):
 * 1. Strip control characters (keep TAB, LF, CR)
 * 2. Strip zero-width / invisible Unicode characters
 * 3. Fullwidth ASCII -> halfwidth
 * 4. Collapse consecutive whitespace, trim
 * 5. Truncate to [MAX_CHARS]
 */
object CommonNormalizer {

    const val MAX_CHARS = 300

    private val CONTROL_CHARS = Regex("[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
    private val ZERO_WIDTH = Regex("[\u200B-\u200F\uFEFF\u2028\u2029]")
    private val MULTI_WHITESPACE = Regex("\\s+")

    fun normalize(text: String): String {
        var s = text
        s = CONTROL_CHARS.replace(s, "")
        s = ZERO_WIDTH.replace(s, "")
        s = toHalfWidth(s)
        s = MULTI_WHITESPACE.replace(s, " ").trim()
        if (s.length > MAX_CHARS) s = s.substring(0, MAX_CHARS)
        return s
    }

    private fun toHalfWidth(text: String): String {
        val sb = StringBuilder(text.length)
        for (ch in text) {
            when {
                ch in '\uFF01'..'\uFF5E' -> sb.append((ch.code - 0xFEE0).toChar())
                ch == '\u3000' -> sb.append(' ')
                else -> sb.append(ch)
            }
        }
        return sb.toString()
    }
}
