package com.afun.mosstts.core.text

import java.util.Locale

/**
 * Top-level text normalization pipeline: L1 (common cleanup) then L2 (number spellout).
 *
 * Runs before tokenization in [com.afun.mosstts.core.engine.TtsEngine].
 */
class TextNormalizer(locale: Locale = Locale.CHINESE) {

    private val numberNormalizer = IcuNumberNormalizer(IcuNumberSpeller(locale))

    fun normalize(text: String): String {
        val cleaned = CommonNormalizer.normalize(text)
        return numberNormalizer.normalize(cleaned)
    }
}
