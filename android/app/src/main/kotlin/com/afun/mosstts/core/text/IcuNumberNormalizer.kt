package com.afun.mosstts.core.text

import java.util.Locale

/**
 * L2 text normalizer: converts numeric tokens to spoken words.
 *
 * Uses a [NumberSpeller] to convert individual number strings to words.
 * On Android, [IcuNumberSpeller] uses `android.icu.text.RuleBasedNumberFormat` SPELLOUT.
 */
class IcuNumberNormalizer(
    private val speller: NumberSpeller,
) {
    fun normalize(text: String): String {
        return replaceNumbers(text) { token -> speller.spell(token) }
    }

    companion object {
        internal val NUMBER_PATTERN = Regex("""-?\d+(?:\.\d+)?%?""")

        internal fun replaceNumbers(text: String, format: (String) -> String): String {
            return NUMBER_PATTERN.replace(text) { match -> format(match.value) }
        }
    }
}

/** Abstraction for number-to-word conversion, allows JVM-testable mocking. */
fun interface NumberSpeller {
    fun spell(token: String): String
}

/**
 * Android ICU-based speller using [android.icu.text.RuleBasedNumberFormat].
 * Must only be instantiated on Android runtime.
 */
class IcuNumberSpeller(private val locale: Locale = Locale.CHINESE) : NumberSpeller {

    private val formatter by lazy {
        val cls = Class.forName("android.icu.text.RuleBasedNumberFormat")
        val spelloutField = cls.getField("SPELLOUT")
        val spelloutValue = spelloutField.getInt(null)
        val ctor = cls.getConstructor(Locale::class.java, Int::class.javaPrimitiveType)
        ctor.newInstance(locale, spelloutValue)
    }

    private val formatMethod by lazy {
        formatter.javaClass.getMethod("format", Double::class.javaPrimitiveType)
    }

    override fun spell(token: String): String {
        val isPercent = token.endsWith('%')
        val numStr = if (isPercent) token.dropLast(1) else token
        val value = numStr.toDoubleOrNull() ?: return token
        val spelled = formatMethod.invoke(formatter, value) as String
        if (!isPercent) return spelled
        return formatPercent(spelled)
    }

    private fun formatPercent(spelledNumber: String): String {
        val lang = locale.language
        return when {
            lang == "zh" || lang == "ja" -> "百分之$spelledNumber"
            lang == "ko" -> "퍼센트 $spelledNumber"
            lang == "ru" -> "$spelledNumber процентов"
            else -> "$spelledNumber percent"
        }
    }
}
