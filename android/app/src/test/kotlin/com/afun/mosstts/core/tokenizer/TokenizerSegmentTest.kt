package com.afun.mosstts.core.tokenizer

import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * Per-segment lengths captured from `sp.encode(text)` on the actual
 * `tokenizer.model` for every literal `PromptBuilder` feeds the
 * tokenizer. If any of these counts diverges from Python, we know
 * exactly which segment to inspect (faster than diffing a 100-token
 * concatenated prompt).
 */
class TokenizerSegmentTest {
    private val tk by lazy {
        Tokenizer.load(javaClass.getResource("/tokenizer_kotlin.json")!!.readText())
    }

    @Test
    fun `prompt-builder segments encode to the same lengths as Python sp`() {
        val cases = listOf(
            "user_role"  to ("user\n" to 2),
            "ref_prefix" to ("<user_inst>\n- Reference(s):\n" to 10),
            "none"       to ("None" to 2),
            "after_ref"  to (
                "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n" +
                    "- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n" to 56
            ),
            "suffix"     to ("\n</user_inst>" to 2),
            "turn"       to ("\n" to 0),
            "asst_role"  to ("assistant\n" to 2),
        )
        val report = StringBuilder()
        var ok = true
        for ((name, payload) in cases) {
            val (text, expected) = payload
            val ids = tk.encode(text)
            if (ids.size != expected) {
                ok = false
                report.appendLine("$name: expected len=$expected, got len=${ids.size}, ids=${ids.toList()}")
            }
        }
        if (!ok) error(report.toString())
    }
}
