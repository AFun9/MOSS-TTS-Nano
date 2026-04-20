package com.afun.mosstts.core.text

import org.junit.Assert.assertEquals
import org.junit.Test

class CommonNormalizerTest {

    @Test
    fun `strips control characters but keeps newline and tab`() {
        assertEquals("helloworld", CommonNormalizer.normalize("hello\u0000world"))
        assertEquals("a b", CommonNormalizer.normalize("a\tb"))
        assertEquals("a b", CommonNormalizer.normalize("a\nb"))
        assertEquals("ab", CommonNormalizer.normalize("a\u0001\u001Fb"))
    }

    @Test
    fun `strips zero-width characters`() {
        assertEquals("你好", CommonNormalizer.normalize("你\u200B好"))
        assertEquals("ab", CommonNormalizer.normalize("a\uFEFFb"))
        assertEquals("test", CommonNormalizer.normalize("te\u200Dst"))
    }

    @Test
    fun `converts fullwidth ASCII to halfwidth`() {
        assertEquals("ABC123", CommonNormalizer.normalize("ＡＢＣ１２３"))
        assertEquals("Hello, World!", CommonNormalizer.normalize("Ｈｅｌｌｏ，\u3000Ｗｏｒｌｄ！"))
    }

    @Test
    fun `collapses whitespace and trims`() {
        assertEquals("a b", CommonNormalizer.normalize("  a   b  "))
        assertEquals("hello world", CommonNormalizer.normalize("\t hello \n\n world \t"))
        assertEquals("x", CommonNormalizer.normalize("   x   "))
    }

    @Test
    fun `truncates to 300 characters`() {
        val long = "a".repeat(301)
        val result = CommonNormalizer.normalize(long)
        assertEquals(300, result.length)
    }

    @Test
    fun `does not truncate at exactly 300`() {
        val exact = "b".repeat(300)
        assertEquals(300, CommonNormalizer.normalize(exact).length)
    }

    @Test
    fun `mixed scenario`() {
        val input = "  Ｈｅｌｌｏ\u0000\u200B，\u3000世界！  "
        assertEquals("Hello, 世界!", CommonNormalizer.normalize(input))
    }

    @Test
    fun `empty and blank input`() {
        assertEquals("", CommonNormalizer.normalize(""))
        assertEquals("", CommonNormalizer.normalize("   "))
        assertEquals("", CommonNormalizer.normalize("\u200B\u200C"))
    }
}
