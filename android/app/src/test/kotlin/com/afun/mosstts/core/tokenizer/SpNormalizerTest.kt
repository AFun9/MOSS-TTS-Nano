package com.afun.mosstts.core.tokenizer

import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * Goldens captured from `sp.normalize(text)` on the actual tokenizer.model.
 *
 * `▁` is U+2581 (LOWER ONE EIGHTH BLOCK), the SentencePiece whitespace marker.
 */
class SpNormalizerTest {
    private val n = SpNormalizer.NMT_NFKC

    @Test
    fun `empty stays empty`() {
        assertThat(n.normalize("")).isEqualTo("")
    }

    @Test
    fun `pure whitespace returns empty (no dummy prefix)`() {
        assertThat(n.normalize(" ")).isEqualTo("")
        assertThat(n.normalize("   ")).isEqualTo("")
        assertThat(n.normalize("\t\n")).isEqualTo("")
    }

    @Test
    fun `simple ascii gets dummy prefix only`() {
        assertThat(n.normalize("AbC")).isEqualTo("▁AbC")
    }

    @Test
    fun `multiple internal spaces collapse to one`() {
        assertThat(n.normalize("  abc  def  ")).isEqualTo("▁abc▁def")
    }

    @Test
    fun `tab and newline are treated as whitespace`() {
        assertThat(n.normalize("a\tb\nc")).isEqualTo("▁a▁b▁c")
    }

    @Test
    fun `non-breaking space NBSP folds to whitespace`() {
        assertThat(n.normalize("Hello\u00A0World")).isEqualTo("▁Hello▁World")
    }

    @Test
    fun `accented latin survives NFKC unchanged`() {
        assertThat(n.normalize("café")).isEqualTo("▁café")
    }

    @Test
    fun `BEL control character is dropped`() {
        assertThat(n.normalize("\u0007bell")).isEqualTo("▁bell")
    }

    @Test
    fun `fullwidth ascii folds to halfwidth via NFKC`() {
        assertThat(n.normalize("FULLWIDTH ＡＢ")).isEqualTo("▁FULLWIDTH▁AB")
    }

    @Test
    fun `chinese punctuation passes through`() {
        assertThat(n.normalize("你好,世界!")).isEqualTo("▁你好,世界!")
    }
}
