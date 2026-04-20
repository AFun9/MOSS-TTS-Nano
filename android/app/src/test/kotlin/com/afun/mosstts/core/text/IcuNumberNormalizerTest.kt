package com.afun.mosstts.core.text

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Pure-JVM tests for the regex extraction and replacement logic.
 * ICU format output tests require Android runtime (androidTest).
 */
class IcuNumberNormalizerTest {

    @Test
    fun `extracts simple integers`() {
        val matches = IcuNumberNormalizer.NUMBER_PATTERN.findAll("价格是123元").map { it.value }.toList()
        assertEquals(listOf("123"), matches)
    }

    @Test
    fun `extracts decimals`() {
        val matches = IcuNumberNormalizer.NUMBER_PATTERN.findAll("温度是36.5度").map { it.value }.toList()
        assertEquals(listOf("36.5"), matches)
    }

    @Test
    fun `extracts negative numbers`() {
        val matches = IcuNumberNormalizer.NUMBER_PATTERN.findAll("零下-7度").map { it.value }.toList()
        assertEquals(listOf("-7"), matches)
    }

    @Test
    fun `extracts percentages`() {
        val matches = IcuNumberNormalizer.NUMBER_PATTERN.findAll("增长了50%").map { it.value }.toList()
        assertEquals(listOf("50%"), matches)
    }

    @Test
    fun `extracts multiple numbers in one sentence`() {
        val matches = IcuNumberNormalizer.NUMBER_PATTERN.findAll("从100到200，涨幅为50%").map { it.value }.toList()
        assertEquals(listOf("100", "200", "50%"), matches)
    }

    @Test
    fun `replaceNumbers substitutes all matches`() {
        val result = IcuNumberNormalizer.replaceNumbers("有123个和456个") { "[${it}]" }
        assertEquals("有[123]个和[456]个", result)
    }

    @Test
    fun `replaceNumbers handles no numbers`() {
        val result = IcuNumberNormalizer.replaceNumbers("没有数字") { "X" }
        assertEquals("没有数字", result)
    }

    @Test
    fun `replaceNumbers handles complex mixed text`() {
        val result = IcuNumberNormalizer.replaceNumbers("温度-3.5度，湿度80%") { "<${it}>" }
        assertEquals("温度<-3.5>度，湿度<80%>", result)
    }

    @Test
    fun `does not match lone dots or percent signs`() {
        val matches = IcuNumberNormalizer.NUMBER_PATTERN.findAll("hello.world %done").map { it.value }.toList()
        assertEquals(emptyList<String>(), matches)
    }
}
