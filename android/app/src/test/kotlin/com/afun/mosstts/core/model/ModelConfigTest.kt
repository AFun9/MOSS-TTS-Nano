package com.afun.mosstts.core.model

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class ModelConfigTest {
    private val sample by lazy {
        javaClass.getResource("/config_sample.json")!!.readText()
    }

    @Test
    fun `parses VQ-related dims`() {
        val c = ModelConfig.parse(sample)
        assertThat(c.nVq).isEqualTo(16)
        assertThat(c.audioCodebookSize).isEqualTo(1024)
    }

    @Test
    fun `parses every special token id required by PromptBuilder`() {
        val c = ModelConfig.parse(sample)
        assertThat(c.imStartTokenId).isEqualTo(4)
        assertThat(c.imEndTokenId).isEqualTo(5)
        assertThat(c.audioStartTokenId).isEqualTo(6)
        assertThat(c.audioEndTokenId).isEqualTo(7)
        assertThat(c.audioUserSlotTokenId).isEqualTo(8)
        assertThat(c.audioAssistantSlotTokenId).isEqualTo(9)
        assertThat(c.audioPadTokenId).isEqualTo(1024)
        assertThat(c.padTokenId).isEqualTo(3)
    }

    @Test
    fun `parses audio tokenizer params`() {
        // NB: plan template said sampleRate==24000 but the real config exposes
        // audio_tokenizer_sample_rate==48000. The 24000 in onnx_infer is a
        // hard-coded playback rate (mono mix of two 24 kHz codec streams);
        // see Decision #30 in DEVLOG.
        val c = ModelConfig.parse(sample)
        assertThat(c.audioTokenizerSampleRate).isEqualTo(48000)
        assertThat(c.audioTokenizerDownsampleRate).isEqualTo(3840)
        assertThat(c.audioTokenizerNumChannels).isEqualTo(2)
    }

    @Test
    fun `parses sampling defaults`() {
        val s = ModelConfig.parse(sample).samplingDefaults
        assertThat(s.audioTopK).isEqualTo(25)
        assertThat(s.audioTopP).isWithin(1e-6f).of(0.95f)
        assertThat(s.audioTemperature).isWithin(1e-6f).of(0.8f)
        assertThat(s.audioRepetitionPenalty).isWithin(1e-6f).of(1.2f)
    }

    @Test
    fun `keeps raw json so future fields can be read without re-parsing`() {
        val c = ModelConfig.parse(sample)
        assertThat(c.rawJson).contains("\"hidden_size\"")
    }

    @Test
    fun `rejects malformed json`() {
        assertThat(runCatching { ModelConfig.parse("not json") }.isFailure).isTrue()
    }
}
