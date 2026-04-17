package com.afun.mosstts.core.model

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class ManifestTest {
    private val sample by lazy {
        javaClass.getResource("/manifest_v2_sample.json")!!.readText()
    }

    @Test
    fun `parses schema and quantization`() {
        val m = Manifest.parse(sample)
        assertThat(m.schema).isEqualTo("moss-tts-nano-onnx-bundle/v2")
        assertThat(m.quantization.weightType).isEqualTo("QInt8")
        assertThat(m.quantization.perChannel).isFalse()
        assertThat(m.quantization.reduceRange).isFalse()
    }

    @Test
    fun `lists every file the runtime needs`() {
        val names = Manifest.parse(sample).files.map { it.name }
        // INT8 onnx subset + tokenizer + json metadata; FP32 variants are
        // present in the bundle as well but not required by the Android app.
        assertThat(names).containsAtLeast(
            "audio_decoder_int8.onnx",
            "audio_encoder_int8.onnx",
            "global_transformer_int8.onnx",
            "local_decoder_text_int8.onnx",
            "local_decoder_audio_int8.onnx",
            "tokenizer.model",
            "config.json",
            "audio_decoder_state_spec.json",
        )
    }

    @Test
    fun `entry exposes file size in bytes`() {
        val gt = Manifest.parse(sample).files
            .single { it.name == "global_transformer_int8.onnx" }
        assertThat(gt.sizeBytes).isEqualTo(110_956_202L)
    }

    @Test
    fun `rejects malformed json`() {
        assertThat(runCatching { Manifest.parse("not json") }.isFailure).isTrue()
    }

    @Test
    fun `rejects missing required fields`() {
        val onlyFiles = """{"files": []}"""
        assertThat(runCatching { Manifest.parse(onlyFiles) }.isFailure).isTrue()
    }
}
