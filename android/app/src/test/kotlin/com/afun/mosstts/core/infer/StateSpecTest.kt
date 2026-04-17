package com.afun.mosstts.core.infer

import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * Pin down the exact schema of `audio_decoder_state_spec.json` —
 * the InferenceLoop assumes very specific input/output ordering and
 * tensor naming when it constructs the per-step ORT feed.
 */
class StateSpecTest {
    private val spec by lazy {
        StateSpec.parse(javaClass.getResource("/state_spec_sample.json")!!.readText())
    }

    @Test
    fun `parses the AudioDecoder constants the engine needs every frame`() {
        assertThat(spec.numQuantizers).isEqualTo(16)
        assertThat(spec.sampleRate).isEqualTo(48000)
        assertThat(spec.downsampleRate).isEqualTo(3840)
    }

    @Test
    fun `input_names ordering starts with the live frame, then KV state`() {
        // Position 0 must be the int64 `codes` slot — the only non-state
        // input we feed each step.
        assertThat(spec.inputNames[0]).isEqualTo("codes")
        // Every other input must correspond to a KV-state tensor name.
        val expectedStateNames = stateTensorNames(spec)
        assertThat(spec.inputNames.drop(1)).containsExactlyElementsIn(expectedStateNames).inOrder()
    }

    @Test
    fun `output_names start with audio and audio_lengths, then new state`() {
        assertThat(spec.outputNames[0]).isEqualTo("audio")
        assertThat(spec.outputNames[1]).isEqualTo("audio_lengths")
        // Every remaining output must begin with `new_` and refer to a
        // state tensor — that is the contract `step()` relies on to copy
        // outputs back into the cache.
        for (i in 2 until spec.outputNames.size) {
            val n = spec.outputNames[i]
            assertThat(n).startsWith("new_")
            assertThat(spec.inputNames).contains(n.removePrefix("new_"))
        }
    }

    @Test
    fun `attention specs have the expected shape fields`() {
        // Sanity: 12 attention layers (4 transformers × 3 layers each in
        // the v0 export). All entries should have positive dims.
        assertThat(spec.attentionSpecs).hasSize(12)
        for (a in spec.attentionSpecs) {
            assertThat(a.numHeads).isGreaterThan(0)
            assertThat(a.context).isGreaterThan(0)
            assertThat(a.headDim).isGreaterThan(0)
            assertThat(a.layerIndex).isAtLeast(0)
            assertThat(a.decoderModuleIndex).isAtLeast(1)
        }
    }

    @Test
    fun `transformer specs cover every decoder module that appears in attention specs`() {
        val attnDecoders = spec.attentionSpecs.map { it.decoderModuleIndex }.toSet()
        val txDecoders = spec.transformerSpecs.map { it.decoderModuleIndex }.toSet()
        assertThat(txDecoders).isEqualTo(attnDecoders)
    }

    private fun stateTensorNames(spec: StateSpec): List<String> {
        val out = ArrayList<String>()
        val grouped = spec.attentionSpecs.groupBy { it.decoderModuleIndex }
        for (ts in spec.transformerSpecs) {
            val di = ts.decoderModuleIndex
            out += "decoder_${di}_transformer_offset"
            for (a in grouped[di].orEmpty()) {
                val p = "decoder_${a.decoderModuleIndex}_layer_${a.layerIndex}"
                out += listOf(
                    "${p}_cached_keys",
                    "${p}_cached_values",
                    "${p}_cached_positions",
                    "${p}_offset",
                )
            }
        }
        return out
    }
}
