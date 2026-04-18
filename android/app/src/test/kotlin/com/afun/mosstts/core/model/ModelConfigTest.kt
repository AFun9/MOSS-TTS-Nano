package com.afun.mosstts.core.model

import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * v1.0.0 release: `ModelConfig` is now derived from the manifest's
 * `tts_config` block, with architecture invariants (hidden_size,
 * num_layers, num_heads, head_dim) hard-coded as constants and
 * cross-checked elsewhere against the manifest's IO shapes.
 */
class ModelConfigTest {
    private val manifestText by lazy {
        javaClass.getResource("/release_manifest_v1.json")!!.readText()
    }

    private val cfg by lazy { ModelConfig.fromManifest(Manifest.parse(manifestText)) }

    @Test
    fun `derives n_vq and codebook size`() {
        assertThat(cfg.nVq).isEqualTo(16)
        assertThat(cfg.audioCodebookSize).isEqualTo(1024)
        assertThat(cfg.rowStride).isEqualTo(17)
    }

    @Test
    fun `exposes every special token id`() {
        assertThat(cfg.imStartTokenId).isEqualTo(4)
        assertThat(cfg.imEndTokenId).isEqualTo(5)
        assertThat(cfg.audioStartTokenId).isEqualTo(6)
        assertThat(cfg.audioEndTokenId).isEqualTo(7)
        assertThat(cfg.audioUserSlotTokenId).isEqualTo(8)
        assertThat(cfg.audioAssistantSlotTokenId).isEqualTo(9)
        assertThat(cfg.audioPadTokenId).isEqualTo(1024)
        assertThat(cfg.padTokenId).isEqualTo(3)
    }

    @Test
    fun `architectural invariants match manifest IO`() {
        // These constants must agree with `manifest.tts.io[prefill].outputs[present_key_*]`
        // shape `[1, prefill_seq, 12, 64]` and `global_hidden` shape `[1, prefill_seq, 768]`.
        assertThat(cfg.hiddenSize).isEqualTo(768)
        assertThat(cfg.numHeads).isEqualTo(12)
        assertThat(cfg.headDim).isEqualTo(64)
        assertThat(cfg.numGlobalLayers).isEqualTo(12)
        assertThat(cfg.numLocalLayers).isEqualTo(1)
    }

    @Test
    fun `codec rate constants are 48 kHz stereo`() {
        assertThat(cfg.codecSampleRate).isEqualTo(48000)
        assertThat(cfg.codecChannels).isEqualTo(2)
    }

    @Test
    fun `rejects manifest with non-uniform codebook sizes`() {
        // Replace exactly one of the per-line `1024`s inside audio_codebook_sizes
        // with a different value to trigger the uniformity check.
        val needle = "\"audio_codebook_sizes\": ["
        val start = manifestText.indexOf(needle)
        val arrEnd = manifestText.indexOf("]", start)
        val arrayBlock = manifestText.substring(start, arrEnd + 1)
        val mutatedArray = arrayBlock.replaceFirst("1024", "512")
        val raw = manifestText.replaceRange(start, arrEnd + 1, mutatedArray)

        val m = Manifest.parse(raw)
        val ex = runCatching { ModelConfig.fromManifest(m) }
        assertThat(ex.isFailure).isTrue()
    }
}
