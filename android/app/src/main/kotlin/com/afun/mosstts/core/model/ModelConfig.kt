package com.afun.mosstts.core.model

/**
 * Runtime view of model architecture constants + IO-relevant token ids.
 *
 * Source of truth has shifted: the v1.0.0 release bundle no longer ships a
 * separate `config.json`. Instead, every value here is either:
 *   - read from `manifest.json:tts_config` ([Manifest.TtsConfig]), or
 *   - hard-coded as an architectural invariant of MOSS-TTS-Nano-100M
 *     (number of layers, heads, hidden_size, head_dim) and validated
 *     against the manifest's IO shapes at session creation time.
 *
 * The fixed-architecture constants below mirror the values in
 * `export_v2/scripts/demo_generate.py` (NUM_GLOBAL_LAYERS = 12, etc.) — if
 * a future model variant ever changes them we'd add a new bundle schema
 * version, not silently overload these.
 */
data class ModelConfig(
    /** Codebook count (16 for MOSS-TTS-Nano). */
    val nVq: Int,
    /** Per-codebook vocabulary (always 1024 in this release). */
    val audioCodebookSize: Int,
    /** Text tokenizer vocab size (16384). */
    val vocabSize: Int,

    // --- Special token ids (from manifest.tts_config) ---
    val audioPadTokenId: Int,
    val padTokenId: Int,
    val imStartTokenId: Int,
    val imEndTokenId: Int,
    val audioStartTokenId: Int,
    val audioEndTokenId: Int,
    val audioUserSlotTokenId: Int,
    val audioAssistantSlotTokenId: Int,
) {
    /** `n_vq + 1` — total columns per prompt row. */
    val rowStride: Int get() = nVq + 1

    /** Architectural invariants (validated by [Manifest] IO shapes). */
    val hiddenSize: Int = HIDDEN_SIZE
    val numGlobalLayers: Int = NUM_GLOBAL_LAYERS
    val numLocalLayers: Int = NUM_LOCAL_LAYERS
    val numHeads: Int = N_HEADS
    val headDim: Int = HEAD_DIM

    /** Codec output sample rate (48 kHz stereo per MOSS-Audio-Tokenizer-Nano). */
    val codecSampleRate: Int = CODEC_SR
    val codecChannels: Int = CODEC_CHANNELS

    companion object {
        const val HIDDEN_SIZE = 768
        const val NUM_GLOBAL_LAYERS = 12
        const val NUM_LOCAL_LAYERS = 1
        const val N_HEADS = 12
        const val HEAD_DIM = 64
        const val CODEC_SR = 48000
        const val CODEC_CHANNELS = 2

        /** Build a [ModelConfig] from the manifest's tts_config block. */
        fun fromManifest(m: Manifest): ModelConfig {
            val c = m.ttsConfig
            require(c.audioCodebookSizes.isNotEmpty()) { "audio_codebook_sizes must be non-empty" }
            val codebookSize = c.audioCodebookSizes.first()
            require(c.audioCodebookSizes.all { it == codebookSize }) {
                "all entries in audio_codebook_sizes must match (got ${c.audioCodebookSizes})"
            }
            require(c.audioCodebookSizes.size == c.nVq) {
                "audio_codebook_sizes.size=${c.audioCodebookSizes.size} != n_vq=${c.nVq}"
            }
            return ModelConfig(
                nVq = c.nVq,
                audioCodebookSize = codebookSize,
                vocabSize = c.vocabSize,
                audioPadTokenId = c.audioPadTokenId,
                padTokenId = c.padTokenId,
                imStartTokenId = c.imStartTokenId,
                imEndTokenId = c.imEndTokenId,
                audioStartTokenId = c.audioStartTokenId,
                audioEndTokenId = c.audioEndTokenId,
                audioUserSlotTokenId = c.audioUserSlotTokenId,
                audioAssistantSlotTokenId = c.audioAssistantSlotTokenId,
            )
        }
    }
}
