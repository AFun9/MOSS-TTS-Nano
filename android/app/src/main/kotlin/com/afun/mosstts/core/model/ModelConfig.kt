package com.afun.mosstts.core.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Mirror of `onnx_export/config.json`.
 *
 * Only the fields the Android runtime currently uses are surfaced as typed
 * properties. The original JSON is preserved in [rawJson] so future code
 * (e.g. extra hyper-parameters) can extract more without re-parsing on
 * every call.
 *
 * NB: `audio_tokenizer_sample_rate=48000` is the *codec* rate. The reference
 * Python pipeline (`onnx_infer.py`) writes WAVs at 24 kHz mono after mixing
 * the two 24 kHz codec channels — that 24 kHz is a runtime constant and is
 * intentionally NOT read from this config; see Decision #30 in DEVLOG.
 */
@Serializable
data class ModelConfig(
    @SerialName("nq") val nq: Int = 16,
    @SerialName("n_vq") val nVq: Int,
    @SerialName("hidden_size") val hiddenSize: Int = 768,
    @SerialName("num_layers") val numLayers: Int = 12,
    @SerialName("num_heads") val numHeads: Int = 12,
    @SerialName("head_dim") val headDim: Int = 64,
    @SerialName("vocab_size") val vocabSize: Int,
    @SerialName("audio_codebook_size") val audioCodebookSize: Int,
    @SerialName("audio_pad_token_id") val audioPadTokenId: Int,
    @SerialName("pad_token_id") val padTokenId: Int,
    @SerialName("im_start_token_id") val imStartTokenId: Int,
    @SerialName("im_end_token_id") val imEndTokenId: Int,
    @SerialName("audio_start_token_id") val audioStartTokenId: Int,
    @SerialName("audio_end_token_id") val audioEndTokenId: Int,
    @SerialName("audio_user_slot_token_id") val audioUserSlotTokenId: Int,
    @SerialName("audio_assistant_slot_token_id") val audioAssistantSlotTokenId: Int,
    @SerialName("audio_tokenizer_sample_rate") val audioTokenizerSampleRate: Int,
    @SerialName("audio_tokenizer_downsample_rate") val audioTokenizerDownsampleRate: Int,
    @SerialName("audio_tokenizer_num_channels") val audioTokenizerNumChannels: Int,
    @SerialName("sampling_defaults") val samplingDefaults: SamplingDefaults,
) {
    /** Original JSON text, kept so callers can extract uncommon fields lazily. */
    var rawJson: String = ""
        internal set

    @Serializable
    data class SamplingDefaults(
        @SerialName("text_temperature") val textTemperature: Float = 1.0f,
        @SerialName("text_top_p") val textTopP: Float = 1.0f,
        @SerialName("text_top_k") val textTopK: Int = 50,
        @SerialName("audio_temperature") val audioTemperature: Float,
        @SerialName("audio_top_p") val audioTopP: Float,
        @SerialName("audio_top_k") val audioTopK: Int,
        @SerialName("audio_repetition_penalty") val audioRepetitionPenalty: Float,
    )

    companion object {
        private val JSON_PARSER = Json { ignoreUnknownKeys = true }

        fun parse(json: String): ModelConfig =
            JSON_PARSER.decodeFromString(serializer(), json).also { it.rawJson = json }
    }
}
