package com.afun.mosstts.core.infer

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Schema of `audio_decoder_state_spec.json` — produced by the ONNX
 * exporter, consumed by [KvCache] and `InferenceLoop` so they know how
 * to allocate per-frame cache tensors and how to wire ORT inputs/outputs
 * for each step.
 *
 * The exporter writes more fields than we read here (`version`,
 * `model_path_*`, `batch_size`, `frames_per_call`); we ignore them via
 * `Json { ignoreUnknownKeys = true }` because the runtime contract is
 * defined by the names + dims, not by paths/versions.
 */
data class StateSpec(
    val numQuantizers: Int,
    val sampleRate: Int,
    val downsampleRate: Int,
    val inputNames: List<String>,
    val outputNames: List<String>,
    val attentionSpecs: List<AttentionSpec>,
    val transformerSpecs: List<TransformerSpec>,
) {
    data class AttentionSpec(
        val decoderModuleIndex: Int,
        val layerIndex: Int,
        val context: Int,
        val numHeads: Int,
        val headDim: Int,
    )

    data class TransformerSpec(
        val decoderModuleIndex: Int,
        val numLayers: Int,
    )

    companion object {
        private val JSON = Json { ignoreUnknownKeys = true }

        fun parse(json: String): StateSpec {
            val raw = JSON.decodeFromString(StateSpecJson.serializer(), json)
            return StateSpec(
                numQuantizers = raw.numQuantizers,
                sampleRate = raw.sampleRate,
                downsampleRate = raw.downsampleRate,
                inputNames = raw.inputNames,
                outputNames = raw.outputNames,
                attentionSpecs = raw.attentionSpecs.map {
                    AttentionSpec(
                        decoderModuleIndex = it.decoderModuleIndex,
                        layerIndex = it.layerIndex,
                        context = it.context,
                        numHeads = it.numHeads,
                        headDim = it.headDim,
                    )
                },
                transformerSpecs = raw.transformerSpecs.map {
                    TransformerSpec(
                        decoderModuleIndex = it.decoderModuleIndex,
                        numLayers = it.numLayers,
                    )
                },
            )
        }
    }
}

@Serializable
internal data class StateSpecJson(
    @SerialName("num_quantizers") val numQuantizers: Int,
    @SerialName("sample_rate") val sampleRate: Int,
    @SerialName("downsample_rate") val downsampleRate: Int,
    @SerialName("input_names") val inputNames: List<String>,
    @SerialName("output_names") val outputNames: List<String>,
    @SerialName("attention_specs") val attentionSpecs: List<AttnJson>,
    @SerialName("transformer_specs") val transformerSpecs: List<TxJson>,
) {
    @Serializable
    data class AttnJson(
        @SerialName("decoder_module_index") val decoderModuleIndex: Int,
        @SerialName("layer_index") val layerIndex: Int,
        val context: Int,
        @SerialName("num_heads") val numHeads: Int,
        @SerialName("head_dim") val headDim: Int,
    )

    @Serializable
    data class TxJson(
        @SerialName("decoder_module_index") val decoderModuleIndex: Int,
        @SerialName("num_layers") val numLayers: Int,
    )
}
