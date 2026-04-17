package com.afun.mosstts.core.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Mirror of `onnx_export/manifest.json` (schema `moss-tts-nano-onnx-bundle/v2`).
 *
 * Only the fields the Android runtime actually consumes are surfaced; the
 * `size_mb` field in the source is intentionally ignored (redundant with
 * `size_bytes`) and unknown future keys are tolerated thanks to
 * [JSON_PARSER]'s `ignoreUnknownKeys = true`.
 */
@Serializable
data class Manifest(
    val schema: String,
    val quantization: Quantization,
    val files: List<FileEntry>,
) {
    @Serializable
    data class Quantization(
        @SerialName("weight_type") val weightType: String,
        @SerialName("per_channel") val perChannel: Boolean,
        @SerialName("reduce_range") val reduceRange: Boolean,
    )

    @Serializable
    data class FileEntry(
        val name: String,
        @SerialName("size_bytes") val sizeBytes: Long,
    )

    fun fileSizeOf(name: String): Long? =
        files.firstOrNull { it.name == name }?.sizeBytes

    companion object {
        private val JSON_PARSER = Json { ignoreUnknownKeys = true }

        /** @throws kotlinx.serialization.SerializationException on malformed input. */
        fun parse(json: String): Manifest = JSON_PARSER.decodeFromString(serializer(), json)
    }
}
