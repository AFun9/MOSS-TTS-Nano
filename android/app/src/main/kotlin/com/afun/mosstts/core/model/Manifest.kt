package com.afun.mosstts.core.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement

/**
 * Mirror of `export_v2/release/manifest.json` (format_version = 1).
 *
 * The release bundle inlines what used to be a separate `config.json` plus
 * `audio_decoder_state_spec.json` plus a hard-coded set of prompt template
 * strings, so the runtime now only needs to consume this single artifact:
 *
 *   - [ttsConfig]: every special-token id and the n_vq / vocab_size dims
 *     used by `PromptBuilder` and `InferenceLoop`. Source of truth for the
 *     official voice_clone protocol.
 *   - [promptTemplates]: pre-tokenized id arrays for the three boilerplate
 *     sections of the prompt. Replaces the previous "encode template
 *     strings via tokenizer" path, which was a known source of off-by-one
 *     drift between Python and Kotlin.
 *   - [generationDefaults]: audio temp / top-k / top-p / repetition penalty
 *     used as defaults when the caller doesn't override.
 *   - [tts] / [codec]: the 4 TTS graph names + shared INT8 weight blob, and
 *     the 3 codec graph names + their shared blob. Used by the loader to
 *     build ORT sessions and validate file presence.
 *   - [files]: every shipped file with size_bytes + sha256 (the manifest
 *     is **self-validating** — no separate sha256 sidecar needed).
 *   - [tokenizer]: SentencePiece file pointer (currently always
 *     `tokenizer.model`, but the runtime reads the relative path here so
 *     a future swap doesn't need code changes).
 *
 * Unknown future keys (e.g. `share_report`, `total_bytes`, `generated_at`)
 * are tolerated thanks to `ignoreUnknownKeys = true`.
 */
@Serializable
data class Manifest(
    @SerialName("format_version") val formatVersion: Int,
    @SerialName("release_version") val releaseVersion: String,
    val tts: GraphGroup,
    val codec: GraphGroup,
    @SerialName("tts_config") val ttsConfig: TtsConfig,
    @SerialName("prompt_templates") val promptTemplates: PromptTemplates,
    @SerialName("generation_defaults") val generationDefaults: GenerationDefaults,
    val tokenizer: TokenizerSpec,
    val files: Map<String, FileEntry>,
) {
    /** A group of ONNX graphs that share one external-data weight blob. */
    @Serializable
    data class GraphGroup(
        val graphs: List<String>,
        @SerialName("shared_blob") val sharedBlob: String,
        /**
         * Each graph -> list of `.data` files it references. With our shared
         * blob layout this is always `[shared_blob]` for every graph, but
         * we read it raw so a future per-graph blob layout still parses.
         */
        @SerialName("external_data_files")
        val externalDataFiles: Map<String, List<String>> = emptyMap(),
        /**
         * Per-graph IO names + shapes. `JsonElement` because shape entries
         * mix int dims (e.g. `768`) with symbolic dims (e.g. `"prefill_seq"`).
         * The runtime doesn't drive logic from this — it's used at session
         * creation time only as a sanity check + for diagnostic logging.
         */
        val io: Map<String, JsonElement> = emptyMap(),
    )

    /** Token ids + dims that drive PromptBuilder + InferenceLoop. */
    @Serializable
    data class TtsConfig(
        @SerialName("n_vq") val nVq: Int,
        @SerialName("audio_pad_token_id") val audioPadTokenId: Int,
        @SerialName("pad_token_id") val padTokenId: Int,
        @SerialName("im_start_token_id") val imStartTokenId: Int,
        @SerialName("im_end_token_id") val imEndTokenId: Int,
        @SerialName("audio_start_token_id") val audioStartTokenId: Int,
        @SerialName("audio_end_token_id") val audioEndTokenId: Int,
        @SerialName("audio_user_slot_token_id") val audioUserSlotTokenId: Int,
        @SerialName("audio_assistant_slot_token_id") val audioAssistantSlotTokenId: Int,
        @SerialName("audio_codebook_sizes") val audioCodebookSizes: List<Int>,
        @SerialName("vocab_size") val vocabSize: Int,
    )

    /** Pre-tokenized prompt boilerplate (matches `demo_generate.py` exactly). */
    @Serializable
    data class PromptTemplates(
        @SerialName("user_prompt_prefix_token_ids")
        val userPromptPrefixTokenIds: List<Int>,
        @SerialName("user_prompt_after_reference_token_ids")
        val userPromptAfterReferenceTokenIds: List<Int>,
        @SerialName("assistant_prompt_prefix_token_ids")
        val assistantPromptPrefixTokenIds: List<Int>,
    )

    /** Defaults used when a generate() call doesn't override. */
    @Serializable
    data class GenerationDefaults(
        @SerialName("max_new_frames") val maxNewFrames: Int = 375,
        @SerialName("do_sample") val doSample: Boolean = true,
        @SerialName("text_temperature") val textTemperature: Float = 1.0f,
        @SerialName("text_top_p") val textTopP: Float = 1.0f,
        @SerialName("text_top_k") val textTopK: Int = 50,
        @SerialName("audio_temperature") val audioTemperature: Float,
        @SerialName("audio_top_p") val audioTopP: Float,
        @SerialName("audio_top_k") val audioTopK: Int,
        @SerialName("audio_repetition_penalty") val audioRepetitionPenalty: Float = 1.0f,
        @SerialName("sample_mode") val sampleMode: String = "fixed",
    )

    @Serializable
    data class TokenizerSpec(
        val type: String,
        val file: String,
    )

    @Serializable
    data class FileEntry(
        val size: Long,
        val sha256: String,
    )

    /** Convenience: `[tts.graphs] ++ [tts.shared_blob] ++ [codec.graphs] ++ [codec.shared_blob] ++ [tokenizer.file]`. */
    fun requiredFileNames(): List<String> = buildList {
        addAll(tts.graphs)
        add(tts.sharedBlob)
        addAll(codec.graphs)
        add(codec.sharedBlob)
        add(tokenizer.file)
    }

    /** Lookup helper used by `ModelManager` validation. */
    fun fileEntryOf(name: String): FileEntry? = files[name]

    companion object {
        private val JSON_PARSER = Json { ignoreUnknownKeys = true }

        /** @throws kotlinx.serialization.SerializationException on malformed input. */
        fun parse(json: String): Manifest = JSON_PARSER.decodeFromString(serializer(), json)
    }
}
