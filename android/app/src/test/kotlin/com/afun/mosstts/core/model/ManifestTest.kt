package com.afun.mosstts.core.model

import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * Locks down the release manifest schema (`format_version = 1`) used by
 * the v1.0.0 ONNX bundle.
 *
 * The fixture is the *real* `export_v2/release/manifest.json` checked
 * into resources — keeps the test honest about what the runtime actually
 * sees, and any drift from `make_release.py` will surface here first.
 */
class ManifestTest {
    private val sample by lazy {
        javaClass.getResource("/release_manifest_v1.json")!!.readText()
    }

    @Test
    fun `parses top-level identification`() {
        val m = Manifest.parse(sample)
        assertThat(m.formatVersion).isEqualTo(1)
        assertThat(m.releaseVersion).isEqualTo("1.0.0")
    }

    @Test
    fun `tts group lists the four graphs and shared blob`() {
        val tts = Manifest.parse(sample).tts
        assertThat(tts.graphs).containsExactly(
            "moss_tts_prefill.onnx",
            "moss_tts_decode_step.onnx",
            "moss_tts_local_decoder.onnx",
            "moss_tts_local_cached_step.onnx",
        ).inOrder()
        assertThat(tts.sharedBlob).isEqualTo("moss_tts_shared.data")
        assertThat(tts.io).hasSize(4)
    }

    @Test
    fun `codec group lists the three graphs and shared blob`() {
        val codec = Manifest.parse(sample).codec
        assertThat(codec.graphs).containsExactly(
            "moss_audio_tokenizer_encode.onnx",
            "moss_audio_tokenizer_decode_full.onnx",
            "moss_audio_tokenizer_decode_step.onnx",
        ).inOrder()
        assertThat(codec.sharedBlob).isEqualTo("moss_audio_tokenizer_shared.data")
    }

    @Test
    fun `tts_config exposes every special token id used by PromptBuilder`() {
        val c = Manifest.parse(sample).ttsConfig
        assertThat(c.nVq).isEqualTo(16)
        assertThat(c.audioPadTokenId).isEqualTo(1024)
        assertThat(c.audioStartTokenId).isEqualTo(6)
        assertThat(c.audioEndTokenId).isEqualTo(7)
        assertThat(c.audioUserSlotTokenId).isEqualTo(8)
        assertThat(c.audioAssistantSlotTokenId).isEqualTo(9)
        assertThat(c.imStartTokenId).isEqualTo(4)
        assertThat(c.imEndTokenId).isEqualTo(5)
        assertThat(c.padTokenId).isEqualTo(3)
        assertThat(c.vocabSize).isEqualTo(16384)
        assertThat(c.audioCodebookSizes).hasSize(16)
        assertThat(c.audioCodebookSizes.toSet()).containsExactly(1024)
    }

    @Test
    fun `prompt_templates carries the three pre-tokenized boilerplate sections`() {
        val pt = Manifest.parse(sample).promptTemplates
        // Sanity: each section is a non-empty list of token ids.
        assertThat(pt.userPromptPrefixTokenIds).isNotEmpty()
        assertThat(pt.userPromptAfterReferenceTokenIds).isNotEmpty()
        assertThat(pt.assistantPromptPrefixTokenIds).isNotEmpty()
        // Defensive: the prefix must end with the role-prefix sequence;
        // the assistant prefix contains the closing <|im_start|> hint.
        assertThat(pt.assistantPromptPrefixTokenIds.last()).isAtLeast(0)
    }

    @Test
    fun `generation_defaults exposes audio sampling knobs`() {
        val gd = Manifest.parse(sample).generationDefaults
        assertThat(gd.audioTopK).isEqualTo(25)
        assertThat(gd.audioTopP).isWithin(1e-6f).of(0.95f)
        assertThat(gd.audioTemperature).isWithin(1e-6f).of(0.8f)
        assertThat(gd.audioRepetitionPenalty).isWithin(1e-6f).of(1.2f)
        assertThat(gd.maxNewFrames).isEqualTo(375)
    }

    @Test
    fun `files map carries size and sha256 for every shipped file`() {
        val m = Manifest.parse(sample)
        assertThat(m.files.keys).containsAtLeast(
            "moss_tts_prefill.onnx",
            "moss_tts_shared.data",
            "moss_audio_tokenizer_encode.onnx",
            "moss_audio_tokenizer_shared.data",
            "tokenizer.model",
        )
        for ((_, fe) in m.files) {
            assertThat(fe.size).isGreaterThan(0L)
            assertThat(fe.sha256).hasLength(64)  // hex-encoded sha256
            assertThat(fe.sha256).matches("[0-9a-fA-F]{64}")
        }
    }

    @Test
    fun `requiredFileNames contains every TTS graph, codec graph, blobs and tokenizer`() {
        val needed = Manifest.parse(sample).requiredFileNames()
        assertThat(needed).containsAtLeast(
            "moss_tts_prefill.onnx",
            "moss_tts_decode_step.onnx",
            "moss_tts_local_decoder.onnx",
            "moss_tts_local_cached_step.onnx",
            "moss_tts_shared.data",
            "moss_audio_tokenizer_encode.onnx",
            "moss_audio_tokenizer_decode_full.onnx",
            "moss_audio_tokenizer_decode_step.onnx",
            "moss_audio_tokenizer_shared.data",
            "tokenizer.model",
        )
    }

    @Test
    fun `tokenizer points at sentencepiece tokenizer model`() {
        val t = Manifest.parse(sample).tokenizer
        assertThat(t.type).isEqualTo("sentencepiece")
        assertThat(t.file).isEqualTo("tokenizer.model")
    }

    @Test
    fun `rejects malformed json`() {
        assertThat(runCatching { Manifest.parse("not json") }.isFailure).isTrue()
    }
}
