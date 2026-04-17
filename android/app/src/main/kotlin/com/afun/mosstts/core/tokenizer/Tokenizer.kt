package com.afun.mosstts.core.tokenizer

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Pure-Kotlin SentencePiece tokenizer aligned with `onnx_export/tokenizer.model`.
 *
 * Build it once at app start from the bundled `assets/tokenizer_kotlin.json`
 * (produced by `tools/export_tokenizer_for_kotlin.py`) and reuse the same
 * instance for every prompt. Thread-safe: nothing mutable after construction.
 *
 * Correctness against the Python reference is verified by
 * `TokenizerGoldenTest` over 64 multi-language fixtures; any divergence
 * fails the build.
 */
class Tokenizer private constructor(
    private val normalizer: SpNormalizer,
    private val encoder: BpeEncoder,
    val vocabSize: Int,
    val unkId: Int,
    val bosId: Int,
    val eosId: Int,
    val padId: Int,
    private val pieceToId: Map<String, Int>,
    /** USER_DEFINED pieces (type=4) sorted longest-first for greedy literal matching. */
    private val userDefinedPieces: List<String>,
) {
    /**
     * SP-style encoding: split by USER_DEFINED specials first (each emitted
     * verbatim as a single id, no normalization), then for every remaining
     * text segment run normalize + BPE + byte_fallback. The dummy '▁' prefix
     * is added only to the very first segment, matching Python sentencepiece.
     */
    fun encode(text: String): IntArray {
        if (text.isEmpty()) {
            return encoder.encode(normalizer.normalize(text))
        }

        val out = ArrayList<Int>(text.length)
        var firstSegment = true
        var bufferStart = 0
        var i = 0
        while (i < text.length) {
            val matched = matchUserDefinedAt(text, i)
            if (matched != null) {
                emitSegment(text, bufferStart, i, firstSegment, out)
                firstSegment = false
                out.add(pieceToId[matched]!!)
                i += matched.length
                bufferStart = i
            } else {
                i++
            }
        }
        // Trailing tail (or the entire text if no specials matched).
        if (bufferStart < text.length || firstSegment) {
            emitSegment(text, bufferStart, text.length, firstSegment, out)
        }
        return out.toIntArray()
    }

    private fun emitSegment(
        text: String,
        start: Int,
        end: Int,
        firstSegment: Boolean,
        out: MutableList<Int>,
    ) {
        val chunk = text.substring(start, end)
        if (chunk.isEmpty()) {
            // Python sp emits a lone '▁' when the very first segment of a
            // non-empty input is empty (e.g. text starts with a USER_DEFINED
            // special token like "<|im_start|>...").
            if (firstSegment && text.isNotEmpty()) {
                pieceToId[SpNormalizer.WHITESPACE_BLOCK.toString()]?.let(out::add)
            }
            return
        }
        val normalized = normalizer.normalize(chunk, addDummyPrefix = firstSegment)
        if (normalized.isEmpty()) return
        for (id in encoder.encode(normalized)) out.add(id)
    }

    private fun matchUserDefinedAt(text: String, pos: Int): String? {
        // userDefinedPieces is sorted longest-first so we always pick the
        // greediest match (e.g. "<|im_start|>" before "<|im").
        for (sp in userDefinedPieces) {
            if (sp.length <= text.length - pos &&
                text.regionMatches(pos, sp, 0, sp.length)
            ) {
                return sp
            }
        }
        return null
    }

    /** Look up the id of a special token by its literal string ("<|im_start|>" etc.). */
    fun specialTokenId(piece: String): Int? = pieceToId[piece]

    companion object {
        private val JSON_PARSER = Json { ignoreUnknownKeys = true }

        fun load(json: String): Tokenizer {
            val tk = JSON_PARSER.decodeFromString(TokenizerJson.serializer(), json)
            val n = tk.pieces.size
            require(n == tk.vocabSize) {
                "vocab_size=${tk.vocabSize} but pieces.size=$n"
            }
            val pieceToId = HashMap<String, Int>(n)
            val scores = FloatArray(n)
            val userDefined = ArrayList<String>()
            for ((id, p) in tk.pieces.withIndex()) {
                // SP guarantees unique piece strings; keep first if duplicates ever appear.
                pieceToId.putIfAbsent(p.piece, id)
                scores[id] = p.score
                // Only USER_DEFINED (type=4) pieces are matched verbatim in user
                // text. CONTROL (type=3, e.g. "<s>") and UNKNOWN (type=2) are
                // intentionally NOT recognized inside input text — Python sp
                // would tokenize a literal "<s>" via byte_fallback, not as bos.
                if (p.type == 4 && p.piece.isNotEmpty()) userDefined.add(p.piece)
            }
            // Longest-first so e.g. "<|im_start|>" wins over a hypothetical "<|im".
            userDefined.sortByDescending { it.length }

            val normalizer = SpNormalizer(
                name = tk.normalizer.name,
                addDummyPrefix = tk.normalizer.addDummyPrefix,
                removeExtraWhitespaces = tk.normalizer.removeExtraWhitespaces,
                escapeWhitespaces = tk.normalizer.escapeWhitespaces,
            )
            val encoder = BpeEncoder(
                vocab = pieceToId,
                scores = scores,
                byteFallbackStart = tk.byteFallbackStart,
                unkId = tk.unkId,
            )
            return Tokenizer(
                normalizer = normalizer,
                encoder = encoder,
                vocabSize = tk.vocabSize,
                unkId = tk.unkId,
                bosId = tk.bosId,
                eosId = tk.eosId,
                padId = tk.padId,
                pieceToId = pieceToId,
                userDefinedPieces = userDefined,
            )
        }
    }
}

@Serializable
internal data class TokenizerJson(
    @SerialName("vocab_size") val vocabSize: Int,
    @SerialName("unk_id") val unkId: Int,
    @SerialName("bos_id") val bosId: Int,
    @SerialName("eos_id") val eosId: Int,
    @SerialName("pad_id") val padId: Int,
    @SerialName("byte_fallback_start") val byteFallbackStart: Int,
    val normalizer: NormalizerSpec,
    val pieces: List<Piece>,
) {
    @Serializable
    data class NormalizerSpec(
        val name: String,
        @SerialName("add_dummy_prefix") val addDummyPrefix: Boolean,
        @SerialName("remove_extra_whitespaces") val removeExtraWhitespaces: Boolean,
        @SerialName("escape_whitespaces") val escapeWhitespaces: Boolean,
    )

    @Serializable
    data class Piece(
        val piece: String,
        val score: Float,
        val type: Int = 1,
    )
}
