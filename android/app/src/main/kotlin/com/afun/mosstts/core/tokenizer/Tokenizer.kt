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
     * SP-style encoding: normalize the *entire* input once (this is what
     * Python sentencepiece does — the user-defined pieces are ASCII-printable
     * so they survive nfkc + whitespace-fold + collapse + trim + escape +
     * dummy-prefix unchanged), then split the normalized string by
     * USER_DEFINED specials. Each special is emitted verbatim as a single id;
     * each remaining text run is BPE-encoded with byte fallback.
     *
     * Doing the normalize *once* (as opposed to per-segment) is critical for
     * matching sp on inputs like "<|im_start|>\nabc" → `▁,<|im_start|>,▁ab,c`,
     * because the post-special "\n" only becomes the leading ▁ of "▁abc"
     * after sp's whole-input ws-fold + escape pipeline runs over both sides
     * of the special token.
     */
    fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val normalized = normalizer.normalize(text)
        if (normalized.isEmpty()) return IntArray(0)

        val out = ArrayList<Int>(normalized.length)
        var bufferStart = 0
        var i = 0
        while (i < normalized.length) {
            val matched = matchUserDefinedAt(normalized, i)
            if (matched != null) {
                if (i > bufferStart) {
                    for (id in encoder.encode(normalized.substring(bufferStart, i))) out.add(id)
                }
                out.add(pieceToId[matched]!!)
                i += matched.length
                bufferStart = i
            } else {
                i++
            }
        }
        if (bufferStart < normalized.length) {
            for (id in encoder.encode(normalized.substring(bufferStart))) out.add(id)
        }
        return out.toIntArray()
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
