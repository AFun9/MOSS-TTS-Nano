package com.afun.mosstts.core.tokenizer

/**
 * SentencePiece BPE encoder with byte_fallback.
 *
 * Mirrors the algorithm in `sentencepiece/src/bpe_model.cc`:
 *
 * 1. Split [text] into a list of single-codepoint pieces (codepoints, not
 *    `Char`s, so astral-plane chars stay intact).
 * 2. Repeatedly find the adjacent pair whose merged form exists in the
 *    vocab with the **highest score** and merge it. SentencePiece stores
 *    larger (= more frequent) BPE merges with higher (less negative) scores.
 * 3. Stop when no adjacent pair is mergeable.
 * 4. Look every remaining piece up in the vocab. Pieces still missing are
 *    expanded to their UTF-8 bytes through the `byte_fallback` table.
 *
 * Complexity is O(N²) where N is the codepoint count; fine for the prompts
 * we send (≤ a few hundred codepoints). Can be replaced with a heap-based
 * O(N log N) variant in M6 if profiling demands.
 */
class BpeEncoder(
    private val vocab: Map<String, Int>,
    private val scores: FloatArray,
    private val byteFallbackStart: Int,
    private val unkId: Int,
) {
    fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)

        val pieces = ArrayList<String>(text.length)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            pieces.add(String(intArrayOf(cp), 0, 1))
            i += Character.charCount(cp)
        }

        // Greedy max-score merging.
        while (pieces.size >= 2) {
            var bestIdx = -1
            var bestScore = Float.NEGATIVE_INFINITY
            for (j in 0 until pieces.size - 1) {
                val merged = pieces[j] + pieces[j + 1]
                val id = vocab[merged] ?: continue
                val s = scores[id]
                if (s > bestScore) {
                    bestScore = s
                    bestIdx = j
                }
            }
            if (bestIdx < 0) break
            pieces[bestIdx] = pieces[bestIdx] + pieces[bestIdx + 1]
            pieces.removeAt(bestIdx + 1)
        }

        // Translate to ids; byte_fallback for OOV pieces.
        val out = ArrayList<Int>(pieces.size)
        for (p in pieces) {
            val id = vocab[p]
            if (id != null) {
                out.add(id)
            } else {
                // UTF-8 byte expansion. byte_fallback_start indexes into
                // the contiguous 256-entry <0xXX> table.
                val bytes = p.toByteArray(Charsets.UTF_8)
                for (b in bytes) {
                    out.add(byteFallbackStart + (b.toInt() and 0xFF))
                }
            }
        }
        return out.toIntArray()
    }
}
