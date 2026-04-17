package com.afun.mosstts.core.infer

import kotlin.math.exp
import kotlin.random.Random

/**
 * Token sampler — Kotlin port of `onnx_tts_utils.py:sample_top_k_top_p`
 * and `apply_repetition_penalty`.
 *
 * Numerical contracts (verified by `SamplerTest`):
 *   - `temperature ≤ 0` ⇒ pure argmax (Python early-returns the same way).
 *   - `topK > 0` keeps only the K highest scores by value (the rest get
 *     `-Inf`); ties at the K-th rank may all be kept exactly like
 *     numpy's `np.partition`.
 *   - `topP ∈ (0, 1)` keeps the smallest prefix whose softmax cumsum
 *     reaches `topP`, plus one (Python's `searchsorted + 1` rule).
 *   - `repetitionPenalty != 1.0` rescales every previously-emitted id:
 *     positive logits divided by p, negative ones multiplied by p
 *     (matches HF's GPT-2 style penalty, mirrored in Python here).
 *
 * Byte-equality with Python on the random draw is *not* a goal — numpy's
 * Mersenne Twister and `kotlin.random.Random` (xorshift64*) emit
 * different streams. The test suite locks down the deterministic paths
 * + the surviving support-set + the empirical distribution, which is
 * all that matters for audio quality.
 *
 * Stateless w.r.t. logits: every `sample` allocates a fresh scores
 * buffer so the caller can reuse the same FloatArray across codebooks.
 */
class Sampler(
    val temperature: Float = 1.0f,
    val topK: Int = 0,
    val topP: Float = 1.0f,
    val repetitionPenalty: Float = 1.0f,
    private val rng: Random = Random.Default,
) {
    fun sample(logits: FloatArray, previousIds: IntArray? = null): Int {
        val penalised = if (previousIds != null) {
            applyRepetitionPenalty(logits, previousIds, repetitionPenalty)
        } else {
            logits
        }
        if (temperature <= 0f) return argmax(penalised)

        val scores = FloatArray(penalised.size) { penalised[it] / temperature }

        if (topK > 0) applyTopK(scores, topK)
        if (topP > 0f && topP < 1f) applyTopP(scores, topP)

        val probs = softmax(scores)
        return drawCategorical(probs)
    }

    /**
     * Public so callers (and tests) can apply repetition penalty
     * once and reuse the result across multiple [sample] calls if
     * they want — matches the Python helper's standalone signature.
     */
    fun applyRepetitionPenalty(
        logits: FloatArray,
        previousIds: IntArray,
        repetitionPenalty: Float = this.repetitionPenalty,
    ): FloatArray {
        require(repetitionPenalty > 0f) { "repetitionPenalty must be positive, got $repetitionPenalty" }
        if (repetitionPenalty == 1.0f || previousIds.isEmpty()) return logits
        val out = logits.copyOf()
        // Python uses np.unique → dedup so a token appearing N times still
        // gets penalised exactly once. Mirror that with a HashSet.
        val seen = HashSet<Int>(previousIds.size)
        for (id in previousIds) {
            if (id < 0 || id >= out.size) continue
            if (!seen.add(id)) continue
            if (out[id] < 0f) {
                out[id] = out[id] * repetitionPenalty
            } else {
                out[id] = out[id] / repetitionPenalty
            }
        }
        return out
    }

    private fun argmax(x: FloatArray): Int {
        var best = 0
        var bestVal = x[0]
        for (i in 1 until x.size) {
            if (x[i] > bestVal) {
                bestVal = x[i]
                best = i
            }
        }
        return best
    }

    private fun applyTopK(scores: FloatArray, k: Int) {
        val effectiveK = minOf(k, scores.size)
        if (effectiveK == scores.size) return
        // Match `np.partition(scores, -k)[-k]`: find the k-th largest value
        // (1-indexed). Anything strictly less than that becomes -Inf; ties at
        // the threshold are *kept* (consistent with numpy's behavior).
        val sortedDesc = scores.copyOf().also { it.sort() }
        val threshold = sortedDesc[scores.size - effectiveK]
        for (i in scores.indices) {
            if (scores[i] < threshold) scores[i] = Float.NEGATIVE_INFINITY
        }
    }

    private fun applyTopP(scores: FloatArray, topP: Float) {
        // Indices sorted by descending score (Python uses argsort()[::-1]).
        val n = scores.size
        val indices = IntArray(n) { it }
        // Selection by descending score; stable order isn't required for the
        // cumsum cutoff rule but Java's Integer sort is stable, so we wrap.
        val boxed = indices.toTypedArray()
        boxed.sortByDescending { scores[it] }
        val sortedScores = FloatArray(n) { scores[boxed[it]] }
        val sortedProbs = softmax(sortedScores)

        var cumsum = 0.0
        var cutoff = n
        for (i in 0 until n) {
            cumsum += sortedProbs[i].toDouble()
            if (cumsum >= topP) {
                // Python: `searchsorted(cumsum, top_p) + 1` → keep up to and
                // including the position whose cumsum first reaches top_p.
                cutoff = i + 1
                break
            }
        }
        // Mask everything outside the cutoff in the *original* index space.
        val keep = BooleanArray(n)
        for (i in 0 until cutoff) keep[boxed[i]] = true
        for (i in 0 until n) {
            if (!keep[i]) scores[i] = Float.NEGATIVE_INFINITY
        }
    }

    private fun softmax(x: FloatArray): FloatArray {
        var max = Float.NEGATIVE_INFINITY
        for (v in x) if (v > max) max = v
        val out = FloatArray(x.size)
        var sum = 0.0
        for (i in x.indices) {
            val e = if (x[i] == Float.NEGATIVE_INFINITY) 0.0 else exp((x[i] - max).toDouble())
            out[i] = e.toFloat()
            sum += e
        }
        // Defensive: if every entry was -Inf (shouldn't happen — top-K
        // always keeps at least 1) fall back to uniform over the array.
        if (sum == 0.0) {
            val u = 1.0f / x.size
            for (i in out.indices) out[i] = u
            return out
        }
        for (i in out.indices) out[i] = (out[i] / sum).toFloat()
        return out
    }

    private fun drawCategorical(probs: FloatArray): Int {
        val u = rng.nextDouble()
        var acc = 0.0
        for (i in probs.indices) {
            acc += probs[i].toDouble()
            if (u < acc) return i
        }
        // Numerical fallback: rounding errors can leave u slightly above the
        // final cumsum. Return the last id whose probability was nonzero.
        for (i in probs.indices.reversed()) {
            if (probs[i] > 0f) return i
        }
        return probs.size - 1
    }
}
