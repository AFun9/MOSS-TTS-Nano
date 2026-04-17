package com.afun.mosstts.core.infer

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * Mirrors `onnx_tts_utils.py:sample_top_k_top_p` and
 * `apply_repetition_penalty`.
 *
 * Byte-equality with Python on the random path is impossible (numpy's
 * Mersenne Twister vs `kotlin.random.Random`'s xorshift), so we lock
 * down:
 *   - the deterministic paths (temp ≤ 0 or topK = 1 → argmax),
 *   - the repetition-penalty math (the *only* numerical contract the
 *     model relies on for stable codebook generation),
 *   - the topK / topP cutoff *set* (which logits survive — must match
 *     Python regardless of the final draw),
 *   - the empirical distribution under a seeded RNG (frequencies must
 *     approach the post-filter softmax distribution).
 */
class SamplerTest {

    private fun softmax(x: FloatArray): FloatArray {
        var m = Float.NEGATIVE_INFINITY
        for (v in x) if (v > m) m = v
        val out = FloatArray(x.size)
        var s = 0.0
        for (i in x.indices) {
            val e = exp((x[i] - m).toDouble())
            out[i] = e.toFloat()
            s += e
        }
        for (i in out.indices) out[i] = (out[i] / s).toFloat()
        return out
    }

    @Test
    fun `temperature lt or eq zero returns argmax`() {
        val logits = floatArrayOf(0.1f, 5.0f, -1.0f, 4.99f)
        val s = Sampler(temperature = 0f, topK = 0, topP = 1f, repetitionPenalty = 1f)
        assertThat(s.sample(logits)).isEqualTo(1)
    }

    @Test
    fun `topK = 1 collapses to argmax regardless of temperature`() {
        val logits = floatArrayOf(0.1f, 5.0f, -1.0f, 4.99f)
        val s = Sampler(temperature = 1f, topK = 1, topP = 1f, repetitionPenalty = 1f)
        repeat(20) { assertThat(s.sample(logits)).isEqualTo(1) }
    }

    @Test
    fun `repetition penalty 1 leaves logits untouched`() {
        val logits = floatArrayOf(1f, -1f, 2f, -2f)
        val s = Sampler()
        val prev = intArrayOf(0, 2)
        val penalised = s.applyRepetitionPenalty(logits, prev, 1f)
        assertThat(penalised).isEqualTo(logits)
    }

    @Test
    fun `repetition penalty divides positive logits and multiplies negative ones`() {
        // Python: scores[id] *= p when scores[id] < 0 else /= p
        val logits = floatArrayOf(2f, -3f, 0f, 5f)
        val s = Sampler()
        val out = s.applyRepetitionPenalty(logits, intArrayOf(0, 1, 3, 3), repetitionPenalty = 2f)
        // id 0: positive 2 → 1.0
        // id 1: negative -3 → -6.0
        // id 2: not in prev → 0
        // id 3: positive 5 → 2.5 (dedup applied — Python uses np.unique)
        assertThat(out.toList()).isEqualTo(listOf(1f, -6f, 0f, 2.5f))
    }

    @Test
    fun `repetition penalty ignores out-of-range previous ids`() {
        val logits = floatArrayOf(2f, -3f, 0f)
        val s = Sampler()
        val out = s.applyRepetitionPenalty(logits, intArrayOf(-1, 0, 7, 99), repetitionPenalty = 4f)
        assertThat(out.toList()).isEqualTo(listOf(0.5f, -3f, 0f))
    }

    @Test
    fun `topK keeps only the k highest scoring ids in the support set`() {
        val logits = floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f)
        val s = Sampler(temperature = 1f, topK = 2, topP = 1f, repetitionPenalty = 1f, rng = Random(42))
        // Only ids 4 and 5 should ever be drawn.
        val seen = HashSet<Int>()
        repeat(200) { seen.add(s.sample(logits)) }
        assertThat(seen).containsExactly(4, 5)
    }

    @Test
    fun `topP cutoff matches the inclusive Python rule`() {
        // softmax of [0,0,0,5,5] ≈ [tiny, tiny, tiny, 0.4998, 0.4998]
        // cumsum sorted = [0.4998, 0.9997, 1.0, 1.0, 1.0]
        // top_p = 0.6 → searchsorted = 1 → cutoff = 2 → keep 2 ids (3,4).
        val logits = floatArrayOf(0f, 0f, 0f, 5f, 5f)
        val s = Sampler(temperature = 1f, topK = 0, topP = 0.6f, repetitionPenalty = 1f, rng = Random(123))
        val seen = HashSet<Int>()
        repeat(400) { seen.add(s.sample(logits)) }
        assertThat(seen).containsExactly(3, 4)
    }

    @Test
    fun `topP equal to 1 disables the filter`() {
        // Without filter, every id has nonzero probability under softmax.
        val logits = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        val s = Sampler(temperature = 1f, topK = 0, topP = 1f, repetitionPenalty = 1f, rng = Random(7))
        val seen = HashSet<Int>()
        repeat(400) { seen.add(s.sample(logits)) }
        assertThat(seen).containsExactly(0, 1, 2, 3)
    }

    @Test
    fun `empirical frequencies converge to the softmax distribution`() {
        // Locked-down RNG → reproducible CI: a temperature-1, no-filter
        // softmax over [ln 1, ln 2, ln 4] = [0, 0.693, 1.386] should give
        // probabilities [1/7, 2/7, 4/7] ≈ [0.143, 0.286, 0.571].
        val logits = floatArrayOf(0f, ln(2f), ln(4f))
        val s = Sampler(temperature = 1f, rng = Random(1234567L))
        val counts = IntArray(3)
        val n = 20_000
        repeat(n) { counts[s.sample(logits)]++ }
        val freq = counts.map { it.toDouble() / n }
        val target = softmax(logits).map { it.toDouble() }
        for (i in target.indices) {
            assertThat(abs(freq[i] - target[i])).isLessThan(0.015)
        }
    }
}
