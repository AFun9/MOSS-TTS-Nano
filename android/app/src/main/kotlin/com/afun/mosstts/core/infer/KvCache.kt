package com.afun.mosstts.core.infer

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Per-utterance KV cache for the audio_decoder ORT session.
 *
 * Mirrors `onnx_infer.py:AudioDecoder._build_zero_state`:
 *
 * ```
 * for each transformer (di):
 *   decoder_{di}_transformer_offset            int64 [1]      = 0
 * for each attention layer (di, li):
 *   decoder_{di}_layer_{li}_cached_keys        f32   [1, nh, ctx, hd] = 0
 *   decoder_{di}_layer_{li}_cached_values      f32   [1, nh, ctx, hd] = 0
 *   decoder_{di}_layer_{li}_cached_positions   int64 [1, ctx]         = -1
 *   decoder_{di}_layer_{li}_offset             int64 [1]              = 0
 * ```
 *
 * The cache is a single contiguous `ByteBuffer` per tensor (direct +
 * little-endian) so M2.5 can pass the same memory straight to
 * `OnnxTensor.createTensor(env, …)` with zero copies on the hot path.
 *
 * `reset()` rewrites the contents in-place — no reallocation, so the
 * `OnnxTensor` references the inference loop hands to ORT stay valid.
 */
class KvCache(private val spec: StateSpec) {
    enum class DType { FLOAT32, INT64 }

    /**
     * One slot in the cache. `data` is a typed view (`FloatBuffer` or
     * `LongBuffer`) over the underlying direct `ByteBuffer`. Tests and
     * the inference loop write through the typed view; ORT consumes
     * the raw `ByteBuffer`.
     */
    class Tensor(
        val name: String,
        val dtype: DType,
        val shape: LongArray,
        /** Direct, little-endian. Capacity = `shape.product * dtype.size`. */
        val backing: ByteBuffer,
    ) {
        val data: java.nio.Buffer = when (dtype) {
            DType.FLOAT32 -> backing.asFloatBuffer()
            DType.INT64 -> backing.asLongBuffer()
        }
    }

    private val byName: LinkedHashMap<String, Tensor> = LinkedHashMap()

    /** Iteration order matches `inputNames` (minus the leading `codes`). */
    val tensorNames: List<String> get() = byName.keys.toList()

    init {
        // Allocate in the exact order Python enumerates: outer loop over
        // transformer_specs (gives us `_transformer_offset`), inner loop
        // over each transformer's attention layers in attention_specs
        // order. This must match StateSpec.input_names ordering or the
        // engine assertion test fails.
        val grouped = spec.attentionSpecs.groupBy { it.decoderModuleIndex }
        for (ts in spec.transformerSpecs) {
            val di = ts.decoderModuleIndex
            allocLong("decoder_${di}_transformer_offset", longArrayOf(1L), fill = 0L)
            for (a in grouped[di].orEmpty()) {
                val p = "decoder_${a.decoderModuleIndex}_layer_${a.layerIndex}"
                val nh = a.numHeads.toLong()
                val ctx = a.context.toLong()
                val hd = a.headDim.toLong()
                allocFloat("${p}_cached_keys", longArrayOf(1L, nh, ctx, hd), fill = 0f)
                allocFloat("${p}_cached_values", longArrayOf(1L, nh, ctx, hd), fill = 0f)
                allocLong("${p}_cached_positions", longArrayOf(1L, ctx), fill = -1L)
                allocLong("${p}_offset", longArrayOf(1L), fill = 0L)
            }
        }
    }

    fun get(name: String): Tensor =
        byName[name] ?: error("KvCache has no tensor named '$name'")

    /**
     * Wipe every tensor back to its initial state (same values
     * `_build_zero_state` writes in Python). Buffers are reused.
     */
    fun reset() {
        for ((_, t) in byName) {
            when (t.dtype) {
                DType.FLOAT32 -> {
                    val buf = t.data as FloatBuffer
                    buf.rewind()
                    while (buf.hasRemaining()) buf.put(0f)
                }
                DType.INT64 -> {
                    val buf = t.data as LongBuffer
                    val fill = if (t.name.endsWith("_cached_positions")) -1L else 0L
                    buf.rewind()
                    while (buf.hasRemaining()) buf.put(fill)
                }
            }
        }
    }

    private fun allocFloat(name: String, shape: LongArray, fill: Float) {
        val n = shape.product()
        val bb = ByteBuffer.allocateDirect((n * 4).toInt()).order(ByteOrder.nativeOrder())
        val t = Tensor(name, DType.FLOAT32, shape, bb)
        if (fill != 0f) {
            val fb = t.data as FloatBuffer
            for (i in 0 until n) fb.put(fill)
        }
        byName[name] = t
    }

    private fun allocLong(name: String, shape: LongArray, fill: Long) {
        val n = shape.product()
        val bb = ByteBuffer.allocateDirect((n * 8).toInt()).order(ByteOrder.nativeOrder())
        val t = Tensor(name, DType.INT64, shape, bb)
        if (fill != 0L) {
            val lb = t.data as LongBuffer
            for (i in 0 until n) lb.put(fill)
        }
        byName[name] = t
    }
}

private fun LongArray.product(): Long {
    var p = 1L
    for (v in this) p *= v
    return p
}
