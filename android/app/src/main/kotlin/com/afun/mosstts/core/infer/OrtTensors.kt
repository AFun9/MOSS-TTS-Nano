package com.afun.mosstts.core.infer

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer

private const val EMPTY_BUF_CAPACITY = 256  // bytes; large enough to back a real native pointer

/**
 * Thin helpers around `OnnxTensor.createTensor(env, ...)` so that
 * [InferenceLoop] reads like the Python reference instead of like a
 * memory-management essay.
 *
 * Every [OnnxTensor] returned **must** be `.close()`d by the caller -
 * see [InferenceLoop.generate]'s try/finally chains. Buffers are direct +
 * little-endian per ORT's contract for zero-copy host inputs.
 *
 * Type coverage:
 *   - int32 / int64 / float32 — the three numeric dtypes the v1.0.0
 *     release uses across all 4 TTS + 3 codec graphs.
 *   - bool — kept around for legacy attention masks; the v1.0.0 graphs
 *     use int32 attention masks instead, but bool support costs nothing.
 */
internal object OrtTensors {
    fun longTensor(env: OrtEnvironment, data: LongArray, shape: LongArray): OnnxTensor {
        val bb = ByteBuffer.allocateDirect(data.size * java.lang.Long.BYTES)
            .order(ByteOrder.nativeOrder())
        val lb: LongBuffer = bb.asLongBuffer()
        lb.put(data)
        bb.rewind()
        return OnnxTensor.createTensor(env, bb.asLongBuffer(), shape)
    }

    fun intTensor(env: OrtEnvironment, data: IntArray, shape: LongArray): OnnxTensor {
        val bb = ByteBuffer.allocateDirect(data.size * java.lang.Integer.BYTES)
            .order(ByteOrder.nativeOrder())
        val ib: IntBuffer = bb.asIntBuffer()
        ib.put(data)
        bb.rewind()
        return OnnxTensor.createTensor(env, bb.asIntBuffer(), shape)
    }

    fun floatTensor(env: OrtEnvironment, data: FloatArray, shape: LongArray): OnnxTensor {
        val bb = ByteBuffer.allocateDirect(data.size * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
        val fb: FloatBuffer = bb.asFloatBuffer()
        fb.put(data)
        bb.rewind()
        return OnnxTensor.createTensor(env, bb.asFloatBuffer(), shape)
    }

    /**
     * BOOL tensor (kept for completeness). ORT-Java represents BOOL as a
     * byte (0 or 1) per element via [OnnxJavaType.BOOL] + a direct
     * [ByteBuffer]. v1.0.0 graphs use int32 masks, but legacy callers may
     * still want this.
     */
    fun boolTensor(env: OrtEnvironment, data: BooleanArray, shape: LongArray): OnnxTensor {
        val bb = ByteBuffer.allocateDirect(data.size).order(ByteOrder.nativeOrder())
        for (b in data) bb.put(if (b) 1.toByte() else 0.toByte())
        bb.rewind()
        return OnnxTensor.createTensor(env, bb, shape, OnnxJavaType.BOOL)
    }

    /**
     * Zero-element float tensor — equivalent to Python's `np.zeros((1, 0, h, d))`.
     * Used as the "empty past_kv" placeholder for the very first local step.
     *
     * Uses a raw [ByteBuffer] with non-zero capacity but `remaining() == 0`,
     * which gives ORT a valid native pointer while correctly reporting
     * `byteSize = 0` to match `product(shape) == 0`.
     */
    fun emptyFloatTensor(env: OrtEnvironment, shape: LongArray): OnnxTensor {
        require(shape.any { it == 0L }) {
            "emptyFloatTensor requires a shape with at least one 0 dim, got ${shape.toList()}"
        }
        val bb = ByteBuffer.allocateDirect(EMPTY_BUF_CAPACITY)
            .order(ByteOrder.nativeOrder())
        bb.position(bb.limit()) // remaining() == 0 → byteSize 0, matches product(shape)
        return OnnxTensor.createTensor(env, bb, shape, OnnxJavaType.FLOAT)
    }

    /**
     * Materialize a 2-D `[1, N]` float OnnxTensor as a flat [FloatArray] of
     * length N. Used for the per-step logits (text head, single audio channel slice).
     */
    fun readFloat1D(t: OnnxTensor): FloatArray {
        @Suppress("UNCHECKED_CAST")
        val rows = t.value as Array<FloatArray>
        check(rows.size == 1) { "expected leading dim 1, got ${rows.size}" }
        return rows[0]
    }

    /**
     * Materialize a 3-D `[1, R, C]` float OnnxTensor's row [rowIdx]
     * as a flat [FloatArray] of length C. Used to slice one channel's
     * logits out of `audio_logits[1, 16, 1024]`.
     */
    fun readFloat3DRow(t: OnnxTensor, rowIdx: Int): FloatArray {
        @Suppress("UNCHECKED_CAST")
        val outer = t.value as Array<Array<FloatArray>>
        check(outer.size == 1) { "expected leading dim 1, got ${outer.size}" }
        check(rowIdx in outer[0].indices) { "rowIdx=$rowIdx out of bounds (size=${outer[0].size})" }
        return outer[0][rowIdx]
    }

    /**
     * Materialize a 3-D `[1, T, H]` float OnnxTensor's last row
     * (`out[0:1, -1, :]`) as a flat [FloatArray] of length H. Used to
     * pull `last_hidden = global_hidden[:, -1, :]` from prefill / decode_step.
     */
    fun readFloat3DLastRow(t: OnnxTensor): FloatArray {
        @Suppress("UNCHECKED_CAST")
        val outer = t.value as Array<Array<FloatArray>>
        check(outer.size == 1) { "expected leading dim 1, got ${outer.size}" }
        val rows = outer[0]
        check(rows.isNotEmpty()) { "expected at least one row in dim 1" }
        return rows[rows.size - 1]
    }
}
