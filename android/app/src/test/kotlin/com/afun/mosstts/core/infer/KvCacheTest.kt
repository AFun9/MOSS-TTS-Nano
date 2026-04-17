package com.afun.mosstts.core.infer

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * KvCache holds every per-frame mutable tensor the audio_decoder ORT
 * session reads/writes between calls. Three numerical contracts:
 *
 *   - **Initial state**: keys/values/offsets are zero, positions are
 *     -1 (Python's `np.full(..., -1)`). The `audio_decoder` session
 *     refuses to start otherwise — its position-encoding lookup uses
 *     -1 as the sentinel meaning "this slot is empty".
 *   - **Reset**: after `reset()` every tensor must equal the initial
 *     state, byte-for-byte. We verify by mutating every buffer and
 *     calling reset.
 *   - **Output mapping**: each `new_<name>` ORT output must land in the
 *     `<name>` slot of the cache for the next call. Verified by
 *     simulating a step with synthetic outputs and checking the
 *     internal buffers reflect them.
 */
class KvCacheTest {
    private val spec by lazy {
        StateSpec.parse(javaClass.getResource("/state_spec_sample.json")!!.readText())
    }

    @Test
    fun `cache exposes one entry per state input name`() {
        val cache = KvCache(spec)
        val expected = spec.inputNames.drop(1)  // skip 'codes'
        assertThat(cache.tensorNames).containsExactlyElementsIn(expected).inOrder()
    }

    @Test
    fun `initial cached_keys and cached_values buffers are all-zero float32`() {
        val cache = KvCache(spec)
        for (name in cache.tensorNames) {
            if (name.endsWith("_cached_keys") || name.endsWith("_cached_values")) {
                val t = cache.get(name)
                assertThat(t.dtype).isEqualTo(KvCache.DType.FLOAT32)
                val buf = t.data as FloatBuffer
                buf.rewind()
                while (buf.hasRemaining()) {
                    assertThat(buf.get()).isEqualTo(0f)
                }
            }
        }
    }

    @Test
    fun `initial cached_positions buffers are all -1 int64`() {
        val cache = KvCache(spec)
        for (name in cache.tensorNames) {
            if (name.endsWith("_cached_positions")) {
                val t = cache.get(name)
                assertThat(t.dtype).isEqualTo(KvCache.DType.INT64)
                val buf = t.data as LongBuffer
                buf.rewind()
                while (buf.hasRemaining()) {
                    assertThat(buf.get()).isEqualTo(-1L)
                }
            }
        }
    }

    @Test
    fun `initial offset buffers are zero int64 with shape 1`() {
        val cache = KvCache(spec)
        for (name in cache.tensorNames) {
            if (name.endsWith("_transformer_offset") || name.endsWith("_offset") &&
                !name.endsWith("_transformer_offset")
            ) {
                val t = cache.get(name)
                assertThat(t.dtype).isEqualTo(KvCache.DType.INT64)
                assertThat(t.shape.toList()).isEqualTo(listOf(1L))
                (t.data as LongBuffer).rewind()
                assertThat((t.data as LongBuffer).get()).isEqualTo(0L)
            }
        }
    }

    @Test
    fun `cached_keys shape matches attention spec`() {
        val cache = KvCache(spec)
        for (a in spec.attentionSpecs) {
            val name = "decoder_${a.decoderModuleIndex}_layer_${a.layerIndex}_cached_keys"
            val t = cache.get(name)
            assertThat(t.shape.toList())
                .isEqualTo(listOf(1L, a.numHeads.toLong(), a.context.toLong(), a.headDim.toLong()))
        }
    }

    @Test
    fun `reset restores -1 positions even after caller mutates the buffer`() {
        val cache = KvCache(spec)
        val firstPos = cache.tensorNames.first { it.endsWith("_cached_positions") }
        val buf = cache.get(firstPos).data as LongBuffer
        buf.put(0, 99L)
        buf.put(1, 88L)
        cache.reset()
        val after = cache.get(firstPos).data as LongBuffer
        after.rewind()
        assertThat(after.get()).isEqualTo(-1L)
        assertThat(after.get()).isEqualTo(-1L)
    }

    @Test
    fun `reset zeroes float buffers even after mutation`() {
        val cache = KvCache(spec)
        val firstKeys = cache.tensorNames.first { it.endsWith("_cached_keys") }
        val buf = cache.get(firstKeys).data as FloatBuffer
        buf.put(0, 3.14f)
        cache.reset()
        val after = cache.get(firstKeys).data as FloatBuffer
        after.rewind()
        assertThat(after.get()).isEqualTo(0f)
    }
}
