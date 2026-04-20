package com.afun.mosstts.core.audio

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Decodes an audio file (mp3, m4a, wav, ogg, etc.) referenced by a content [Uri]
 * into raw PCM [DecodedAudio] using Android's [MediaExtractor] + [MediaCodec].
 *
 * All work is synchronous and should be called from a background thread.
 */
object AudioFileDecoder {

    private const val TIMEOUT_US = 10_000L

    fun decode(context: Context, uri: Uri): DecodedAudio {
        val extractor = MediaExtractor()
        try {
            context.contentResolver.openFileDescriptor(uri, "r")?.use { pfd ->
                extractor.setDataSource(pfd.fileDescriptor)
            } ?: error("Cannot open URI: $uri")

            val trackIdx = findAudioTrack(extractor)
                ?: error("No audio track found in $uri")
            extractor.selectTrack(trackIdx)
            val format = extractor.getTrackFormat(trackIdx)

            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: error("No MIME type in track format")
            val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            val codec = MediaCodec.createDecoderByType(mime)
            try {
                codec.configure(format, null, null, 0)
                codec.start()

                val rawShorts = decodeLoop(extractor, codec)
                val pcm = shortsToChannelFloat(rawShorts, channels)
                return DecodedAudio(pcm = pcm, sampleRate = sampleRate, channels = channels)
            } finally {
                codec.stop()
                codec.release()
            }
        } finally {
            extractor.release()
        }
    }

    private fun findAudioTrack(extractor: MediaExtractor): Int? {
        for (i in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) return i
        }
        return null
    }

    private fun decodeLoop(extractor: MediaExtractor, codec: MediaCodec): ShortArray {
        val bufInfo = MediaCodec.BufferInfo()
        var eos = false
        val allShorts = ArrayList<ShortArray>(256)
        var totalShorts = 0

        while (true) {
            // Feed input
            if (!eos) {
                val inIdx = codec.dequeueInputBuffer(TIMEOUT_US)
                if (inIdx >= 0) {
                    val inBuf = codec.getInputBuffer(inIdx)!!
                    val read = extractor.readSampleData(inBuf, 0)
                    if (read < 0) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0,
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        eos = true
                    } else {
                        codec.queueInputBuffer(inIdx, 0, read,
                            extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            // Drain output
            val outIdx = codec.dequeueOutputBuffer(bufInfo, TIMEOUT_US)
            if (outIdx >= 0) {
                if (bufInfo.size > 0) {
                    val outBuf = codec.getOutputBuffer(outIdx)!!
                    outBuf.position(bufInfo.offset)
                    outBuf.limit(bufInfo.offset + bufInfo.size)
                    val shorts = ShortArray(bufInfo.size / 2)
                    outBuf.order(ByteOrder.nativeOrder()).asShortBuffer().get(shorts)
                    allShorts.add(shorts)
                    totalShorts += shorts.size
                }
                codec.releaseOutputBuffer(outIdx, false)
                if (bufInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            } else if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                // Format changed, continue
            }
        }

        val result = ShortArray(totalShorts)
        var pos = 0
        for (chunk in allShorts) {
            System.arraycopy(chunk, 0, result, pos, chunk.size)
            pos += chunk.size
        }
        return result
    }

    private fun shortsToChannelFloat(
        interleaved: ShortArray,
        channels: Int,
    ): Array<FloatArray> {
        val samplesPerChannel = interleaved.size / channels
        val pcm = Array(channels) { FloatArray(samplesPerChannel) }
        val inv = 1f / 32768f
        for (i in 0 until samplesPerChannel) {
            for (c in 0 until channels) {
                pcm[c][i] = interleaved[i * channels + c] * inv
            }
        }
        return pcm
    }
}
