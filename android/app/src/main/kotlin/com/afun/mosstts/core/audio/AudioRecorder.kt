package com.afun.mosstts.core.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.max

/**
 * Records mono 16-bit PCM at 44100 Hz using [AudioRecord].
 * Max recording duration is [MAX_SECONDS]; auto-stops when reached.
 *
 * Usage:
 * ```
 * val recorder = AudioRecorder()
 * recorder.start()
 * // ... user presses stop ...
 * val audio = recorder.stop()
 * ```
 */
class AudioRecorder {

    companion object {
        const val SAMPLE_RATE = 44100
        const val MAX_SECONDS = 20
    }

    private val recording = AtomicBoolean(false)
    private var recorder: AudioRecord? = null
    private var capturedSamples: ShortArray? = null

    private val _amplitude = MutableStateFlow(0f)
    /** Normalized peak amplitude [0..1] updated ~20 times/sec while recording. */
    val amplitude: StateFlow<Float> = _amplitude.asStateFlow()

    private val _recordingSeconds = MutableStateFlow(0f)
    /** Elapsed recording time in seconds. */
    val recordingSeconds: StateFlow<Float> = _recordingSeconds.asStateFlow()

    val isRecording: Boolean get() = recording.get()

    /**
     * Starts recording on a background thread. Returns immediately.
     * Call [stop] to finish and retrieve the audio.
     */
    @Suppress("MissingPermission")
    suspend fun start() = withContext(Dispatchers.IO) {
        check(!recording.get()) { "Already recording" }

        val bufSize = maxOf(
            AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            ),
            SAMPLE_RATE * 2, // at least 1 second buffer
        )

        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize,
        )
        check(rec.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord failed to initialize" }

        val maxSamples = SAMPLE_RATE * MAX_SECONDS
        val allSamples = ShortArray(maxSamples)
        var written = 0

        rec.startRecording()
        recorder = rec
        recording.set(true)
        _amplitude.value = 0f
        _recordingSeconds.value = 0f

        val readBuf = ShortArray(2048)
        while (recording.get() && written < maxSamples) {
            val toRead = minOf(readBuf.size, maxSamples - written)
            val n = rec.read(readBuf, 0, toRead)
            if (n > 0) {
                System.arraycopy(readBuf, 0, allSamples, written, n)
                written += n

                var peak: Short = 0
                for (i in 0 until n) peak = max(peak.toInt(), abs(readBuf[i].toInt())).toShort()
                _amplitude.value = peak.toFloat() / 32768f
                _recordingSeconds.value = written.toFloat() / SAMPLE_RATE
            }
        }

        recording.set(false)
        _amplitude.value = 0f
        rec.stop()
        rec.release()
        recorder = null

        capturedSamples = if (written > 0) allSamples.copyOf(written) else null
    }

    /**
     * Stops recording and returns the captured audio as [DecodedAudio].
     * Returns null if no audio was captured.
     */
    fun stop(): DecodedAudio? {
        recording.set(false)
        // Wait briefly for the recording coroutine to finish and set capturedSamples.
        // In practice, the caller should await the start() coroutine.
        val samples = capturedSamples ?: return null
        capturedSamples = null

        val inv = 1f / 32768f
        val floats = FloatArray(samples.size) { samples[it] * inv }
        return DecodedAudio(
            pcm = arrayOf(floats),
            sampleRate = SAMPLE_RATE,
            channels = 1,
        )
    }
}
