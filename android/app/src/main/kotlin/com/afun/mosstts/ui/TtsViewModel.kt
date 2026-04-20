package com.afun.mosstts.ui

import android.app.Application
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.afun.mosstts.core.audio.AudioFileDecoder
import com.afun.mosstts.core.audio.AudioRecorder
import com.afun.mosstts.core.engine.StreamEvent
import com.afun.mosstts.core.engine.TtsEngine
import com.afun.mosstts.core.infer.AudioCodes
import com.afun.mosstts.core.voice.BuiltinVoice
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class Phase { Idle, Initializing, Generating, Ready, Error }

data class TtsUiState(
    val text: String = "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。",
    val selectedVoiceIdx: Int = 0,
    val voices: List<BuiltinVoice> = emptyList(),
    val phase: Phase = Phase.Idle,
    val isPlaying: Boolean = false,
    val errorMessage: String? = null,
    val generationTimeMs: Long = 0,
    val audioDurationSec: Float = 0f,
    val framesGenerated: Int = 0,
    val cloneAudioCodes: AudioCodes? = null,
    val cloneAudioDuration: Float = 0f,
    val isEncoding: Boolean = false,
    val isRecording: Boolean = false,
)

class TtsViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = TtsEngine(app)
    private val _state = MutableStateFlow(TtsUiState())
    val state: StateFlow<TtsUiState> = _state.asStateFlow()

    private var streamTrack: AudioTrack? = null
    private var replayTrack: AudioTrack? = null
    private var replayPcm: ShortArray? = null
    private var replaySampleRate = ModelConfig_CODEC_SR
    private var replayChannels = ModelConfig_CODEC_CH
    private var streamFramesWritten = 0

    init {
        viewModelScope.launch {
            _state.update { it.copy(phase = Phase.Initializing) }
            try {
                engine.initialize()
                _state.update {
                    it.copy(phase = Phase.Idle, voices = engine.getVoices())
                }
            } catch (e: Exception) {
                _state.update {
                    it.copy(phase = Phase.Error, errorMessage = "Init failed: ${e.message}")
                }
            }
        }
    }

    fun onTextChanged(text: String) {
        _state.update { it.copy(text = text) }
    }

    fun onVoiceSelected(idx: Int) {
        _state.update { it.copy(selectedVoiceIdx = idx) }
    }

    // ---- Clone: recording ----

    private val recorder = AudioRecorder()
    private var recordJob: Job? = null

    val recordingAmplitude = recorder.amplitude
    val recordingSeconds = recorder.recordingSeconds

    fun startRecording() {
        if (_state.value.isRecording) return
        _state.update { it.copy(isRecording = true, cloneAudioCodes = null, cloneAudioDuration = 0f) }
        recordJob = viewModelScope.launch {
            try {
                recorder.start()
                val audio = recorder.stop()
                _state.update { it.copy(isRecording = false) }
                if (audio != null) {
                    encodeAndSetClone(audio.pcm, audio.sampleRate, audio.durationSeconds)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Recording failed", e)
                _state.update { it.copy(isRecording = false, errorMessage = "录音失败: ${e.message}") }
            }
        }
    }

    fun stopRecording() {
        recorder.stop()
        // The start() coroutine will finish and call encodeAndSetClone
    }

    // ---- Clone: file import ----

    fun importAudio(uri: Uri) {
        viewModelScope.launch {
            _state.update { it.copy(isEncoding = true, cloneAudioCodes = null, cloneAudioDuration = 0f) }
            try {
                val audio = withContext(Dispatchers.IO) {
                    AudioFileDecoder.decode(getApplication(), uri)
                }
                encodeAndSetClone(audio.pcm, audio.sampleRate, audio.durationSeconds)
            } catch (e: Exception) {
                Log.e(TAG, "Import failed", e)
                _state.update { it.copy(isEncoding = false, errorMessage = "导入失败: ${e.message}") }
            }
        }
    }

    private suspend fun encodeAndSetClone(pcm: Array<FloatArray>, sampleRate: Int, duration: Float) {
        _state.update { it.copy(isEncoding = true) }
        try {
            val codes = engine.encodeReferenceAudio(pcm, sampleRate)
            _state.update { it.copy(cloneAudioCodes = codes, cloneAudioDuration = duration, isEncoding = false) }
        } catch (e: Exception) {
            Log.e(TAG, "Encode failed", e)
            _state.update { it.copy(isEncoding = false, errorMessage = "编码失败: ${e.message}") }
        }
    }

    fun clearCloneAudio() {
        _state.update { it.copy(cloneAudioCodes = null, cloneAudioDuration = 0f) }
    }

    // ---- Generation ----

    fun generate() {
        launchGeneration(customCodes = null)
    }

    fun generateWithClone() {
        val codes = _state.value.cloneAudioCodes ?: return
        launchGeneration(customCodes = codes)
    }

    private fun launchGeneration(customCodes: AudioCodes?) {
        val s = _state.value
        if (s.phase == Phase.Generating) {
            engine.cancelGeneration()
            return
        }
        if (s.text.isBlank() || !engine.isInitialized) return
        stopAllPlayback()

        val pcmChunks = ArrayList<ShortArray>()
        streamFramesWritten = 0

        viewModelScope.launch {
            _state.update {
                it.copy(
                    phase = Phase.Generating,
                    framesGenerated = 0,
                    errorMessage = null,
                    isPlaying = false,
                )
            }
            try {
                engine.generateStreaming(
                    s.text.trim(), s.selectedVoiceIdx,
                    customCodes = customCodes,
                ) { event ->
                    when (event) {
                        is StreamEvent.Progress -> {
                            _state.update { it.copy(framesGenerated = event.frame) }
                        }
                        is StreamEvent.AudioChunk -> {
                            pcmChunks.add(event.pcmShorts)
                            ensureStreamTrack(event.sampleRate, event.channels)
                            streamTrack?.write(event.pcmShorts, 0, event.pcmShorts.size)
                            streamFramesWritten += event.pcmShorts.size / event.channels
                            if (!_state.value.isPlaying) {
                                _state.update { it.copy(isPlaying = true) }
                            }
                        }
                        is StreamEvent.Done -> {
                            replayPcm = mergeChunks(pcmChunks)
                            _state.update {
                                it.copy(
                                    phase = Phase.Ready,
                                    generationTimeMs = event.genTimeMs,
                                    audioDurationSec = event.durationSec,
                                )
                            }
                        }
                    }
                }

                awaitStreamDrain()
            } catch (e: Exception) {
                Log.e(TAG, "generateStreaming failed", e)
                stopAllPlayback()
                _state.update {
                    it.copy(phase = Phase.Error, errorMessage = "Generate failed: ${e.message}")
                }
            }
        }
    }

    fun togglePlayback() {
        if (_state.value.isPlaying) {
            stopAllPlayback()
            return
        }
        val pcm = replayPcm ?: return
        stopAllPlayback()
        replayWithStatic(pcm)
    }

    fun dismissError() {
        _state.update { it.copy(errorMessage = null, phase = Phase.Idle) }
    }

    // ---------- streaming internals ----------

    private fun ensureStreamTrack(sampleRate: Int, channels: Int) {
        if (streamTrack != null) return
        replaySampleRate = sampleRate
        replayChannels = channels

        val chCfg = if (channels == 2) AudioFormat.CHANNEL_OUT_STEREO else AudioFormat.CHANNEL_OUT_MONO
        val minBuf = AudioTrack.getMinBufferSize(sampleRate, chCfg, AudioFormat.ENCODING_PCM_16BIT)
        streamTrack = AudioTrack.Builder()
            .setAudioAttributes(speechAttrs())
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(chCfg)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBuf * 4, 65536))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
            .also { it.play() }
    }

    private suspend fun awaitStreamDrain() {
        val track = streamTrack ?: return
        val target = streamFramesWritten
        if (target <= 0) {
            releaseStreamTrack()
            return
        }
        withContext(Dispatchers.Default) {
            while (track.playState == AudioTrack.PLAYSTATE_PLAYING) {
                if (track.playbackHeadPosition >= target) break
                delay(80)
            }
        }
        releaseStreamTrack()
        _state.update { it.copy(isPlaying = false) }
    }

    private fun releaseStreamTrack() {
        streamTrack?.let { t ->
            try { t.stop() } catch (_: Exception) {}
            t.release()
        }
        streamTrack = null
    }

    // ---------- static replay ----------

    private fun replayWithStatic(pcm: ShortArray) {
        val chCfg = if (replayChannels == 2) AudioFormat.CHANNEL_OUT_STEREO else AudioFormat.CHANNEL_OUT_MONO
        val track = AudioTrack.Builder()
            .setAudioAttributes(speechAttrs())
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(replaySampleRate)
                    .setChannelMask(chCfg)
                    .build()
            )
            .setBufferSizeInBytes(pcm.size * 2)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        track.write(pcm, 0, pcm.size)
        val nSamples = pcm.size / replayChannels
        track.setNotificationMarkerPosition(nSamples)
        track.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
            override fun onMarkerReached(t: AudioTrack?) {
                _state.update { it.copy(isPlaying = false) }
            }
            override fun onPeriodicNotification(t: AudioTrack?) {}
        })
        track.play()
        replayTrack = track
        _state.update { it.copy(isPlaying = true) }
    }

    // ---------- cleanup ----------

    private fun stopAllPlayback() {
        streamTrack?.let { t ->
            try { t.pause() } catch (_: Exception) {}
            try { t.stop() } catch (_: Exception) {}
            t.release()
        }
        streamTrack = null

        replayTrack?.let { t ->
            try { t.stop() } catch (_: Exception) {}
            t.release()
        }
        replayTrack = null
        replayPcm = null

        _state.update { it.copy(isPlaying = false) }
    }

    override fun onCleared() {
        stopAllPlayback()
        engine.release()
        super.onCleared()
    }

    // ---------- helpers ----------

    private fun mergeChunks(chunks: List<ShortArray>): ShortArray {
        val total = chunks.sumOf { it.size }
        val out = ShortArray(total)
        var pos = 0
        for (c in chunks) {
            System.arraycopy(c, 0, out, pos, c.size)
            pos += c.size
        }
        return out
    }

    private fun speechAttrs() = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_MEDIA)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()

    companion object {
        private const val TAG = "TtsViewModel"
        private const val ModelConfig_CODEC_SR = 48000
        private const val ModelConfig_CODEC_CH = 2
    }
}
