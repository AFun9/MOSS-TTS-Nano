package com.afun.mosstts.core.voice

import android.content.Context
import com.afun.mosstts.core.infer.AudioCodes
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class BuiltinVoice(
    val voice: String,
    @SerialName("display_name") val displayName: String,
    val group: String,
    @SerialName("prompt_audio_codes") val promptAudioCodes: List<List<Int>>,
) {
    fun toAudioCodes(nVq: Int): AudioCodes {
        val frames = promptAudioCodes.size
        val data = LongArray(frames * nVq)
        for (f in 0 until frames) {
            val row = promptAudioCodes[f]
            for (k in 0 until nVq) data[f * nVq + k] = row[k].toLong()
        }
        return AudioCodes(frames = frames, nVq = nVq, data = data)
    }

    companion object {
        private val parser = Json { ignoreUnknownKeys = true }

        fun loadAll(context: Context): List<BuiltinVoice> {
            val json = context.assets.open("builtin_voices.json")
                .bufferedReader().use { it.readText() }
            return parser.decodeFromString(json)
        }
    }
}
