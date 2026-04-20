package com.afun.mosstts.core.audio

/**
 * Raw PCM audio with metadata. [pcm] is channel-major: `[channels][samples]`,
 * values normalised to `[-1, 1]`.
 */
data class DecodedAudio(
    val pcm: Array<FloatArray>,
    val sampleRate: Int,
    val channels: Int,
) {
    val durationSeconds: Float get() = if (pcm.isNotEmpty() && pcm[0].isNotEmpty())
        pcm[0].size.toFloat() / sampleRate else 0f

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DecodedAudio) return false
        if (sampleRate != other.sampleRate || channels != other.channels) return false
        if (pcm.size != other.pcm.size) return false
        for (i in pcm.indices) if (!pcm[i].contentEquals(other.pcm[i])) return false
        return true
    }

    override fun hashCode(): Int {
        var h = sampleRate * 31 + channels
        for (ch in pcm) h = h * 31 + ch.contentHashCode()
        return h
    }
}
