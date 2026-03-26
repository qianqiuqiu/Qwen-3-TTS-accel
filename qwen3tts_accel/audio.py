from __future__ import annotations

import io
import wave

import numpy as np


def waveform_to_pcm_s16le_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


def waveform_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm_bytes = waveform_to_pcm_s16le_bytes(audio)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()
