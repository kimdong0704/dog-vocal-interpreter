"""Read any audio file and write it as 16 kHz mono 16-bit WAV (pure Python, no ffmpeg)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile
import soxr

from woofdecoder.config import load_config
from woofdecoder.preprocess.media import to_mono_16k

TARGET_SAMPLE_RATE: int = load_config()["media"]["sample_rate"]
WAV_ENCODING: str = "PCM_16"
# soxr "HQ" quality: deterministic and good enough for 16 kHz speech-band audio.
RESAMPLE_QUALITY: str = "HQ"


def load_audio(source: str | Path | bytes) -> tuple[np.ndarray, int]:
    """Read an audio file (path or raw bytes) of any bit depth or channel count.

    Returns:
        (samples, sample_rate): float32 samples in [-1, 1], shape (num_samples, num_channels).
    """
    audio_file: str | Path | io.BytesIO
    if isinstance(source, bytes):
        audio_file = io.BytesIO(source)
    else:
        audio_file = source
    samples, sample_rate = soundfile.read(audio_file, dtype="float32", always_2d=True)
    return samples, sample_rate  # samples shape (num_samples, num_channels)


def to_mono_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Average the channels and resample to TARGET_SAMPLE_RATE with soxr.

    Args:
        samples: shape (num_samples,) or (num_samples, num_channels).
        sample_rate: Sample rate of samples in Hz.

    Returns:
        float32 samples, shape (num_samples_16k,).
    """
    if samples.ndim == 2:
        mono = samples.mean(axis=1)  # shape (num_samples,)
    else:
        mono = samples
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = soxr.resample(mono, sample_rate, TARGET_SAMPLE_RATE, quality=RESAMPLE_QUALITY)
    return mono.astype(np.float32, copy=False)  # shape (num_samples_16k,)


def write_wav_16k(path: Path, samples: np.ndarray) -> None:
    """Write mono samples (shape (num_samples,)) as a TARGET_SAMPLE_RATE 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)  # values beyond full scale would wrap around
    soundfile.write(path, clipped, TARGET_SAMPLE_RATE, subtype=WAV_ENCODING)


def convert_audio(source: str | Path | bytes, out_path: Path) -> float:
    """Read any audio file (path or bytes), write it as 16 kHz mono WAV, return its seconds."""
    samples, sample_rate = load_audio(source)
    mono = to_mono_16k(samples, sample_rate)  # shape (num_samples_16k,)
    write_wav_16k(out_path, mono)
    return len(mono) / TARGET_SAMPLE_RATE
