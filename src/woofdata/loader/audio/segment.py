"""Read one manifest row's time window from its 16 kHz WAV."""

from __future__ import annotations

import numpy as np
import soundfile

from woofdata.loader.annotations import Row
from woofdata.path import data_path
from woofdata.prepare.utils.audio import TARGET_SAMPLE_RATE


def load_audio_segment(row: Row, max_seconds: float | None = None) -> np.ndarray:
    """Read the row's start_s..end_s from its 16 kHz WAV.

    A window running past the end of the file is padded with zeros. When max_seconds is
    set and the segment is longer, only its centre max_seconds is read (deterministic).

    Returns:
        float32 samples, shape (num_samples,).

    Raises:
        ValueError: If the file is not mono at TARGET_SAMPLE_RATE.
    """
    start_s = float(row["start_s"])
    end_s = float(row["end_s"])
    if max_seconds is not None and end_s - start_s > max_seconds:
        centre_s = (start_s + end_s) / 2
        start_s = centre_s - max_seconds / 2
        end_s = start_s + max_seconds
    start_frame = round(start_s * TARGET_SAMPLE_RATE)
    num_frames = round((end_s - start_s) * TARGET_SAMPLE_RATE)

    samples, sample_rate = soundfile.read(
        data_path(row["file_path"]),
        frames=num_frames,
        start=start_frame,
        dtype="float32",
        fill_value=0.0,  # frames past the end of the file
    )
    if sample_rate != TARGET_SAMPLE_RATE or samples.ndim != 1:
        raise ValueError(f"expected mono {TARGET_SAMPLE_RATE} Hz, got {sample_rate} Hz")
    return samples  # shape (num_frames,)
