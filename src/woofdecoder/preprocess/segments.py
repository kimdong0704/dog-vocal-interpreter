"""BarkSegment: one detected vocalization plus the media around it"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BarkSegment:
    """
    Bark detector --> Braches

    Attributes:
        start_s: Start time in the source clip, in seconds.
        end_s: End time in the source clip, in seconds.
        audio: Mono float32 waveform, shape (num_samples,), or None if there is no audio.
        sample_rate: Audio sample rate in Hz.
        frames: RGB uint8 video frames, each of shape (H, W, 3). Empty if there is no video.
        frame_times_s: Timestamp of each frame in seconds, same length as frames.
        source_id: Video/clip id, for logging and debugging.
    """

    start_s: float
    end_s: float
    audio: np.ndarray | None = None  # shape (num_samples,), mono float32
    sample_rate: int = 16_000  # 16 kHz
    frames: list[np.ndarray] = field(default_factory=list)  # each shape (H, W, 3), RGB uint8
    frame_times_s: list[float] = field(default_factory=list)  # timestamp of each frame in sec
    source_id: str = ""  # for logging and debugging purposes

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def has_audio(self) -> bool:
        return (self.audio is not None) and (self.audio.size > 0)

    @property
    def has_video(self) -> bool:
        return len(self.frames) > 0
