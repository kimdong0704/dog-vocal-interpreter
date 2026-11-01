"""BarkSegment: one detected vocalization plus the media around it"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from woofdecoder.sed_classes import SED_CLASSES

TIME_TOLERANCE_S: float = 1e-3


class SoundSection(NamedTuple):
    label: str
    span_s: tuple[float, float]
    score: float

@dataclass
class BarkSegment:
    """One vocalization, passed from the preprocessing step to the branches.
 
    Preprocessing runs the bark detector on the audio and a dog detector on each sampled
    frame, then packs everything one branch needs about one vocalization in here.
 
    Attributes:
        start_s: Start time in the source clip, in seconds.
        end_s: End time in the source clip, in seconds. Must not be before start_s.
        audio: Mono float32 waveform, shape (num_samples,), or None if there is no audio.
        sample_rate: Audio sample rate in Hz.
        frames: RGB uint8 video frames, each of shape (height, width, 3). Empty if there
            is no video.
        frame_times_s: Timestamp of each frame in the source clip, in seconds. Same length
            and order as frames.
        source_id: Video/clip id, for logging and debugging.
        confidence: How sure the bark detector is that a dog is vocalizing in this
            segment: the mean dog-vocalization score over its frames, from 0 to 1 (a
            detector score, not a calibrated probability), or None if unknown.
        dog_boxes: One entry per frame, same order as frames. Each box is a float32 array
            of shape (4,) as x, y, w, h in pixels of that frame, or None if no dog was
            found in that frame. An empty list means dog detection was not run. If several
            dogs are visible, preprocessing picks one: the box is the dog this segment is
            about.
        sound_sections: What the sounds are and when, in time order: one SoundSection per
            stretch where one sound is strongest, e.g.
            [("growl", (1.20, 1.40), 0.41), ("bark", (1.40, 1.64), 0.22)].
            Empty if unknown (for example hand-marked stand-in segments).
 
    Raises:
        ValueError: If the fields do not fit together (see __post_init__).
    """

    start_s: float
    end_s: float
    audio: np.ndarray | None = None
    sample_rate: int = 16_000
    frames: list[np.ndarray] = field(default_factory=list)
    frame_times_s: list[float] = field(default_factory=list)
    source_id: str = ""
    confidence: float | None = None
    sound_sections: list[SoundSection] = field(default_factory=list)
    dog_boxes: list[np.ndarray | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check that the fields fit together and store each dog box as float32.
 
        Raises:
            ValueError: If end_s is before start_s, frame_times_s or dog_boxes do not
                match frames in length, a box is not shape (4,) or has no area,
                confidence is outside 0 to 1, or a sound section has an unknown label,
                lies outside the segment or has a score outside 0 to 1.
        """
        if self.end_s < self.start_s:
            raise ValueError(f"end_s ({self.end_s}) is before start_s ({self.start_s})")

        if len(self.frame_times_s) != len(self.frames):
            raise ValueError(
                f"frame_times_s has {len(self.frame_times_s)} entries but frames has "
                f"{len(self.frames)}"
            )

        if self.has_dog_boxes and len(self.dog_boxes) != len(self.frames):
            raise ValueError(
                f"dog_boxes has {len(self.dog_boxes)} entries but frames has "
                f"{len(self.frames)}; use one entry per frame, or an empty list"
            )

        checked_boxes: list[np.ndarray | None] = []
        for frame_index, box in enumerate(self.dog_boxes):
            if box is None:
                checked_boxes.append(None)
                continue

            box_array = np.asarray(box, dtype=np.float32)  # shape (4,) as x, y, w, h
            if box_array.shape != (4,):
                raise ValueError(
                    f"dog box for frame {frame_index} has shape {box_array.shape}, "
                    f"expected (4,) as x, y, w, h"
                )

            box_width = box_array[2]
            box_height = box_array[3]
            if box_width <= 0 or box_height <= 0:
                raise ValueError(
                    f"dog box for frame {frame_index} has w={box_width}, h={box_height}; "
                    f"both must be > 0"
                )

            checked_boxes.append(box_array)
        self.dog_boxes = checked_boxes

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} is not between 0 and 1")
        
        checked_sections: list[SoundSection] = []
        for label, (section_start_s, section_end_s), score in self.sound_sections:
            if label not in SED_CLASSES:
                raise ValueError(f"section label{label!r} is not one of {SED_CLASSES}")
            
            inside = (
                self.start_s - TIME_TOLERANCE_S 
                <= section_start_s
                < section_end_s
                <= self.end_s + TIME_TOLERANCE_S
            )
            if not inside:
                raise ValueError(f"sound section ({section_start_s}, {section_end_s}) is outside the segment")

            if not 0.0 <= score <= 1.0:
                raise ValueError(f"sound section score{score} is not between 0 and 1")
            
            checked_sections.append(SoundSection(label, (section_start_s, section_end_s), score))
        self.sound_sections = checked_sections
                
    @property
    def duration_s(self) -> float:
        """Length of the segment in seconds."""
        return self.end_s - self.start_s

    @property
    def has_audio(self) -> bool:
        """True if the segment carries a non-empty waveform."""
        return (self.audio is not None) and (self.audio.size > 0)

    @property
    def has_video(self) -> bool:
        """True if the segment carries at least one frame."""
        return len(self.frames) > 0

    @property
    def has_dog_boxes(self) -> bool:
        """True if dog detection was run on the frames (dog_boxes is not empty)."""
        return len(self.dog_boxes) > 0

    @property
    def dog_visible(self) -> bool:
        """True if a dog was found in at least one frame."""
        for box in self.dog_boxes:
            if box is not None:
                return True
        return False
