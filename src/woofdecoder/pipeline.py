"""Clip -> predictions. For now: preprocessing (bark detector) + audio branch only.

segments = build_segments(clip)            one BarkSegment per detected dog sound
output   = audio_branch.predict(segment)   5 category probabilities + extras
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from woofdecoder.audio import AudioBranch
from woofdecoder.base_branch import BranchOutput
from woofdecoder.config import load_config
from woofdecoder.preprocess.bark_detector import load_bark_detector, settings_from_config
from woofdecoder.preprocess.bark_segments import build_segments
from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.taxonomy import CATEGORIES


@dataclass
class SegmentResult:
    """The prediction for one detected vocalization."""

    segment: BarkSegment
    output: BranchOutput

    @property
    def category(self) -> str:
        return CATEGORIES[int(self.output.probs.argmax())]

    @property
    def probability(self) -> float:
        return float(self.output.probs.max())

    @property
    def extras(self) -> dict[str, Any]:
        return self.output.extras


class Pipeline:
    """Loads the models once; call run() per clip."""

    def __init__(self, config: dict | None = None, device: str | None = None) -> None:
        if config is None:
            config = load_config()

        # Preprocess
        self.settings = settings_from_config(config)
        self.detector = load_bark_detector(config["bark_detector"]["model"], device, verbose=False)

        # Audio
        self.audio = AudioBranch.from_config(config, device)

    def run(self, path: str | Path, verbose: bool = False) -> list[SegmentResult]:
        segments = build_segments(path, self.detector, self.settings, verbose=verbose)
        return [SegmentResult(segment, self.audio.predict(segment)) for segment in segments]
