"""Clip -> predictions: bark detector, then the audio, body and scene branches, then fusion.

segments = build_segments(clip)                     one BarkSegment per detected dog sound
outputs  = {name: branch.predict(segment)}          audio, body, scene: 5 probabilities each
fused    = fusion.predict(outputs)                  learned fusion head (models/fusion/)
fusion   = fusion.explain(outputs)                  the steps behind `fused`, for the app

Same branches and order as wooftrainer.fusion_trainer.branch_outputs.cache_branch_outputs(),
so a clip gives the same branch outputs here as in the fusion cache.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from woofdecoder.audio import AudioBranch
from woofdecoder.base_branch import Branch, BranchOutput
from woofdecoder.body import BodyBranch
from woofdecoder.config import load_config
from woofdecoder.constants import BRANCH_NAMES
from woofdecoder.fusion import FusionModel
from woofdecoder.preprocess.bark_detector import load_bark_detector, settings_from_config
from woofdecoder.preprocess.bark_segments import build_segments
from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.scene import SceneBranch
from woofdecoder.taxonomy import CATEGORIES

# progress(stage, done, total). Stages: "extracting audio/frames", "detecting barks", then per
# segment "audio", "body", "scene", "fusion" with done = segments finished, total = segments.
Progress = Callable[[str, int, int], None]

# How far the explained fusion probabilities may be from FusionModel.predict(). Both use the
# same LinearHead calls, so they are equal in practice; this only guards against drift.
FUSION_EXPLAIN_TOLERANCE: float = 1e-9


@dataclass
class SegmentResult:
    """Everything the pipeline computed for one detected vocalization.

    Attributes:
        segment: The bark segment (times, detector confidence, sound sections, frames).
        branches: One BranchOutput per name in BRANCH_NAMES.
        fused: The fusion head's BranchOutput (BranchOutput.missing() if no branch ran).
        fusion: FusionModel.explain() for this segment, or None if no branch ran.
        scene: Scene settings and the best prompt per frame (see Pipeline.scene_details).
    """

    segment: BarkSegment
    branches: dict[str, BranchOutput]
    fused: BranchOutput
    fusion: dict[str, Any] | None
    scene: dict[str, Any] = field(default_factory=dict)

    @property
    def output(self) -> BranchOutput:
        return self.fused

    @property
    def category(self) -> str:
        return CATEGORIES[int(self.fused.probs.argmax())]

    @property
    def probability(self) -> float:
        return float(self.fused.probs.max())

    @property
    def extras(self) -> dict[str, Any]:
        return self.fused.extras


def check_scene_schema(fusion: FusionModel, scene: SceneBranch) -> None:
    """Raise if the fusion head was trained on scene outputs made with other settings."""
    if fusion.scene_schema_id != scene.encoder.schema_id:
        raise ValueError(
            "The fusion head (models/fusion) was trained on scene-branch outputs made with other "
            "scene settings than configs/pipeline.yaml branches.scene "
            f"(fusion: {fusion.scene_schema_id[:12]}…, scene now: {scene.encoder.schema_id[:12]}…). "
            "Undo the scene change, or rerun notebooks 50-51 to rebuild the cache and the head."
        )


class Pipeline:
    """Loads every model once; call run() per clip."""

    def __init__(self, config: dict | None = None, device: str | None = None) -> None:
        if config is None:
            config = load_config()

        # Preprocess
        self.settings = settings_from_config(config)
        self.detector = load_bark_detector(config["bark_detector"]["model"], device, verbose=False)

        # Branches (same construction as branch_outputs.load_branches, plus the device)
        self.scene = SceneBranch.from_config(config, device)
        self.branches: dict[str, Branch] = {
            "audio": AudioBranch.from_config(config, device),
            "body": BodyBranch.from_config(config, device),
            "scene": self.scene,
        }

        # Fusion
        self.fusion = FusionModel.from_config(config)
        check_scene_schema(self.fusion, self.scene)

    def scene_details(self, output: BranchOutput) -> dict[str, Any]:
        """Scene settings plus, per frame, the prompt CLIP matched best and its probability."""
        encoder = self.scene.encoder
        frames: list[dict[str, Any]] = []
        frame_times = output.extras.get("frame_times_s", [])
        for time_s, frame_probs in zip(frame_times, output.extras.get("frame_probs", [])):
            best = int(np.argmax(frame_probs))
            frames.append(
                {
                    "time_s": float(time_s),
                    "top_category": CATEGORIES[best],
                    "top_prompt": encoder.prompts[best],
                    "score": float(frame_probs[best]),
                }
            )
        return {
            "aggregation": encoder.aggregation,
            "temperature": encoder.temperature,
            "confidence_weight": encoder.confidence_weight
            if encoder.aggregation == "confidence_weighted_logits"
            else None,
            "schema_id": encoder.schema_id,
            "prompts": dict(zip(CATEGORIES, encoder.prompts, strict=True)),
            "frames": frames,
        }

    def run_segment(
        self,
        segment: BarkSegment,
        progress: Progress | None = None,
        number: int = 0,
        total: int = 1,
    ) -> SegmentResult:
        outputs: dict[str, BranchOutput] = {}
        for name in BRANCH_NAMES:
            if progress is not None:
                progress(name, number, total)
            outputs[name] = self.branches[name].predict(segment)

        if progress is not None:
            progress("fusion", number, total)
        fused = self.fusion.predict(outputs)
        fusion = self.fusion.explain(outputs)
        if fusion is not None and not np.allclose(
            fusion["probs"], fused.probs, rtol=0, atol=FUSION_EXPLAIN_TOLERANCE
        ):
            raise AssertionError("fusion explanation does not match FusionModel.predict()")

        return SegmentResult(segment, outputs, fused, fusion, self.scene_details(outputs["scene"]))

    def run(self, path: str | Path, progress: Progress | None = None) -> list[SegmentResult]:
        segments = build_segments(
            path, self.detector, self.settings, verbose=False, progress=progress
        )
        results: list[SegmentResult] = []
        for number, segment in enumerate(segments):
            results.append(self.run_segment(segment, progress, number, len(segments)))
        return results
