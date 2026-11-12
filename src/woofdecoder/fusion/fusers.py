from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from woofdecoder.audio.head import LinearHead
from woofdecoder.base_branch import BranchOutput
from woofdecoder.taxonomy import CATEGORIES, NUM_CATEGORIES

# features (18) = for audio, body, scene: present * centred log-probabilities (5 each) + 3 present flags
# probs (5) = softmax((W @ standardised features + b) / T)   (a LinearHead)

BRANCH_NAMES: tuple[str, ...] = ("audio", "body", "scene")
PROB_FLOOR: float = 1e-6  # so log(0) cannot occur
FUSION_HEAD_FILENAME: str = "fusion_head.npz"
FUSION_INFO_FILENAME: str = "fusion_info.json"
# LinearHead stores which encoder and layer it reads; for fusion these are fixed labels.
FUSION_ENCODER: str = "fusion_v1"
FUSION_LAYER: str = "mean"

FEATURE_NAMES: list[str] = []
for _branch in BRANCH_NAMES:
    for _category in CATEGORIES:
        FEATURE_NAMES.append(f"{_branch}:{_category}")
for _branch in BRANCH_NAMES:
    FEATURE_NAMES.append(f"{_branch}:present")


def fusion_features(probs: dict[str, np.ndarray], present: dict[str, bool]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for name in BRANCH_NAMES:
        if present[name]:
            log_probs = np.log(np.clip(np.asarray(probs[name], dtype=float), PROB_FLOOR, 1.0))
            # Centered log-ratio (CLR) for removing constants that carry no information
            # Centering the logits
            parts.append(log_probs - log_probs.mean())
        else:
            parts.append(np.zeros(NUM_CATEGORIES))

    flags: list[float] = []
    for name in BRANCH_NAMES:
        flags.append(1.0 if present[name] else 0.0)

    parts.append(np.array(flags))

    return np.concatenate(parts)  # shape (18,)


def head_to_categories(probs: np.ndarray, classes: list[str]) -> np.ndarray:
    columns: list[int] = []
    for category in CATEGORIES:
        columns.append(classes.index(category))

    return np.asarray(probs)[..., columns]


@dataclass
class FusionModel:
    head: LinearHead
    scene_schema_id: str

    def predict(self, outputs: dict[str, BranchOutput]) -> BranchOutput:
        probs: dict[str, np.ndarray] = {}
        present: dict[str, bool] = {}
        used: list[str] = []

        for name in BRANCH_NAMES:
            probs[name] = outputs[name].probs
            present[name] = bool(outputs[name].present)
            if present[name]:
                used.append(name)

        if not used:
            return BranchOutput.missing()

        head_probs = self.head.predict_proba(fusion_features(probs, present))
        fused = head_to_categories(head_probs, self.head.classes)
        return BranchOutput(
            probs=fused,
            extras={"top_category": CATEGORIES[int(fused.argmax())], "branches_used": used},
        )

    def save(self, directory: Path) -> None:
        """Write fusion_head.npz and fusion_info.json into directory."""
        directory.mkdir(parents=True, exist_ok=True)
        self.head.save(directory / FUSION_HEAD_FILENAME)
        info = {
            "scene_schema_id": self.scene_schema_id,
            "branches": list(BRANCH_NAMES),
            "features": FEATURE_NAMES,
        }
        (directory / FUSION_INFO_FILENAME).write_text(json.dumps(info, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> FusionModel:
        """Read a model written by save().

        Raises:
            ValueError: If the saved feature list differs from FEATURE_NAMES.
        """
        info = json.loads((directory / FUSION_INFO_FILENAME).read_text(encoding="utf-8"))
        if info["features"] != FEATURE_NAMES:
            raise ValueError(f"{directory} was trained with other fusion inputs")
        return cls(LinearHead.load(directory / FUSION_HEAD_FILENAME), info["scene_schema_id"])
