from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from woofdecoder.audio.head import LinearHead
from woofdecoder.base_branch import BranchOutput
from woofdecoder.config import REPO_ROOT, load_config
from woofdecoder.constants import (
    BRANCH_NAMES,
    FEATURE_NAMES,
    FUSION_HEAD_FILENAME,
    FUSION_INFO_FILENAME,
    PROB_FLOOR,
)
from woofdecoder.taxonomy import CATEGORIES, NUM_CATEGORIES

# features (18) = for audio, body, scene: present * centred log-probabilities (5 each) + 3 present flags
# probs (5) = softmax((W @ standardised features + b) / T)   (a LinearHead)


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

    def explain(self, outputs: dict[str, BranchOutput]) -> dict[str, Any] | None:
        """Step-by-step view of predict() for one segment, or None if no branch is present.

        Every array is in CATEGORIES order. logits = bias + sum of contributions[branch], where a
        branch's contribution is W @ standardised features over its 5 inputs and present flag.
        probs is computed with the same head calls as predict(), so it equals predict().probs.
        """
        probs = {name: outputs[name].probs for name in BRANCH_NAMES}
        present = {name: bool(outputs[name].present) for name in BRANCH_NAMES}
        if not any(present.values()):
            return None

        head = self.head
        raw = fusion_features(probs, present)  # shape (18,)
        standardized = (raw - head.mean) / head.scale  # shape (18,)
        logits = head.logits(raw)  # shape (5,), head order
        head_probs = head.probs_from_logits(logits)

        contributions: dict[str, np.ndarray] = {}
        for position, name in enumerate(BRANCH_NAMES):
            start = position * NUM_CATEGORIES
            flag_column = len(BRANCH_NAMES) * NUM_CATEGORIES + position
            columns = [*range(start, start + NUM_CATEGORIES), flag_column]
            contributions[name] = head_to_categories(
                head.weights[:, columns] @ standardized[columns], head.classes
            )

        return {
            "features": [
                {"name": feature, "raw": float(r), "standardized": float(z)}
                for feature, r, z in zip(FEATURE_NAMES, raw, standardized, strict=True)
            ],
            "present": present,
            "bias": head_to_categories(head.bias, head.classes),
            "contributions": contributions,
            "logits": head_to_categories(logits, head.classes),
            "temperature": head.temperature,
            "scaled_logits": head_to_categories(logits / head.temperature, head.classes),
            "probs": head_to_categories(head_probs, head.classes),
        }

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

    @classmethod
    def from_config(cls, config: dict | None = None) -> FusionModel:
        """Load the model from the folder in configs/pipeline.yaml (fusion.dir).

        Raises:
            FileNotFoundError: If the folder has no saved fusion model.
        """
        if config is None:
            config = load_config()
        directory = REPO_ROOT / config["fusion"]["dir"]
        if not (directory / FUSION_INFO_FILENAME).is_file():
            raise FileNotFoundError(
                f"no fusion model in {directory}; re-run notebook 51 "
                "(or fix fusion.dir in configs/pipeline.yaml)"
            )
        return cls.load(directory)
