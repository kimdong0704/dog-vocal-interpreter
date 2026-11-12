"""Audio branch: one BarkSegment -> 5 category probabilities, with arousal / valence in extras.

    embedding (12, 768) = encoder.embed(segment)                 (None -> BranchOutput.missing())
    features (768)      = select_layer(embedding, layer)
    emotion (6)         = arousal (3) + valence (3) calibrated probabilities
    logits (5)          = situation.logits(features) + nudge_strength * nudge.shift(emotion)
    probs (5)           = softmax(logits / T_situation), reordered into CATEGORIES order

The four head files are trained in notebook 23 and live in heads_dir. Every check
that could make training and the app disagree (encoder, layer, classes, nudge inputs) runs
once, when the branch is created.

extras["valence"] is close to chance on unseen dogs (notebook 24): the app should not show it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from woofdecoder.audio.encoder import AudioEncoder, encoder_from_config
from woofdecoder.audio.features import Layer, check_layer, select_layer
from woofdecoder.audio.head import (
    AROUSAL_HEAD_FILENAME,
    NUDGE_FILENAME,
    SITUATION_HEAD_FILENAME,
    VALENCE_HEAD_FILENAME,
    EmotionNudge,
    LinearHead,
    emotion_input_names,
    emotion_inputs,
)
from woofdecoder.base_branch import Branch, BranchOutput
from woofdecoder.config import REPO_ROOT, load_config
from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.taxonomy import CATEGORIES, CATEGORY_INDEX, NUM_CATEGORIES

# Where the heads live when pipeline.yaml sets no heads_dir (relative to the repo root).
DEFAULT_HEADS_DIR: str = "models/audio"
# Scales the emotion nudge; 0 turns it off. Must be >= 0
DEFAULT_NUDGE_STRENGTH: float = 1.0


def load_heads(directory: Path) -> tuple[LinearHead, LinearHead, LinearHead, EmotionNudge]:
    paths: list[Path] = []

    for filename in (
        SITUATION_HEAD_FILENAME,
        AROUSAL_HEAD_FILENAME,
        VALENCE_HEAD_FILENAME,
        NUDGE_FILENAME,
    ):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Train and save the audio heads in notebook 23 "
                f"(see configs/pipeline.yaml, branches.audio.heads_dir)."
            )
        paths.append(path)

    situation_path, arousal_path, valence_path, nudge_path = paths

    return (
        LinearHead.load(situation_path),
        LinearHead.load(arousal_path),
        LinearHead.load(valence_path),
        EmotionNudge.load(nudge_path),
    )


def check_heads(
    encoder_name: str,
    layer: Layer,
    situation: LinearHead,
    arousal: LinearHead,
    valence: LinearHead,
    nudge: EmotionNudge,
) -> None:
    # Raise error if heads do not fit the encoder, layer and taxonomy
    for head_name, head in (("situation", situation), ("arousal", arousal), ("valence", valence)):
        if head.encoder != encoder_name:
            raise ValueError(
                f"{head_name} head was trained on encoder {head.encoder!r}, not {encoder_name!r}"
            )
        if head.layer != layer:
            raise ValueError(f"{head_name} head reads layer {head.layer!r}, config says {layer!r}")
    if sorted(situation.classes) != sorted(CATEGORIES):
        raise ValueError(f"situation head classes {situation.classes} are not {CATEGORIES}")
    if nudge.classes != situation.classes:
        raise ValueError(f"nudge rows {nudge.classes} differ from {situation.classes}")
    if nudge.inputs != emotion_input_names(arousal, valence):
        raise ValueError(f"nudge inputs {nudge.inputs} differ from the emotion heads' classes")


def to_categories_order(probs: np.ndarray, classes: list[str]) -> np.ndarray:
    result = np.zeros(NUM_CATEGORIES)

    for position, class_name in enumerate(classes):
        result[CATEGORY_INDEX[class_name]] = probs[position]

    return result


def named(classes: list[str], values: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}

    for position, class_name in enumerate(classes):
        result[class_name] = float(values[position])

    return result


class AudioBranch(Branch):
    name = "audio"

    def __init__(
        self,
        encoder: AudioEncoder,
        situation: LinearHead,
        arousal: LinearHead,
        valence: LinearHead,
        nudge: EmotionNudge,
        layer: Layer,
        nudge_strength: float = DEFAULT_NUDGE_STRENGTH,
    ) -> None:
        check_layer(layer, encoder.num_layers)
        check_heads(encoder.name, layer, situation, arousal, valence, nudge)

        if nudge_strength < 0:
            raise ValueError(f"nudge_strength must be >= 0, got {nudge_strength}")

        self.encoder = encoder
        self.situation = situation
        self.arousal = arousal
        self.valence = valence
        self.nudge = nudge
        self.layer = layer
        self.nudge_strength = float(nudge_strength)

    def predict(self, segment: BarkSegment) -> BranchOutput:

        embedding = self.encoder.embed(segment)  # shape (num_layers, dim), or None
        if embedding is None:
            return BranchOutput.missing()

        features = select_layer(embedding, self.layer)  # shape (dim,)
        emotion = emotion_inputs(features, self.arousal, self.valence)  # shape (6,)
        shift = self.nudge_strength * self.nudge.shift(emotion)  # shape (5,), head order
        logits = self.situation.logits(features) + shift
        head_probs = self.situation.probs_from_logits(logits)  # shape (5,), head order

        probs = to_categories_order(head_probs, self.situation.classes)
        num_arousal = len(self.arousal.classes)
        extras: dict[str, Any] = {
            "top_category": CATEGORIES[int(probs.argmax())],
            "arousal": named(self.arousal.classes, emotion[:num_arousal]),
            "valence": named(self.valence.classes, emotion[num_arousal:]),
            "nudge": named(CATEGORIES, to_categories_order(shift, self.situation.classes)),
        }
        return BranchOutput(probs=probs, extras=extras)

    @classmethod
    def from_config(cls, config: dict | None = None, device: str | None = None) -> AudioBranch:
        if config is None:
            config = load_config()

        section = config["branches"]["audio"]

        layer = section.get("layer")
        if layer is None:
            raise ValueError("configs/pipeline.yaml branches.audio.layer is not set (notebook 20)")

        encoder = encoder_from_config(config, device)
        heads_dir = section.get("heads_dir") or DEFAULT_HEADS_DIR.format(encoder=encoder.name)
        situation, arousal, valence, nudge = load_heads(REPO_ROOT / heads_dir)
        nudge_strength = section.get("nudge_strength", DEFAULT_NUDGE_STRENGTH)

        return cls(encoder, situation, arousal, valence, nudge, layer, nudge_strength)
