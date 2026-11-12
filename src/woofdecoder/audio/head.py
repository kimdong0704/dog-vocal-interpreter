from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from woofdecoder.audio.features import LAYER_MEAN, Layer, check_layer

FORMAT_VERSION: int = 1

SITUATION_HEAD_FILENAME: str = "situation_head.npz"
AROUSAL_HEAD_FILENAME: str = "arousal_head.npz"
VALENCE_HEAD_FILENAME: str = "valence_head.npz"
NUDGE_FILENAME: str = "emotion_nudge.npz"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def layer_from_text(text: str) -> Layer:
    if text == LAYER_MEAN:
        return LAYER_MEAN
    return int(text)


def check_format(path: Path, data: np.lib.npyio.NpzFile) -> None:
    version = int(data["format_version"])
    if version != FORMAT_VERSION:
        raise ValueError(f"{path} has format version {version}, expected {FORMAT_VERSION}")


@dataclass
class LinearHead:
    """
    Attributes:
        classes: Class names in the order of the weight rows (not necessarily CATEGORIES).
        weights: shape (num_classes, dim).
        bias: shape (num_classes,).
        mean: Per-feature mean used to standardise, shape (dim,).
        scale: Per-feature standard deviation used to standardise, shape (dim,), all > 0.
        layer: Which encoder layer the head reads (select_layer).
        encoder: Name of the encoder whose embeddings the head was trained on.
        temperature: Logits are divided by this before softmax (fitted on out-of-fold data).
    """

    classes: list[str]
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    layer: Layer
    encoder: str
    temperature: float = 1.0

    def __post_init__(self) -> None:
        self.classes = list(self.classes)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.bias = np.asarray(self.bias, dtype=np.float64)
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.scale = np.asarray(self.scale, dtype=np.float64)
        self.temperature = float(self.temperature)

        if self.weights.ndim != 2:
            raise ValueError(f"weights must be (num_classes, dim), got {self.weights.shape}")
        num_classes, dim = self.weights.shape
        if len(self.classes) != num_classes or len(set(self.classes)) != num_classes:
            raise ValueError(f"need {num_classes} different class names, got {self.classes}")
        if self.bias.shape != (num_classes,):
            raise ValueError(f"bias must have shape ({num_classes},), got {self.bias.shape}")
        if self.mean.shape != (dim,) or self.scale.shape != (dim,):
            raise ValueError(f"mean and scale must have shape ({dim},)")
        if not (self.scale > 0).all():
            raise ValueError("every scale value must be > 0")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        check_layer(self.layer)

    @property
    def dim(self) -> int:
        return self.weights.shape[1]

    def logits(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape[-1] != self.dim:
            raise ValueError(f"features must end in {self.dim}, got shape {features.shape}")
        standardized = (features - self.mean) / self.scale
        return standardized @ self.weights.T + self.bias

    def probs_from_logits(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=np.float64) / self.temperature)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.probs_from_logits(self.logits(features))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            format_version=np.array(FORMAT_VERSION),
            classes=np.array(self.classes, dtype=str),
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            scale=self.scale,
            layer=np.array(str(self.layer)),
            encoder=np.array(self.encoder),
            temperature=np.array(self.temperature),
        )

    @classmethod
    def load(cls, path: Path) -> LinearHead:
        with np.load(path, allow_pickle=False) as data:
            check_format(path, data)
            return cls(
                classes=data["classes"].tolist(),
                weights=data["weights"],
                bias=data["bias"],
                mean=data["mean"],
                scale=data["scale"],
                layer=layer_from_text(str(data["layer"])),
                encoder=str(data["encoder"]),
                temperature=float(data["temperature"]),
            )


@dataclass
class EmotionNudge:
    classes: list[str]
    inputs: list[str]
    weights: np.ndarray

    def __post_init__(self) -> None:
        self.classes = list(self.classes)
        self.inputs = list(self.inputs)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        expected_shape = (len(self.classes), len(self.inputs))
        if self.weights.shape != expected_shape:
            raise ValueError(f"weights must have shape {expected_shape}, got {self.weights.shape}")

    def shift(self, emotion_inputs: np.ndarray) -> np.ndarray:
        emotion_inputs = np.asarray(emotion_inputs, dtype=np.float64)
        if emotion_inputs.shape[-1] != len(self.inputs):
            raise ValueError(
                f"emotion inputs must end in {len(self.inputs)}, got {emotion_inputs.shape}"
            )
        return emotion_inputs @ self.weights.T

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            format_version=np.array(FORMAT_VERSION),
            classes=np.array(self.classes, dtype=str),
            inputs=np.array(self.inputs, dtype=str),
            weights=self.weights,
        )

    @classmethod
    def load(cls, path: Path) -> EmotionNudge:
        with np.load(path, allow_pickle=False) as data:
            check_format(path, data)
            return cls(
                classes=data["classes"].tolist(),
                inputs=data["inputs"].tolist(),
                weights=data["weights"],
            )


def emotion_input_names(arousal_head: LinearHead, valence_head: LinearHead) -> list[str]:
    names: list[str] = []
    for target, head in (("arousal", arousal_head), ("valence", valence_head)):
        for class_name in head.classes:
            names.append(f"{target}:{class_name}")
    return names


def emotion_inputs(
    features: np.ndarray, arousal_head: LinearHead, valence_head: LinearHead
) -> np.ndarray:
    return np.concatenate(
        [arousal_head.predict_proba(features), valence_head.predict_proba(features)], axis=-1
    )
