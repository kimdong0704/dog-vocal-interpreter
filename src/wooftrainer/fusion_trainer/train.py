from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from woofdecoder.audio.head import softmax
from woofdecoder.fusion.fusers import (
    BRANCH_NAMES,
    FEATURE_NAMES,
    FUSION_ENCODER,
    FUSION_LAYER,
    FusionModel,
    fusion_features,
    head_to_categories,
)
from woofdecoder.taxonomy import CATEGORIES
from wooftrainer.audio_trainer.audio_heads import (
    balanced_weights,
    choose_C,
    fit_linear_head,
    fit_temperature,
    make_folds,
    out_of_fold_logits,
)

TRAINVAL_SPLITS: tuple[str, ...] = ("train", "val")
FUSION_C_VALUES: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

# Which of the 18 inputs each learned method may use.
METHOD_COLUMNS: dict[str, list[int]] = {}
for _branch in BRANCH_NAMES:
    _columns: list[int] = []
    for _index, _name in enumerate(FEATURE_NAMES):
        if _name.startswith(f"{_branch}:"):
            _columns.append(_index)
    METHOD_COLUMNS[f"{_branch} learned"] = _columns
METHOD_COLUMNS["fusion learned"] = list(range(len(FEATURE_NAMES)))


def branch_probs(segments: pd.DataFrame, name: str) -> np.ndarray:
    columns: list[str] = []

    for category in CATEGORIES:
        columns.append(f"{name}_{category}")

    return segments[columns].to_numpy(dtype=float)


def features_from_cache(segments: pd.DataFrame) -> np.ndarray:
    all_probs: dict[str, np.ndarray] = {}

    for name in BRANCH_NAMES:
        all_probs[name] = branch_probs(segments, name)

    rows: list[np.ndarray] = []
    for position in range(len(segments)):
        probs: dict[str, np.ndarray] = {}
        present: dict[str, bool] = {}

        for name in BRANCH_NAMES:
            probs[name] = all_probs[name][position]
            present[name] = bool(segments[f"{name}_present"].iloc[position])

        rows.append(fusion_features(probs, present))

    # (n, 18)
    return np.vstack(rows)


def video_weights(segments: pd.DataFrame) -> np.ndarray:
    # video weights = 1/segments in that video
    counts = segments.groupby("video_id")["video_id"].transform("size")
    return 1.0 / counts.to_numpy(dtype=float)


def raw_average(segments: pd.DataFrame) -> np.ndarray:
    total = np.zeros((len(segments), len(CATEGORIES)))
    count = np.zeros((len(segments), 1))

    for name in BRANCH_NAMES:
        present = segments[f"{name}_present"].to_numpy(dtype=bool)[:, np.newaxis]
        total += branch_probs(segments, name) * present
        count += present

    uniform = np.full_like(total, 1.0 / len(CATEGORIES))

    return np.where(count > 0, total / np.maximum(count, 1), uniform)


def sorted_columns(probs: np.ndarray) -> np.ndarray:
    """Reorder probabilities from CATEGORIES order into alphabetical order.

    sklearn's log_loss assumes its probability columns are in sorted label order.
    """
    columns: list[int] = []
    for category in sorted(CATEGORIES):
        columns.append(CATEGORIES.index(category))
    return probs[:, columns]


def score(probs: np.ndarray, segments: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    labels = segments["category"].to_numpy()
    predicted = np.array(CATEGORIES)[probs.argmax(axis=1)]
    clipped = np.clip(probs, 1e-12, 1.0)
    per_video = pd.DataFrame(probs, columns=CATEGORIES)
    per_video["video_id"] = segments["video_id"].to_numpy()
    per_video["category"] = labels
    video_means = per_video.groupby("video_id").agg(
        {**dict.fromkeys(CATEGORIES, "mean"), "category": "first"}
    )
    video_predicted = video_means[CATEGORIES].idxmax(axis=1)
    return {
        "segment_log_loss": float(
            log_loss(
                labels,
                sorted_columns(clipped / clipped.sum(axis=1, keepdims=True)),
                labels=sorted(CATEGORIES),
                sample_weight=weights,
            )
        ),
        "segment_macro_f1": float(
            f1_score(labels, predicted, labels=CATEGORIES, average="macro", sample_weight=weights)
        ),
        "video_macro_f1": float(
            f1_score(video_means["category"], video_predicted, labels=CATEGORIES, average="macro")
        ),
        "video_accuracy": float((video_predicted == video_means["category"]).mean()),
    }


@dataclass
class LearnedMethod:
    name: str
    columns: list[int]
    C: float
    temperature: float
    oof_probs: np.ndarray
    c_table: pd.DataFrame


def learn_method(
    name: str, features: np.ndarray, labels: np.ndarray, weights: np.ndarray, folds: list
) -> LearnedMethod:
    columns = METHOD_COLUMNS[name]
    subset = features[:, columns]
    best_C, c_table = choose_C(subset, labels, weights, folds, FUSION_C_VALUES)
    logits, classes = out_of_fold_logits(subset, labels, weights, folds, best_C)
    temperature = fit_temperature(logits, labels, classes, weights)
    oof_probs = head_to_categories(softmax(logits / temperature), classes)

    return LearnedMethod(name, columns, best_C, temperature, oof_probs, c_table)


def compare_methods(
    segments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LearnedMethod], pd.DataFrame]:
    trainval = segments[segments["split"].isin(TRAINVAL_SPLITS)].reset_index(drop=True)
    labels = trainval["category"].to_numpy()
    weights = balanced_weights(labels, video_weights(trainval))
    folds = make_folds(labels, groups=trainval["video_id"].to_numpy())
    features = features_from_cache(trainval)

    rows: list[dict] = []
    for name in BRANCH_NAMES:
        rows.append(
            {"method": f"{name} raw", **score(branch_probs(trainval, name), trainval, weights)}
        )
    rows.append({"method": "average raw", **score(raw_average(trainval), trainval, weights)})

    learned: dict[str, LearnedMethod] = {}
    for name in METHOD_COLUMNS:
        method = learn_method(name, features, labels, weights, folds)
        learned[name] = method
        result = {"method": name, "C": method.C, "temperature": method.temperature}
        result.update(score(method.oof_probs, trainval, weights))
        rows.append(result)
    return pd.DataFrame(rows), learned, trainval


def fit_fusion(segments: pd.DataFrame, method: LearnedMethod, scene_schema_id: str) -> FusionModel:
    if method.columns != list(range(len(FEATURE_NAMES))):
        raise ValueError("only 'fusion learned' (all inputs) can be saved as the fusion model")

    trainval = segments[segments["split"].isin(TRAINVAL_SPLITS)].reset_index(drop=True)
    labels = trainval["category"].to_numpy()
    weights = balanced_weights(labels, video_weights(trainval))
    head = fit_linear_head(
        features_from_cache(trainval), labels, weights, method.C, FUSION_LAYER, FUSION_ENCODER
    )
    head.temperature = method.temperature

    return FusionModel(head, scene_schema_id)
