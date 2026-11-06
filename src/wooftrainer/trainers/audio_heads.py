from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar, minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from woofdecoder.branches.audio.features import LAYER_MEAN, Layer, select_layer
from woofdecoder.branches.audio.head import (
    AROUSAL_HEAD_FILENAME,
    NUDGE_FILENAME,
    SITUATION_HEAD_FILENAME,
    VALENCE_HEAD_FILENAME,
    EmotionNudge,
    LinearHead,
    emotion_input_names,
    emotion_inputs,
    softmax,
)

from wooftrainer.trainers.embeddings import attach_labels, load_embeddings

DEFAULT_C_VALUES: tuple[float, ...] = (0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0)
NUM_FOLDS: int = 5
SEED: int = 0
MAX_ITER: int = 3000
TEMPERATURE_BOUNDS: tuple[float, float] = (0.25, 10.0)

MISSING_CLASS_LOGIT: float = -30.0

DEFAULT_L2_VALUES: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
NO_NUDGE: str = "none"  # the "no nudge" row of the nudge table
MIN_NUDGE_GAIN: float = 0.01
EMOTION_TASK: str = "audio_emotion"
SITUATION_TASK: str = "audio_context"
TRAIN_SPLIT: str = "train"

Folds = list[tuple[np.ndarray, np.ndarray]]

def clip_weights(table: pd.DataFrame) -> np.ndarray:
    if "num_windows" not in table.columns:
        return np.ones(len(table))
    num_windows = pd.to_numeric(table["num_windows"], errors="coerce").fillna(1.0)
    return 1.0 / num_windows.to_numpy(dtype=float)

def balanced_weights(labels: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    sample_weight = np.asarray(sample_weight, dtype=float)
    classes = np.unique(labels)
    total = sample_weight.sum()
    weights = sample_weight.copy()
    for class_name in classes:
        in_class = labels == class_name
        weights[in_class] *= total / (len(classes) * sample_weight[in_class].sum())
    return weights

def fit_linear_head(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    C: float,
    layer: Layer,
    encoder: str
) -> LinearHead:
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=C, max_iter=MAX_ITER)
    model.fit(scaler.transform(features), labels, sample_weight=weights)

    coefficients = model.coef_
    intercepts = model.intercept_
    if len(model.classes_) == 2:
        coefficients = np.vstack([-coefficients[0]/2], coefficients[0]/2)
        intercepts = np.array([-intercepts[0]/2], intercepts[0]/2)
    return LinearHead(
        classes=model.classes_.tolist(),
        weights=coefficients,
        bias=intercepts,
        mean=scaler.mean_,
        scale=scaler.scale_,
        layer=layer,
        encoder=encoder
    )
    
def make_folds(
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    num_folds: int = NUM_FOLDS,
    seed: int = SEED
) -> Folds:
    placeholder = np.zeros(len(labels))
    if groups is None:
        splitter = StratifiedKFold(num_folds, shuffle=True, random_state=seed)
        return list(splitter.split(placeholder, labels))
    
    group_splitter = StratifiedGroupKFold(num_folds, shuffle=True, random_state=seed)
    return list(group_splitter.split(placeholder, labels, groups))

def out_of_fold_logits(
    features: np.ndarray, 
    labels: np.ndarray,
    weights: np.ndarray,
    folds: Folds,
    C: float
) -> tuple[np.ndarray, list[str]]:
    classes = np.unique(labels).tolist()
    logits = np.full((len(labels), len(classes)), MISSING_CLASS_LOGIT)
    for train_index, held_out_index in folds:
        head = fit_linear_head(
            features[train_index], 
            labels[train_index], 
            weights[train_index], 
            C, 
            layer=LAYER_MEAN, 
            encoder="fold"
        )
        fold_logits = head.logits(features[held_out_index])
        for column, class_name in enumerate(head.classes):
            logits[held_out_index, classes.index(class_name)] = fold_logits[:, column]
    return logits, classes

def score_probs(
    labels: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    weights: np.ndarray
) -> dict[str, float]:
    predicted = np.asarray(classes)[probs.argmax(axis=1)]
    return {
        "log_loss": float(log_loss(labels, probs, labels=classes, sample_weight=weights)),
        "macro_f1": float(f1_score(labels, predicted, labels=classes, average="macro", sample_weight=weights))
    }
    
def choose_C(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    folds: Folds,
    C_values: tuple[float, ...] = DEFAULT_C_VALUES
) -> tuple[float, pd.DataFrame]:
    results: list[dict[str, float]] = []
    for C in C_values:
        logits, classes = out_of_fold_logits(features, labels, weights, folds, C)
        result = score_probs(labels, softmax(logits), classes, weights)
        result["C"] = C
        results.append(result)
    table = pd.DataFrame(results)
    best_C = float(table.loc[table["log_loss"].idxmin(), "C"])
    return best_C, table

def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    weights: np.ndarray
) -> float:
    
    def loss_at(temperature: float) -> float:
        probs = softmax(logits/temperature)
        return float(log_loss(labels, probs, labels=classes, sample_weight=weights))
    
    result = minimize_scalar(loss_at, bounds=TEMPERATURE_BOUNDS, method="bounded")
    temperature = float(result.x)
    low, high = TEMPERATURE_BOUNDS
    if temperature < low * 1.01 or temperature > high * 0.99:
        print(f"Temperature {temperature:.2f} is at the search bound {TEMPERATURE_BOUNDS}")
    return temperature


# --- Emotion nudge ---

def one_hot(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    targets = np.zeros((len(labels), len(classes)))
    for row, label in enumerate(labels):
        targets[row, classes.index(label)] = 1.0
    return targets

# gradient descent for nudge weights
def fit_nudge_weights(
    logits: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    weights: np.ndarray,
    l2: float,
) -> np.ndarray:
    targets = one_hot(labels, classes)
    weight_total = weights.sum()
    num_classes = len(classes)
    num_inputs = inputs.shape[1]
 
    def loss_and_gradient(flat_weights: np.ndarray) -> tuple[float, np.ndarray]:
        nudge_weights = flat_weights.reshape(num_classes, num_inputs)
        probs = softmax(logits + inputs @ nudge_weights.T) # shape (n, num_classes)
        
        true_class_probs = np.clip((probs * targets).sum(axis=1), 1e-12, None) # prediction number for true class
        # -mean(log_likelihood) --> cross-entropy loss
        # weights are flattened weights (n, )
        # weights are clip weight * class weight, so we divide by weight_total to get the mean
        loss = -(weights * np.log(true_class_probs)).sum() / weight_total 
        loss += l2 * (nudge_weights**2).sum() # punish on large weights to avoid overfitting
        
        # Single layer gradient: dL/dW = (probs - targets) * inputs, dz
        logit_gradient = (probs - targets) * weights[:, np.newaxis] / weight_total
        # delta
        gradient = logit_gradient.T @ inputs + 2 * l2 * nudge_weights
        
        return float(loss), gradient.ravel()
 
    # nudge weights initialized to zero, shape (num_classes, num_inputs)
    start = np.zeros(num_classes * num_inputs)
    # L-BFGS-B is a plain gradient descent optimizer (like Adam)
    result = minimize(loss_and_gradient, start, jac=True, method="L-BFGS-B")
    return result.x.reshape(num_classes, num_inputs)
 
# training the nudge weights on the training folds
def out_of_fold_nudged_logits(
    logits: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    weights: np.ndarray,
    folds: Folds,
    l2: float,
) -> np.ndarray:
    nudged = logits.copy()
    # Full batch gradient descent on each fold
    for train_index, held_out_index in folds:
        nudge_weights = fit_nudge_weights(
            logits[train_index],
            inputs[train_index],
            labels[train_index],
            classes,
            weights[train_index],
            l2,
        )
        nudged[held_out_index] = logits[held_out_index] + inputs[held_out_index] @ nudge_weights.T
    return nudged
 
 
def choose_nudge_l2(
    logits: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    weights: np.ndarray,
    folds: Folds,
    l2_values: tuple[float, ...] = DEFAULT_L2_VALUES,
) -> tuple[float | None, pd.DataFrame]:
    results: list[dict] = []
    
    no_nudge = score_probs(labels, softmax(logits), classes, weights)
    no_nudge["l2"] = NO_NUDGE
    results.append(no_nudge)
    
    for l2 in l2_values:
        nudged = out_of_fold_nudged_logits(logits, inputs, labels, classes, weights, folds, l2)
        result = score_probs(labels, softmax(nudged), classes, weights)
        result["l2"] = l2
        results.append(result)
    
    table = pd.DataFrame(results)
    nudged = table.loc[table["l2"] != NO_NUDGE]
    best = nudged.loc[nudged["log_loss"].idxmin()]
    gain = no_nudge["log_loss"] - best["log_loss"]
    if gain < MIN_NUDGE_GAIN or best["macro_f1"] < no_nudge["macro_f1"]:
        return None, table
    return float(best["l2"]), table
 
 
# --- Training all heads ---
 
 
@dataclass
class AudioHeads:
    """
    Attributes:
        situation: The 5-category head; its temperature applies AFTER the nudge.
        arousal: Arousal head with its own temperature.
        valence: Valence head with its own temperature.
        nudge: The emotion nudge (all zeros if it did not help).
        summary: One row per head: chosen C (and l2), temperature, out-of-fold scores.
        tables: The C search tables per head and the nudge table, for the notebook.
        situation_oof_probs: Final out-of-fold situation probabilities (after nudge and
            temperature), shape (n, 5), columns in situation.classes order.
        situation_rows: clip_id, source, dog_id, mapped_category of those rows.
    """
 
    situation: LinearHead
    arousal: LinearHead
    valence: LinearHead
    nudge: EmotionNudge
    summary: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    situation_oof_probs: np.ndarray | None = None
    situation_rows: pd.DataFrame | None = None
 
 
def prefixed_scores(prefix: str, scores: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in scores.items():
        result[f"{prefix}_{key}"] = value
    return result
 
def train_emotion_head(
    embeddings: np.ndarray,
    table: pd.DataFrame,
    target: str,
    layer: Layer,
    encoder: str,
    C_values: tuple[float, ...],
) -> tuple[LinearHead, pd.DataFrame, dict]:
    features = select_layer(embeddings, layer)
    labels = table[target].to_numpy()
    weights = balanced_weights(labels, clip_weights(table))
    folds = make_folds(labels)
 
    best_C, c_table = choose_C(features, labels, weights, folds, C_values)
    logits, classes = out_of_fold_logits(features, labels, weights, folds, best_C)
    temperature = fit_temperature(logits, labels, classes, weights)
 
    head = fit_linear_head(features, labels, weights, best_C, layer, encoder)
    head.temperature = temperature
    summary = {"head": target, "C": best_C, "l2": "", "temperature": temperature}
    summary.update(prefixed_scores("before", score_probs(labels, softmax(logits), classes, weights)))
    after = softmax(logits / temperature)
    summary.update(prefixed_scores("after", score_probs(labels, after, classes, weights)))
    return head, c_table, summary
 
def train_audio_heads(
    encoder_name: str,
    layer: Layer,
    C_values: tuple[float, ...] = DEFAULT_C_VALUES,
    l2_values: tuple[float, ...] = DEFAULT_L2_VALUES,
) -> AudioHeads:
    # Encoder -> embeddings for emotional heads
    emotion_x, emotion_table = load_embeddings(EMOTION_TASK, TRAIN_SPLIT, encoder_name)
    emotion_table = attach_labels(emotion_table, EMOTION_TASK, TRAIN_SPLIT)
    
    # Emotional heads
    # Train arousal head 
    arousal, arousal_c, arousal_summary = train_emotion_head(
        emotion_x, emotion_table, "arousal", layer, encoder_name, C_values
    )
    # Train valence head
    valence, valence_c, valence_summary = train_emotion_head(
        emotion_x, emotion_table, "valence", layer, encoder_name, C_values
    )
 
    # Encoder -> embeddings for situation head
    context_x, context_table = load_embeddings(SITUATION_TASK, TRAIN_SPLIT, encoder_name)
    context_table = attach_labels(context_table, SITUATION_TASK, TRAIN_SPLIT)
    labels = context_table["mapped_category"].to_numpy()
    if (labels == "").any():
        raise ValueError(f"{int((labels == '').sum())} {SITUATION_TASK} rows have no category")
    
    features = select_layer(context_x, layer)
    weights = balanced_weights(labels, clip_weights(context_table))
    folds = make_folds(labels, groups=context_table["dog_id"].to_numpy())
 
    best_C, situation_c = choose_C(features, labels, weights, folds, C_values)
    logits, classes = out_of_fold_logits(features, labels, weights, folds, best_C)
    inputs = emotion_inputs(features, arousal, valence)  # shape (n, 6)
 
    best_l2, nudge_table = choose_nudge_l2(
        logits, inputs, labels, classes, weights, folds, l2_values
    )
    if best_l2 is None:
        final_logits = logits
        nudge_weights = np.zeros((len(classes), inputs.shape[1]))
    else:
        final_logits = out_of_fold_nudged_logits(
            logits, inputs, labels, classes, weights, folds, best_l2
        )
        nudge_weights = fit_nudge_weights(logits, inputs, labels, classes, weights, best_l2)
    temperature = fit_temperature(final_logits, labels, classes, weights)
 
    situation = fit_linear_head(features, labels, weights, best_C, layer, encoder_name)
    situation.temperature = temperature
    nudge = EmotionNudge(classes, emotion_input_names(arousal, valence), nudge_weights)
 
    situation_summary = {
        "head": "situation",
        "C": best_C,
        "l2": NO_NUDGE if best_l2 is None else best_l2,
        "temperature": temperature,
    }
    before = score_probs(labels, softmax(logits), classes, weights)
    final_probs = softmax(final_logits / temperature)
    situation_summary.update(prefixed_scores("before", before))
    situation_summary.update(
        prefixed_scores("after", score_probs(labels, final_probs, classes, weights))
    )
 
    return AudioHeads(
        situation=situation,
        arousal=arousal,
        valence=valence,
        nudge=nudge,
        summary=pd.DataFrame([arousal_summary, valence_summary, situation_summary]),
        tables={
            "C arousal": arousal_c,
            "C valence": valence_c,
            "C situation": situation_c,
            "nudge": nudge_table,
        },
        situation_oof_probs=final_probs,
        situation_rows=context_table[["clip_id", "source", "dog_id", "mapped_category"]],
    )
 
 
def save_audio_heads(heads: AudioHeads, directory: Path) -> None:
    heads.situation.save(directory / SITUATION_HEAD_FILENAME)
    heads.arousal.save(directory / AROUSAL_HEAD_FILENAME)
    heads.valence.save(directory / VALENCE_HEAD_FILENAME)
    heads.nudge.save(directory / NUDGE_FILENAME)