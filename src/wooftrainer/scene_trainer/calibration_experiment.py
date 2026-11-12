"""Exploratory, group-held-out CLIP score centering and temperature comparison.

Inputs are CLIP's *native scaled logits*, not raw cosine similarities. This is
an offline experiment; it does not change the scene branch or write model files.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.model_selection import StratifiedGroupKFold

from woofdecoder.taxonomy import CATEGORIES
from wooftrainer.settings import (
    SCENE_BOOTSTRAP_REPEATS,
    SCENE_BOOTSTRAP_SEED,
    SCENE_CV_FOLDS,
    SCENE_CV_SEED,
)

METHODS = (
    "current_weighted",
    "mean_logits",
    "bias_only",
    "temperature_only",
    "bias_and_temperature",
)


def canonical_source_id(value):
    """Normalize YouTube links while keeping non-YouTube sources distinct."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Each segment needs a nonempty source URL or identifier")
    source = value.strip()
    parsed = urlparse(source)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    video = None
    if host == "youtu.be" and parts:
        video = parts[0]
    elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        if parsed.path.rstrip("/") == "/watch":
            video = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
            video = parts[1]
    return f"youtube:{video}" if video else source


def connected_groups(dog_ids, source_urls, linked_group_ids=None):
    """Group transitive dog/source links and optional groups computed from all labels."""
    dogs, sources = list(dog_ids), list(source_urls)
    if not dogs or len(dogs) != len(sources):
        raise ValueError("Supply equal, nonempty dog and source lists")
    parent = list(range(len(dogs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    linked = None if linked_group_ids is None else list(linked_group_ids)
    if linked is not None and len(linked) != len(dogs):
        raise ValueError("Supply one linked group ID per segment")
    by_dog, by_source, by_link = {}, {}, {}
    for i, (dog, source) in enumerate(zip(dogs, sources, strict=True)):
        if not isinstance(dog, str) or not dog.strip():
            raise ValueError("Each segment needs a nonempty dog ID")
        dog = dog.strip().casefold()
        source = canonical_source_id(source)
        if dog in by_dog:
            union(i, by_dog[dog])
        else:
            by_dog[dog] = i
        if source in by_source:
            union(i, by_source[source])
        else:
            by_source[source] = i
        if linked is not None:
            group = linked[i]
            if (
                isinstance(group, bool)
                or not isinstance(group, (str, int, np.integer))
                or not str(group).strip()
            ):
                raise ValueError("Each linked group ID must be a nonempty string or integer")
            group = str(group).strip()
            if group in by_link:
                union(i, by_link[group])
            else:
                by_link[group] = i
    return np.array([find(i) for i in range(len(dogs))], dtype=int)


def _validate_ids(ids, name, size):
    values = list(ids)
    if len(values) != size or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"Each segment needs a nonempty {name}")
    return np.array([x.strip() for x in values], dtype=object)


def _validated(logits, labels, span_ids):
    frames = [np.asarray(x, dtype=np.float64) for x in logits]
    if not frames:
        raise ValueError("Supply at least one segment")
    if any(
        x.ndim != 2 or x.shape[1] != len(CATEGORIES) or not len(x) or not np.isfinite(x).all()
        for x in frames
    ):
        raise ValueError("Each segment needs finite frame logits with shape (frames, 5)")
    names = list(labels)
    if len(names) != len(frames) or any(x not in CATEGORIES for x in names):
        raise ValueError("Labels must use the shared five-category taxonomy")
    y = np.array([CATEGORIES.index(x) for x in names], dtype=int)
    spans = _validate_ids(span_ids, "span ID", len(frames))
    for span in dict.fromkeys(spans):
        if len(np.unique(y[spans == span])) != 1:
            raise ValueError(f"Span {span!r} has conflicting labels")
    return frames, y, spans


def _span_means(values, spans):
    unique = list(dict.fromkeys(spans))
    return np.stack([values[spans == span].mean(axis=0) for span in unique]), unique


def fit_class_bias(mean_logits, labels, span_ids):
    """Estimate class-balanced mean logits using equally weighted labeled spans."""
    x = np.asarray(mean_logits, dtype=np.float64)
    labels, spans = np.asarray(labels), np.asarray(span_ids)
    if x.ndim != 2 or x.shape[1] != len(CATEGORIES) or not np.isfinite(x).all():
        raise ValueError("Expected finite mean logits of shape (segments, 5)")
    if labels.shape != (len(x),) or spans.shape != (len(x),):
        raise ValueError("One label and span ID are required per segment")
    if set(labels.tolist()) != set(range(len(CATEGORIES))):
        raise ValueError("Each training fold must contain all five classes")
    means, span_list = _span_means(x, spans)
    span_labels = []
    for span in span_list:
        unique_labels = np.unique(labels[spans == span])
        if len(unique_labels) != 1:
            raise ValueError("Each labeled span must have exactly one category")
        span_labels.append(unique_labels[0])
    span_labels = np.asarray(span_labels)
    return np.stack([means[span_labels == k].mean(axis=0) for k in range(len(CATEGORIES))]).mean(0)


def fit_temperature(mean_logits, labels, span_ids, bounds=(-6.0, 6.0)):
    """Fit one positive T by training-fold log loss, equally weighting spans."""
    x = np.asarray(mean_logits, dtype=np.float64)
    y, spans = np.asarray(labels), np.asarray(span_ids)
    if (
        x.ndim != 2
        or x.shape[1] != len(CATEGORIES)
        or not np.isfinite(x).all()
        or y.shape != (len(x),)
        or spans.shape != (len(x),)
        or set(y.tolist()) != set(range(len(CATEGORIES)))
    ):
        raise ValueError("Temperature fitting requires finite logits and all five categories")
    if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("Log-temperature bounds must be finite and increasing")

    def objective(log_temperature):
        scaled = x / np.exp(log_temperature)
        losses = logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y.astype(int)]
        return float(_span_means(losses, spans)[0].mean())

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature optimization did not converge")
    return float(np.exp(result.x))


def cross_validate(
    frame_logits,
    labels,
    span_ids,
    dog_ids,
    source_urls,
    *,
    n_splits=SCENE_CV_FOLDS,
    alpha=0.5,
    baseline_temperature=1.0,
    random_state=SCENE_CV_SEED,
    linked_group_ids=None,
):
    """Evaluate each variant once per held-out segment; never fit on held-out groups.

    `span_ids` identify annotation spans, not event IDs. Repeated barks from a span
    receive equal total weight in fitting. `dog_ids` and `source_urls` are linked
    transitively before splitting at span level. Compute `linked_group_ids` from
    all valid label rows before excluding no-event videos or limiting samples; this
    preserves links through a video that produced no scored event.
    """
    frames, y, spans = _validated(frame_logits, labels, span_ids)
    size = len(frames)
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, int)
        or n_splits < 2
        or not np.isfinite(alpha)
        or not 0 <= alpha <= 1
        or not np.isfinite(baseline_temperature)
        or baseline_temperature <= 0
    ):
        raise ValueError("Invalid fold count, confidence weight, or baseline temperature")
    dogs = _validate_ids(dog_ids, "dog ID", size)
    sources = _validate_ids(source_urls, "source URL", size)
    groups = connected_groups(dogs, sources, linked_group_ids)
    span_names = list(dict.fromkeys(spans))
    span_y, span_group = [], []
    for span in span_names:
        mask = spans == span
        label_values, group_values = np.unique(y[mask]), np.unique(groups[mask])
        if len(label_values) != 1 or len(group_values) != 1:
            raise ValueError(f"Span {span!r} must share one label and one dog/source group")
        span_y.append(label_values[0])
        span_group.append(group_values[0])
    span_y, span_group = np.asarray(span_y), np.asarray(span_group)
    if len(np.unique(span_group)) < n_splits:
        raise ValueError("Not enough independent dog/source groups for the requested folds")
    means = np.stack([frame.mean(axis=0) for frame in frames])
    current = []
    for frame in frames:
        confidence = softmax(frame / baseline_temperature, axis=1).max(axis=1)
        weights = (1 - alpha) + alpha * confidence
        current.append(softmax(np.average(frame, axis=0, weights=weights) / baseline_temperature))
    current = np.stack(current)
    predicted = {name: np.empty((size, len(CATEGORIES)), dtype=np.float64) for name in METHODS}
    fold_id = np.full(size, -1, dtype=int)
    folds = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_span_idx, valid_span_idx) in enumerate(
        splitter.split(np.zeros(len(span_names)), span_y, span_group)
    ):
        fit_mask = np.isin(spans, np.asarray(span_names, dtype=object)[train_span_idx])
        held_mask = np.isin(spans, np.asarray(span_names, dtype=object)[valid_span_idx])
        if set(groups[fit_mask]) & set(groups[held_mask]):
            raise ValueError("A dog/source group crossed the fold boundary")
        fit_idx, held_idx = np.flatnonzero(fit_mask), np.flatnonzero(held_mask)
        if not len(fit_idx) or not len(held_idx):
            raise ValueError("Empty fold; use fewer splits")
        try:
            bias = fit_class_bias(means[fit_idx], y[fit_idx], spans[fit_idx])
            temperature_only = fit_temperature(means[fit_idx], y[fit_idx], spans[fit_idx])
            temperature_bias = fit_temperature(means[fit_idx] - bias, y[fit_idx], spans[fit_idx])
        except ValueError as exc:
            raise ValueError(
                f"Fold {fold}: {exc}; reduce folds or collect more independent labels"
            ) from exc
        predicted["current_weighted"][held_idx] = current[held_idx]
        predicted["mean_logits"][held_idx] = softmax(means[held_idx] / baseline_temperature, axis=1)
        predicted["bias_only"][held_idx] = softmax(
            (means[held_idx] - bias) / baseline_temperature, axis=1
        )
        predicted["temperature_only"][held_idx] = softmax(
            means[held_idx] / temperature_only, axis=1
        )
        predicted["bias_and_temperature"][held_idx] = softmax(
            (means[held_idx] - bias) / temperature_bias, axis=1
        )
        fold_id[held_idx] = fold
        folds.append(
            {
                "fold": fold,
                "fit_indices": fit_idx,
                "held_indices": held_idx,
                "bias": bias,
                "temperature_only": temperature_only,
                "temperature_bias": temperature_bias,
            }
        )
    if (fold_id < 0).any():
        raise RuntimeError("Some segments were not assigned a held-out prediction")
    return {
        "predictions": predicted,
        "labels": y,
        "fold_id": fold_id,
        "group_id": groups,
        "folds": folds,
    }


def paired_group_bootstrap(
    candidate_probs,
    baseline_probs,
    labels,
    span_ids,
    group_ids,
    *,
    n_boot=SCENE_BOOTSTRAP_REPEATS,
    seed=SCENE_BOOTSTRAP_SEED,
):
    """Bootstrap span-weighted paired log-loss difference by dog/source group.

    Candidate minus baseline: negative favors candidate. Interval is conditional on
    the fixed out-of-fold predictions and does not capture model refitting variance.
    """
    first, second = (
        np.asarray(candidate_probs, dtype=float),
        np.asarray(baseline_probs, dtype=float),
    )
    y, spans, groups = np.asarray(labels), np.asarray(span_ids), np.asarray(group_ids)
    n = len(y)
    if (
        not n
        or y.shape != (n,)
        or spans.shape != (n,)
        or groups.shape != (n,)
        or first.shape != (n, len(CATEGORIES))
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or (first < 0).any()
        or (second < 0).any()
        or not np.allclose(first.sum(axis=1), 1, atol=1e-5)
        or not np.allclose(second.sum(axis=1), 1, atol=1e-5)
        or not set(y.tolist()).issubset(set(range(len(CATEGORIES))))
    ):
        raise ValueError("Invalid paired predictions or segment metadata")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    spans = _validate_ids(spans, "span ID", n)
    group_values = []
    for group in groups:
        if (
            isinstance(group, (bool, np.bool_))
            or not isinstance(group, (str, int, np.integer))
            or not str(group).strip()
        ):
            raise ValueError("Each segment needs a nonempty group ID")
        group_values.append(str(group).strip())
    groups = np.asarray(group_values, dtype=object)
    tiny = np.finfo(float).tiny
    row = np.arange(n)
    diffs = -np.log(np.clip(first[row, y.astype(int)], tiny, 1)) + np.log(
        np.clip(second[row, y.astype(int)], tiny, 1)
    )
    span_names = list(dict.fromkeys(spans))
    span_diff, span_group = [], []
    for span in span_names:
        mask = spans == span
        if len(np.unique(groups[mask])) != 1 or len(np.unique(y[mask])) != 1:
            raise ValueError("Every span must belong to one group and one category")
        span_diff.append(float(diffs[mask].mean()))
        span_group.append(groups[mask][0])
    span_diff, span_group = np.asarray(span_diff), np.asarray(span_group)
    unique = list(dict.fromkeys(span_group))
    if len(unique) < 2:
        raise ValueError("At least two dog/source groups are needed for bootstrap")
    totals = np.array([span_diff[span_group == g].sum() for g in unique])
    counts = np.array([(span_group == g).sum() for g in unique])
    random = np.random.default_rng(seed)
    sampled = random.integers(0, len(unique), size=(n_boot, len(unique)))
    differences = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    lo, hi = np.quantile(differences, [0.025, 0.975])
    return {
        "candidate_minus_baseline": float(span_diff.mean()),
        "ci_95_lower": float(lo),
        "ci_95_upper": float(hi),
        "groups": len(unique),
        "spans": len(span_names),
        "n_boot": n_boot,
        "scope": "conditional_on_fixed_out_of_fold_predictions",
    }
