"""Grouped, held-out CLIP calibration experiment tests."""

import numpy as np
import pytest
from scipy.special import softmax

from woofdecoder.taxonomy import CATEGORIES
from wooftrainer.scene_trainer.calibration_experiment import (
    canonical_source_id,
    connected_groups,
    cross_validate,
    fit_class_bias,
    fit_temperature,
)


def collection():
    """Four independent spans per class, with two barks in each span."""
    frames, labels, spans, dogs, sources = [], [], [], [], []
    for category in CATEGORIES:
        k = CATEGORIES.index(category)
        for group in range(4):
            for event in range(2):
                row = np.full(5, -1.0)
                row[k] = 2.0 if (group + event) % 3 else -2.0
                row[(k + 1) % 5] = 1.0 if (group + event) % 3 == 0 else 0.0
                frames.append(np.stack([row, row + np.array([0.1, -0.3, 0.5, 0.0, 0.2])]))
                labels.append(category)
                spans.append(f"{category}-{group}")
                dogs.append(f"dog-{category}-{group}")
                sources.append(f"https://youtu.be/{category}-{group}?si=unrelated")
    return frames, labels, spans, dogs, sources


def test_youtube_canonicalization_and_transitive_groups():
    assert canonical_source_id("https://youtu.be/AbCd?si=123") == canonical_source_id(
        "https://www.youtube.com/watch?v=AbCd&t=30"
    )
    assert canonical_source_id("https://youtube.com/shorts/AbCd") == "youtube:AbCd"
    assert canonical_source_id("https://youtu.be/abcd") != "youtube:AbCd"
    group = connected_groups(
        ["dog-a", "dog-b", "dog-b", "dog-c"],
        ["https://youtu.be/x", "https://youtube.com/watch?v=x", "https://youtu.be/y", "z"],
    )
    assert group[0] == group[1] == group[2] and group[3] != group[0]
    with pytest.raises(ValueError, match="dog"):
        connected_groups([""], ["https://youtu.be/x"])
    with pytest.raises(ValueError, match="source"):
        connected_groups(["dog"], [""])


def test_bias_weights_spans_equally_then_balances_classes():
    x = np.array(
        [[0] * 5, [3] * 5, [6] * 5, [10] * 5, [1] * 5, [2] * 5, [3] * 5, [4] * 5], dtype=float
    )
    labels = np.array([0, 0, 0, 0, 1, 2, 3, 4])
    spans = np.array(["long", "long", "long", "short", "one", "two", "three", "four"])
    # Class 0 mean is (mean(0, 3, 6) + 10) / 2 = 6.5.
    np.testing.assert_allclose(fit_class_bias(x, labels, spans), np.full(5, 3.3))
    with pytest.raises(ValueError, match="five classes"):
        fit_class_bias(x[:-1], labels[:-1], spans[:-1])


def test_temperature_positive_and_scaled_units_are_consistent():
    labels = np.repeat(np.arange(5), 5)
    logits = np.zeros((len(labels), 5))
    for i, category in enumerate(labels):
        logits[i, category if i % 5 else (category + 1) % 5] = 2
    spans = np.array([f"span-{i}" for i in range(len(labels))])
    t1 = fit_temperature(logits, labels, spans)
    t2 = fit_temperature(10 * logits, labels, spans)
    assert t1 > 0 and t2 > 0
    np.testing.assert_allclose(t2, 10 * t1, rtol=1e-4)
    with pytest.raises(ValueError, match="five categories"):
        fit_temperature(logits[labels != 4], labels[labels != 4], spans[labels != 4])


def test_cross_validation_predictions_and_native_baselines():
    data = collection()
    result = cross_validate(*data, n_splits=3, alpha=0.5, baseline_temperature=1.0)
    assert set(result["predictions"]) == {
        "current_weighted",
        "mean_logits",
        "bias_only",
        "temperature_only",
        "bias_and_temperature",
    }
    assert len(result["fold_id"]) == len(data[0])
    for value in result["predictions"].values():
        assert value.shape == (len(data[0]), 5)
        np.testing.assert_allclose(value.sum(axis=1), 1)
        assert np.isfinite(value).all()
    for fold in result["folds"]:
        assert not set(result["group_id"][fold["fit_indices"]]) & set(
            result["group_id"][fold["held_indices"]]
        )
        assert np.isfinite(fold["bias"]).all()
        assert fold["temperature_only"] > 0 and fold["temperature_bias"] > 0
    for i, frame in enumerate(data[0]):
        np.testing.assert_allclose(result["predictions"]["mean_logits"][i], softmax(frame.mean(0)))
        p = softmax(frame, axis=1)
        weight = 0.5 + 0.5 * p.max(axis=1)
        np.testing.assert_allclose(
            result["predictions"]["current_weighted"][i],
            softmax(np.average(frame, axis=0, weights=weight)),
        )
    no_weight = cross_validate(*data, n_splits=3, alpha=0)
    np.testing.assert_allclose(
        no_weight["predictions"]["current_weighted"], no_weight["predictions"]["mean_logits"]
    )


def test_held_out_logit_changes_do_not_change_its_fold_fitted_parameters():
    frames, labels, spans, dogs, sources = collection()
    original = cross_validate(frames, labels, spans, dogs, sources)
    changed = [x.copy() for x in frames]
    held_indices = original["folds"][0]["held_indices"]
    for i in held_indices:
        changed[i] = changed[i] * 12 + 9
    rerun = cross_validate(changed, labels, spans, dogs, sources)
    assert np.array_equal(original["fold_id"], rerun["fold_id"])
    for key in ("bias", "temperature_only", "temperature_bias"):
        np.testing.assert_allclose(original["folds"][0][key], rerun["folds"][0][key])


def test_rejects_bad_frames_span_labels_and_missing_classes():
    frames, labels, spans, dogs, sources = collection()
    with pytest.raises(ValueError, match="logits"):
        cross_validate([np.zeros((0, 5))], [CATEGORIES[0]], ["a"], ["d"], ["source"])
    labels[1] = CATEGORIES[1]
    with pytest.raises(ValueError, match="conflicting"):
        cross_validate(frames, labels, spans, dogs, sources)
    frames, labels, spans, dogs, sources = collection()
    mask = [i for i, label in enumerate(labels) if label != CATEGORIES[-1]]
    with pytest.raises(ValueError, match="five classes"):
        cross_validate(*[[seq[i] for i in mask] for seq in (frames, labels, spans, dogs, sources)])


def test_temperature_weights_each_span_equally_even_with_repeated_barks():
    labels = np.repeat(np.arange(5), 5)
    logits = np.zeros((len(labels), 5))
    for i, category in enumerate(labels):
        logits[i, category if i % 5 else (category + 1) % 5] = 2
    spans = np.array([f"span-{i}" for i in range(len(labels))])
    reference = fit_temperature(logits, labels, spans)
    # A long video generating many identical bark events should not change its weight.
    repeat = np.repeat([0], 20)
    actual = fit_temperature(
        np.concatenate([logits, logits[repeat]]),
        np.concatenate([labels, labels[repeat]]),
        np.concatenate([spans, spans[repeat]]),
    )
    np.testing.assert_allclose(actual, reference, rtol=1e-6)


def test_unscored_intermediate_row_preserves_transitive_group_links():
    # Dog A at source X; unscored Dog B at X; Dog B at source Y.
    all_groups = connected_groups(
        ["dog-a", "dog-b", "dog-b"], ["youtube-X", "youtube-X", "youtube-Y"]
    )
    naive = connected_groups(["dog-a", "dog-b"], ["youtube-X", "youtube-Y"])
    assert naive[0] != naive[1]
    linked = connected_groups(["dog-a", "dog-b"], ["youtube-X", "youtube-Y"], all_groups[[0, 2]])
    assert linked[0] == linked[1]
    frames, labels, spans, dogs, sources = collection()
    # Link two distinct, scored spans based on the full metadata graph.
    ids = np.arange(len(frames)) // 2
    ids[2:4] = ids[0]
    result = cross_validate(frames, labels, spans, dogs, sources, linked_group_ids=ids)
    assert result["group_id"][0] == result["group_id"][2]
    assert result["fold_id"][0] == result["fold_id"][2]
    with pytest.raises(ValueError, match="linked"):
        cross_validate(frames, labels, spans, dogs, sources, linked_group_ids=[0])


def test_bootstrap_is_paired_grouped_and_span_weighted():
    from wooftrainer.scene_trainer.calibration_experiment import paired_group_bootstrap

    # The three rows from span-a have one-third of the collective influence each.
    baseline = np.full((5, 5), 0.2)
    candidate = np.array(
        [[0.6, 0.1, 0.1, 0.1, 0.1]] * 3
        + [[0.4, 0.15, 0.15, 0.15, 0.15], [0.1, 0.225, 0.225, 0.225, 0.225]]
    )
    labels = [0] * 5
    spans = ["span-a"] * 3 + ["span-b", "span-c"]
    groups = ["dog-a"] * 3 + ["dog-b", "dog-c"]
    result = paired_group_bootstrap(candidate, baseline, labels, spans, groups, n_boot=500)
    expected = np.mean([-np.log(0.6 / 0.2), -np.log(0.4 / 0.2), -np.log(0.1 / 0.2)])
    np.testing.assert_allclose(result["candidate_minus_baseline"], expected)
    assert result["groups"] == 3 and result["spans"] == 3
    assert result["ci_95_lower"] <= result["ci_95_upper"]
    assert result["scope"] == "conditional_on_fixed_out_of_fold_predictions"
    identical = paired_group_bootstrap(baseline, baseline, labels, spans, groups, n_boot=10)
    assert (
        identical["candidate_minus_baseline"]
        == identical["ci_95_lower"]
        == identical["ci_95_upper"]
        == 0
    )
    with pytest.raises(ValueError, match="two"):
        paired_group_bootstrap(candidate, baseline, labels, spans, ["one"] * 5, n_boot=10)
