"""Segment aggregation and export regression tests; no weight download required."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.scene.encoder import SceneEncoder, average_frame_probs
from woofdecoder.taxonomy import CATEGORIES
from wooftrainer.scene_trainer.embeddings import (
    load_scene_cache,
    save_scene_cache,
    schema_identity,
)


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    def __call__(self, images, **kwargs):
        return Inputs(pixel_values=torch.tensor([np.asarray(im)[0, 0, 0] for im in images]))


class Model:
    def eval(self):
        return self

    def __call__(self, pixel_values):
        logits = torch.zeros(len(pixel_values), 5)
        logits[:, 0] = pixel_values.float()
        return SimpleNamespace(
            logits_per_image=logits, image_embeds=torch.ones(len(pixel_values), 3)
        )


def encoder():
    scene = SceneEncoder.__new__(SceneEncoder)
    scene.device = "cpu"
    scene.batch_size = 2
    scene.temperature = 1.0
    scene.aggregation = "mean_frame_probabilities"
    scene.confidence_weight = 0.5
    scene.class_bias = np.zeros(5, dtype=np.float32)
    scene.dim = 3
    scene.processor = Processor()
    scene.model = Model()
    scene.text_inputs = {}
    scene.schema = {"embedding_dim": 3, "categories": list(CATEGORIES)}
    scene.schema_id = schema_identity(scene.schema)
    return scene


def segment():
    return BarkSegment(
        start_s=1,
        end_s=2,
        source_id="video",
        frames=[np.full((4, 4, 3), i, np.uint8) for i in (0, 1, 4)],
        frame_times_s=[1.1, 1.5, 1.9],
    )


def test_uneven_batches_average_frames_not_batches_or_logits():
    scene = encoder()
    output = scene.predict(segment())
    expected = (
        torch.tensor([[0.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0], [4.0, 0, 0, 0, 0]]).softmax(-1).numpy()
    )
    np.testing.assert_allclose(output.probs, expected.mean(0), rtol=1e-6)
    assert output.present
    assert output.probs.shape == (5,)
    assert output.extras["frame_count"] == 3
    scene.batch_size = 1
    np.testing.assert_allclose(scene.predict(segment()).probs, output.probs)


def test_missing_video_is_masked_and_has_no_embedding():
    scene = encoder()
    missing = BarkSegment(start_s=1, end_s=2)
    result = scene.encode_segment(missing)
    assert not scene.predict(missing).present
    np.testing.assert_allclose(result["probs"], np.full(5, 0.2))
    assert not result["embedding"].any()
    assert scene.embed(missing) is None


@pytest.mark.parametrize(
    "values", [np.ones((2, 4)), [[np.nan] * 5], [[-1, 2, 0, 0, 0]], [[0.1] * 5]]
)
def test_invalid_frame_distributions_rejected(values):
    with pytest.raises(ValueError):
        average_frame_probs(values)


def test_out_of_segment_frames_rejected():
    bad = segment()
    bad.frame_times_s = [0, 1.5, 1.9]
    with pytest.raises(ValueError, match="inside"):
        encoder().encode_segment(bad)


def test_cache_round_trip_alignment_and_missing_mask(tmp_path):
    scene = encoder()
    results = [
        scene.encode_segment(segment()),
        scene.encode_segment(BarkSegment(start_s=1, end_s=2)),
    ]
    path = tmp_path / "scene.npz"
    save_scene_cache(path, ["video@1-2", "missing@1-2"], results, scene)
    features, rows = load_scene_cache(
        path, clip_ids=["missing@1-2", "video@1-2"], expected_schema_id=scene.schema_id
    )
    assert rows["present"].tolist() == [False, True]
    np.testing.assert_allclose(rows.iloc[1]["probs"], results[0]["probs"])
    np.testing.assert_allclose(features[1], results[0]["embedding"])
    with pytest.raises(ValueError, match="missing"):
        load_scene_cache(path, clip_ids=["unknown"])
    with pytest.raises(ValueError, match="schema"):
        load_scene_cache(path, expected_schema_id="wrong")


def test_old_cache_rejected(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, format_version=2)
    with pytest.raises(ValueError, match="outdated"):
        load_scene_cache(path)


def test_scene_branch_contract():
    from woofdecoder.scene import SceneBranch

    branch = SceneBranch(encoder())
    assert branch.name == "scene"
    outputs = branch.predict_many([segment(), BarkSegment(start_s=0, end_s=1)])
    assert outputs[0].present
    assert not outputs[1].present
    np.testing.assert_allclose(outputs[0].probs, branch.encoder.predict(segment()).probs)


def test_scene_export_uses_clip_and_preserves_alignment(tmp_path, monkeypatch, capsys):
    import json

    from PIL import Image

    import woofdecoder.scene
    from wooftrainer.scene_export import main

    Image.fromarray(segment().frames[0]).save(tmp_path / "frame.png")
    manifest = tmp_path / "segments.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "clip_id": "bark-1",
                    "source_id": "video",
                    "start_s": 1,
                    "end_s": 2,
                    "frames": ["frame.png"],
                    "frame_times_s": [1.5],
                    "dog_id": "dog-1",
                }
            ]
        )
    )
    scene = encoder()
    monkeypatch.setattr(woofdecoder.scene, "encoder_from_config", lambda *a, **kw: scene)
    cache = tmp_path / "clip.npz"
    assert main(["--manifest", str(manifest), "--output", str(cache)]) == 0
    assert "CLIP segments: 5 probabilities" in capsys.readouterr().out
    features, rows = load_scene_cache(cache)
    assert features.shape == (1, scene.dim)
    assert rows.iloc[0]["dog_id"] == "dog-1"
    assert rows.iloc[0]["clip_id"] == "bark-1"
    np.testing.assert_allclose(rows.iloc[0]["probs"], np.full(5, 0.2))


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_mean_logits_then_softmax_with_temperature_and_uneven_batches(temperature):
    scene = encoder()
    scene.aggregation = "mean_frame_logits"
    scene.temperature = temperature
    logits = torch.tensor([[0.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0], [4.0, 0, 0, 0, 0]])
    expected = torch.softmax(logits.mean(dim=0) / temperature, dim=0).numpy()
    # Temperature must be applied to logits, after their mean, before softmax.
    output = scene.predict(segment())
    np.testing.assert_allclose(output.probs, expected, rtol=1e-6)
    assert output.extras["aggregation"] == "mean_frame_logits"
    assert output.present
    assert output.probs.shape == (5,)
    np.testing.assert_allclose(np.sum(output.probs), 1, atol=1e-6)
    for batch_size in [1, 3, 5]:
        scene.batch_size = batch_size
        np.testing.assert_allclose(scene.predict(segment()).probs, expected, rtol=1e-6)
    # These inputs distinguish logit pooling from the baseline probability pooling.
    baseline = torch.softmax(logits / temperature, dim=-1).mean(dim=0).numpy()
    assert not np.allclose(expected, baseline)


def test_higher_temperature_softens_logit_mean_without_changing_winner():
    scene = encoder()
    scene.aggregation = "mean_frame_logits"
    scene.temperature = 1.0
    cold = scene.predict(segment()).probs
    scene.temperature = 2.0
    warm = scene.predict(segment()).probs
    assert int(cold.argmax()) == int(warm.argmax())
    assert warm.max() < cold.max()
    assert warm.min() > cold.min()


@pytest.mark.parametrize("aggregation", ["mean_frame_probabilities", "mean_frame_logits"])
def test_missing_video_has_same_mask_and_uniform_scores_in_both_modes(aggregation):
    scene = encoder()
    scene.aggregation = aggregation
    scene.temperature = 2.0
    output = scene.predict(BarkSegment(start_s=1, end_s=2))
    assert not output.present
    assert output.extras["aggregation"] == aggregation
    assert output.extras["frame_count"] == 0
    np.testing.assert_allclose(output.probs, np.full(5, 0.2))


def test_invalid_aggregation_rejected_before_model_load():
    with pytest.raises(ValueError, match="aggregation"):
        SceneEncoder(aggregation="median")


@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
def test_confidence_weighted_logits_formula_and_batch_invariance(temperature):
    scene = encoder()
    scene.aggregation = "confidence_weighted_logits"
    scene.confidence_weight = 0.5
    scene.temperature = temperature
    # Calculate the expectation independently from the model's returned values.
    raw = np.array([[0.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0], [4.0, 0, 0, 0, 0]])
    frame_scaled = raw / temperature
    frame_exp = np.exp(frame_scaled - frame_scaled.max(axis=1, keepdims=True))
    frame_probs = frame_exp / frame_exp.sum(axis=1, keepdims=True)
    weights = 0.5 + 0.5 * frame_probs.max(axis=1)
    logits = np.average(raw, axis=0, weights=weights) / temperature
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    for batch_size in (1, 2, 3, 5):
        scene.batch_size = batch_size
        output = scene.predict(segment())
        assert output.present
        assert output.extras["aggregation"] == "confidence_weighted_logits"
        np.testing.assert_allclose(output.probs, expected, rtol=1e-6)
        assert np.isclose(output.probs.sum(), 1)
    scene.aggregation = "mean_frame_logits"
    assert not np.allclose(scene.predict(segment()).probs, expected)


@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
def test_zero_confidence_weight_matches_mean_logits(temperature):
    scene = encoder()
    scene.temperature = temperature
    scene.aggregation = "mean_frame_logits"
    expected = scene.predict(segment()).probs
    scene.aggregation = "confidence_weighted_logits"
    scene.confidence_weight = 0.0
    np.testing.assert_allclose(scene.predict(segment()).probs, expected, rtol=1e-6)


def test_confidence_weighted_missing_video_is_masked():
    scene = encoder()
    scene.aggregation = "confidence_weighted_logits"
    output = scene.predict(BarkSegment(start_s=0, end_s=1))
    assert not output.present
    np.testing.assert_allclose(output.probs, np.full(5, 0.2))


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence_weight_rejected_before_model_load(strength):
    with pytest.raises(ValueError, match="confidence_weight"):
        SceneEncoder(confidence_weight=strength)


def test_project_config_selects_weighted_experiment():
    from woofdecoder.config import load_config

    scene = load_config()["branches"]["scene"]
    assert scene["aggregation"] == "confidence_weighted_logits"
    assert scene["confidence_weight"] == 0.5
    assert scene["temperature"] == 1.0


@pytest.mark.parametrize(
    "aggregation", ["mean_frame_probabilities", "mean_frame_logits", "confidence_weighted_logits"]
)
def test_class_bias_applied_before_temperature_and_softmax(aggregation):
    scene = encoder()
    scene.aggregation = aggregation
    scene.temperature = 2.0
    scene.class_bias = np.array([1.0, -0.5, 0, 0, 0], dtype=np.float32)
    raw = torch.tensor([[0.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0], [4.0, 0, 0, 0, 0]])
    bias = torch.tensor(scene.class_bias)
    if aggregation == "mean_frame_probabilities":
        expected = ((raw - bias) / scene.temperature).softmax(-1).mean(0)
    else:
        if aggregation == "confidence_weighted_logits":
            confidence = (raw / scene.temperature).softmax(-1).max(-1).values
            weights = 0.5 + 0.5 * confidence
            mean = (raw * weights[:, None]).sum(0) / weights.sum()
        else:
            mean = raw.mean(0)
        expected = ((mean - bias) / scene.temperature).softmax(-1)
    output = scene.predict(segment())
    np.testing.assert_allclose(output.probs, expected.numpy(), rtol=1e-6)
    # Frame probabilities and raw logits remain available for the confidence weights.
    np.testing.assert_allclose(
        output.extras["frame_probs"], (raw / 2).softmax(-1).numpy(), rtol=1e-6
    )
    missing = scene.predict(BarkSegment(start_s=0, end_s=1))
    assert not missing.present
    np.testing.assert_allclose(missing.probs, np.full(5, 0.2))


@pytest.mark.parametrize(
    "bias",
    [
        [1, 2],
        [0, 0, 0, 0, float("nan")],
        [True, True, True, True, True],
        [0, 0, 0, 0, True],
        "invalid",
    ],
)
def test_invalid_class_bias_rejected_before_model_load(bias):
    with pytest.raises(ValueError, match="class_bias"):
        SceneEncoder(class_bias=bias)


def test_class_bias_config_and_cache_identity(monkeypatch):
    import sys

    from woofdecoder.scene.encoder import encoder_from_config

    class FakeModel:
        config = SimpleNamespace(projection_dim=3, _commit_hash="revision")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

    class FakeProcessor:
        image_processor = SimpleNamespace(to_dict=dict)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, **kwargs):
            return Inputs()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(CLIPModel=FakeModel, CLIPProcessor=FakeProcessor),
    )
    config = {"branches": {"scene": {"encoder": "clip_scene_v1", "model": "mock"}}}
    disabled = encoder_from_config(config, device="cpu")
    config["branches"]["scene"]["class_bias"] = [0, 0, 0, 0, 0]
    zero = encoder_from_config(config, device="cpu")
    assert disabled.schema_id == zero.schema_id
    config["branches"]["scene"]["class_bias"] = [1, 0, 0, 0, 0]
    corrected = encoder_from_config(config, device="cpu")
    assert corrected.schema["class_bias"] == [1, 0, 0, 0, 0]
    assert corrected.schema_id != zero.schema_id
