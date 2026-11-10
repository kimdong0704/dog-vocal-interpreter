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
