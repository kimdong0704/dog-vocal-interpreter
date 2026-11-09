"""Mock outputs test the scene export contract, not model recognition quality."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from woofdecoder.branches.scene import ObjectFeaturePool, SceneEncoder
from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.taxonomy import CATEGORIES
from wooftrainer.trainers.scene_embeddings import (
    cache_scene_segments,
    load_scene_cache,
    save_scene_cache,
)


@pytest.fixture
def encoder():
    schema = {
        "encoder": "test-scene",
        "vocabulary": ["dog", "person"],
        "feature_names": list(ObjectFeaturePool.feature_names),
    }
    schema_id = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    return SimpleNamespace(schema=schema, schema_id=schema_id, dim=20)


def scene_result(encoder, class_id=0, frame_count=2, detected=True, source_id="video-a"):
    objects = []
    if detected:
        objects.append(
            {
                "frame": 0,
                "class_id": class_id,
                "name": encoder.schema["vocabulary"][class_id],
                "confidence": 0.8,
                "box": [0.1, 0.2, 0.5, 0.8],
                "source": "rfdetr" if class_id == 0 else "yoloe",
            }
        )
    embedding = ObjectFeaturePool(encoder.schema["vocabulary"]).encode(objects, frame_count)
    return {
        "embedding": embedding,
        "detected_objects": objects,
        "present": frame_count > 0,
        "has_detections": bool(objects),
        "frame_count": frame_count,
        "schema_id": encoder.schema_id,
        "source_id": source_id,
        "start_s": 10.0,
        "end_s": 12.0,
        "frame_times_s": [10.25 + i / 2 for i in range(frame_count)],
    }


def rewrite_cache(path, **changes):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


def test_cache_loads_without_models_and_aligns_by_explicit_ids(tmp_path, encoder, monkeypatch):
    results = [scene_result(encoder), scene_result(encoder, class_id=1, source_id="video-b")]
    path = tmp_path / "scene.npz"
    saved_path = save_scene_cache(path, ["segment-a", "segment-b"], results, encoder)
    assert saved_path == path

    def no_models(*args, **kwargs):
        raise AssertionError("Loading scene features must not construct a detector")

    monkeypatch.setattr(SceneEncoder, "__init__", no_models)
    features, table = load_scene_cache(path, clip_ids=["segment-b", "segment-a"])
    assert features.dtype == np.float32
    assert features.shape == (2, encoder.dim)
    np.testing.assert_array_equal(features, np.stack([r["embedding"] for r in results[::-1]]))
    assert table["clip_id"].tolist() == ["segment-b", "segment-a"]
    assert table["source_id"].tolist() == ["video-b", "video-a"]
    assert table.attrs["scene_schema"] == encoder.schema
    assert table.attrs["scene_schema_id"] == encoder.schema_id

    subset, subset_table = load_scene_cache(path, clip_ids=["segment-a"])
    np.testing.assert_array_equal(subset[0], results[0]["embedding"])
    assert subset_table["clip_id"].tolist() == ["segment-a"]


def test_metadata_roundtrip_preserves_actual_labels_and_segment_times(tmp_path, encoder):
    result = scene_result(encoder)
    annotations = [{"dog_id": "dog-a", "split": "train", "mapped_category": CATEGORIES[1]}]
    path = save_scene_cache(
        tmp_path / "scene.npz", ["segment-a"], [result], encoder, metadata=annotations
    )
    _, table = load_scene_cache(path)
    row = table.iloc[0]
    assert row["dog_id"] == "dog-a"
    assert row["split"] == "train"
    assert row["mapped_category"] == CATEGORIES[1]
    assert row["source_id"] == result["source_id"]
    assert row["start_s"] == result["start_s"]
    assert row["end_s"] == result["end_s"]
    assert row["frame_times_s"] == result["frame_times_s"]


@pytest.mark.parametrize("ids", [["same", "same"], ["", "b"], ["  ", "b"], [1, "1"]])
def test_export_rejects_invalid_ids(tmp_path, encoder, ids):
    results = [scene_result(encoder), scene_result(encoder, class_id=1)]
    with pytest.raises(ValueError):
        save_scene_cache(tmp_path / "scene.npz", ids, results, encoder)


@pytest.mark.parametrize("ids", [["missing"], ["a", "a"], [""], [1]])
def test_requested_alignment_rejects_missing_or_invalid_ids(tmp_path, encoder, ids):
    path = save_scene_cache(tmp_path / "scene.npz", ["a"], [scene_result(encoder)], encoder)
    with pytest.raises(ValueError):
        load_scene_cache(path, clip_ids=ids)


def test_cache_load_rejects_duplicate_stored_ids(tmp_path, encoder):
    path = save_scene_cache(
        tmp_path / "scene.npz",
        ["a", "b"],
        [scene_result(encoder), scene_result(encoder, class_id=1)],
        encoder,
    )
    rewrite_cache(path, clip_id=np.array(["a", "a"]))
    with pytest.raises(ValueError):
        load_scene_cache(path)


def test_scene_cache_checks_schema_without_loading_encoder(tmp_path, encoder):
    path = save_scene_cache(tmp_path / "scene.npz", ["a"], [scene_result(encoder)], encoder)
    with pytest.raises(ValueError):
        load_scene_cache(path, expected_schema_id="another-scene-encoder")
    features, _ = load_scene_cache(path, expected_schema_id=encoder.schema_id)
    assert features.shape == (1, encoder.dim)
    rewrite_cache(path, schema_id=np.array("tampered"))
    with pytest.raises(ValueError):
        load_scene_cache(path)


def test_changed_vocabulary_invalidates_stored_schema_hash(tmp_path, encoder):
    path = save_scene_cache(tmp_path / "scene.npz", ["a"], [scene_result(encoder)], encoder)
    changed = {**encoder.schema, "vocabulary": ["person", "dog"]}
    rewrite_cache(path, schema_json=np.array(json.dumps(changed, sort_keys=True)))
    with pytest.raises(ValueError):
        load_scene_cache(path)


def test_missing_video_is_distinct_from_no_detected_objects(tmp_path, encoder):
    results = [
        scene_result(encoder, frame_count=0, detected=False),
        scene_result(encoder, detected=False),
    ]
    path = save_scene_cache(tmp_path / "scene.npz", ["missing", "empty"], results, encoder)
    features, table = load_scene_cache(path)
    assert not features.any()
    assert table["present"].tolist() == [False, True]
    assert table["has_detections"].tolist() == [False, False]
    assert table["frame_count"].tolist() == [0, 2]


@pytest.mark.parametrize(
    "changes",
    [
        {"present": False},
        {"frame_count": -1},
        {"has_detections": False},
    ],
)
def test_export_rejects_inconsistent_availability(tmp_path, encoder, changes):
    result = {**scene_result(encoder), **changes}
    with pytest.raises(ValueError):
        save_scene_cache(tmp_path / "scene.npz", ["a"], [result], encoder)


@pytest.mark.parametrize(
    "changes",
    [
        {"present": np.array([False])},
        {"frame_count": np.array([-1], dtype=np.int32)},
        {"has_detections": np.array([False])},
        {"embedding": np.full((1, 20), np.nan, dtype=np.float32)},
    ],
)
def test_load_rejects_corrupted_features_or_masks(tmp_path, encoder, changes):
    path = save_scene_cache(tmp_path / "scene.npz", ["a"], [scene_result(encoder)], encoder)
    rewrite_cache(path, **changes)
    with pytest.raises(ValueError):
        load_scene_cache(path)


@pytest.mark.parametrize(
    "metadata",
    [
        {"clip_id": "other"},
        {"present": False},
        {"source_id": "other"},
        {"start_s": 99.0},
        {"mapped_category": "happy"},
        {"split": "development"},
    ],
)
def test_annotations_cannot_override_identity_or_invent_categories(tmp_path, encoder, metadata):
    with pytest.raises(ValueError):
        save_scene_cache(
            tmp_path / "scene.npz", ["a"], [scene_result(encoder)], encoder, metadata=[metadata]
        )


def test_rejected_export_preserves_existing_cache(tmp_path, encoder):
    path = save_scene_cache(tmp_path / "scene.npz", ["a"], [scene_result(encoder)], encoder)
    original = path.read_bytes()
    broken = deepcopy(scene_result(encoder))
    broken["embedding"][0] = np.nan
    with pytest.raises(ValueError):
        save_scene_cache(path, ["b"], [broken], encoder)
    assert path.read_bytes() == original


def test_interrupted_write_preserves_existing_cache(tmp_path, encoder, monkeypatch):
    path = save_scene_cache(tmp_path / "scene.npz", ["a"], [scene_result(encoder)], encoder)
    original = path.read_bytes()

    def interrupted(destination, *args, **kwargs):
        if hasattr(destination, "write"):
            destination.write(b"partial")
        else:
            Path(destination).write_bytes(b"partial")
        raise OSError("simulated interrupted cache write")

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    with pytest.raises(OSError):
        save_scene_cache(path, ["b"], [scene_result(encoder)], encoder)
    assert path.read_bytes() == original


def test_export_requires_an_explicit_npz_path(tmp_path, encoder):
    with pytest.raises(ValueError):
        save_scene_cache(tmp_path / "scene", ["a"], [scene_result(encoder)], encoder)


def test_encode_segment_preserves_preprocessing_timestamps(monkeypatch):
    encoder = SceneEncoder.__new__(SceneEncoder)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)] * 2
    segment = BarkSegment(10, 12, frames=frames, frame_times_s=[10.25, 11.25], source_id="video-a")
    calls = []

    def encode_frames(actual_frames, frame_times_s=None):
        calls.append((actual_frames, frame_times_s))
        return {"embedding": np.zeros(20, dtype=np.float32)}

    monkeypatch.setattr(encoder, "encode_frames", encode_frames)
    result = encoder.encode_segment(segment)
    assert calls[0][0] is segment.frames
    assert calls[0][1] == segment.frame_times_s
    assert result["source_id"] == "video-a"
    assert result["start_s"] == 10
    assert result["end_s"] == 12
    assert result["frame_times_s"] == [10.25, 11.25]


def test_encode_segments_keeps_input_order(monkeypatch):
    encoder = SceneEncoder.__new__(SceneEncoder)
    monkeypatch.setattr(encoder, "encode_segment", lambda segment: {"source_id": segment.source_id})
    segments = [BarkSegment(0, 1, source_id="b"), BarkSegment(1, 2, source_id="a")]
    assert encoder.encode_segments(iter(segments)) == [{"source_id": "b"}, {"source_id": "a"}]


def test_segment_export_accepts_a_preprocessing_iterator(tmp_path, encoder):
    segments = [BarkSegment(10, 12, source_id="a"), BarkSegment(10, 12, source_id="b")]
    encoder.encode_segment = lambda segment: scene_result(
        encoder, frame_count=0, detected=False, source_id=segment.source_id
    )
    encoder.encode_segments = lambda items: [encoder.encode_segment(s) for s in items]
    path = cache_scene_segments(
        tmp_path / "scene.npz", ["segment-a", "segment-b"], iter(segments), encoder
    )
    _, table = load_scene_cache(path)
    assert table["source_id"].tolist() == ["a", "b"]


def test_segment_export_rejects_count_mismatch_without_writing(tmp_path, encoder):
    encoder.encode_segment = lambda segment: scene_result(encoder, frame_count=0, detected=False)
    encoder.encode_segments = lambda items: [encoder.encode_segment(s) for s in items]
    path = tmp_path / "scene.npz"
    with pytest.raises(ValueError):
        cache_scene_segments(path, ["a", "b"], iter([BarkSegment(0, 1)]), encoder)
    assert not path.exists()
