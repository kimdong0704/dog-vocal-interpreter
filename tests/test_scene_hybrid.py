"""Mock detections check detector merging, not real recognition accuracy."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from woofdecoder.branches import scene as scene_module
from woofdecoder.branches.scene import ObjectFeaturePool, merge_dog_detections
from woofdecoder.preprocess.dog_detector import DogDetector, load_dog_detector
from woofdecoder.preprocess.segments import BarkSegment
from wooftrainer.trainers.scene_embeddings import load_scene_cache, save_scene_cache


def yolo_object(name="reindeer", box=None, class_id=0, confidence=0.95, frame=0):
    return {
        "frame": frame,
        "class_id": class_id,
        "name": name,
        "confidence": confidence,
        "box": [0.1, 0.2, 0.5, 0.8] if box is None else box,
        "source": "yoloe",
    }


def dog(box=None, confidence=0.7):
    return np.array([20, 20, 80, 60] if box is None else box, dtype=np.float32), confidence


@pytest.mark.parametrize("wrong_name", ["reindeer", "horse", "livestock"])
def test_rf_detr_replaces_wrong_label_without_comparing_confidence(wrong_name):
    objects = [yolo_object(name=wrong_name, confidence=0.99)]
    before = deepcopy(objects)

    combined, suppressed = merge_dog_detections(
        objects, [dog(confidence=0.55)], 0, (100, 200, 3), 2
    )

    assert len(combined) == 1
    assert combined[0]["name"] == "dog"
    assert combined[0]["class_id"] == 2
    assert combined[0]["source"] == "rfdetr"
    assert combined[0]["confidence"] == pytest.approx(0.55)
    np.testing.assert_allclose(combined[0]["box"], [0.1, 0.2, 0.5, 0.8])
    assert len(suppressed) == 1
    assert suppressed[0]["name"] == wrong_name
    assert objects == before


def test_same_dog_from_two_detectors_is_counted_once():
    objects = [yolo_object(name="dog", class_id=2)]

    combined, suppressed = merge_dog_detections(objects, [dog()], 0, (100, 200, 3), 2)

    assert [obj["name"] for obj in combined] == ["dog"]
    assert len(suppressed) == 1


def test_multiple_dogs_survive_merging_and_pool_into_the_dog_class():
    objects = [
        yolo_object(),
        yolo_object(box=[0.6, 0.2, 0.9, 0.8]),
        yolo_object(name="person", class_id=1, box=[0.0, 0.0, 0.05, 0.9]),
    ]
    dogs = [dog(), dog([120, 20, 60, 60], confidence=0.8)]

    combined, suppressed = merge_dog_detections(objects, dogs, 0, (100, 200, 3), 2)

    assert sum(obj["name"] == "dog" for obj in combined) == 2
    assert sum(obj["name"] == "person" for obj in combined) == 1
    assert len(suppressed) == 2
    pool = ObjectFeaturePool(["reindeer", "person", "dog"])
    features = pool.encode(combined, 1).reshape(3, -1)
    assert not features[0].any()
    assert features[1, 1] == 1
    assert features[2, 1] == 2
    assert features[2, 2] == pytest.approx(0.8)


def test_small_contained_bowl_and_nearby_person_are_preserved():
    objects = [
        yolo_object(),
        yolo_object(name="bowl", class_id=3, box=[0.25, 0.4, 0.3, 0.45]),
        yolo_object(name="person", class_id=1, box=[0.45, 0.0, 0.8, 0.9]),
    ]

    combined, suppressed = merge_dog_detections(objects, [dog()], 0, (100, 200, 3), 2)

    assert sorted(obj["name"] for obj in combined) == ["bowl", "dog", "person"]
    assert [obj["name"] for obj in suppressed] == ["reindeer"]


def test_same_box_in_another_frame_is_not_suppressed():
    objects = [yolo_object(frame=0), yolo_object(frame=1)]

    combined, suppressed = merge_dog_detections(objects, [dog()], 0, (100, 200, 3), 2)

    assert any(obj["frame"] == 1 and obj["name"] == "reindeer" for obj in combined)
    assert any(obj["frame"] == 0 and obj["name"] == "dog" for obj in combined)
    assert len(suppressed) == 1


def test_overlap_threshold_controls_replacement():
    objects = [yolo_object()]
    dogs = [dog([40, 20, 80, 60])]

    merged, suppressed = merge_dog_detections(objects, dogs, 0, (100, 200, 3), 2, overlap_iou=0.5)
    retained, not_suppressed = merge_dog_detections(
        objects, dogs, 0, (100, 200, 3), 2, overlap_iou=0.75
    )

    assert len(merged) == 1 and len(suppressed) == 1
    assert len(retained) == 2 and not_suppressed == []


def test_no_dog_detections_leave_yoloe_results_unchanged():
    objects = [yolo_object(name="person", class_id=1)]

    combined, suppressed = merge_dog_detections(objects, [], 0, (100, 200, 3), 2)

    assert combined == objects
    assert suppressed == []


def test_pixel_boxes_are_clipped_and_zero_area_boxes_are_skipped():
    dogs = [dog([-20, -10, 100, 70]), dog([250, 20, 30, 20]), dog([10, 10, 0, 20])]

    combined, suppressed = merge_dog_detections([], dogs, 0, (100, 200, 3), 2)

    assert len(combined) == 1
    np.testing.assert_allclose(combined[0]["box"], [0, 0, 0.4, 0.6])
    assert suppressed == []


def test_empty_detections_are_valid():
    combined, suppressed = merge_dog_detections([], [], 0, (100, 200, 3), 2)

    assert combined == []
    assert suppressed == []


@pytest.mark.parametrize(
    "bad_dog",
    [dog([np.nan, 20, 80, 60]), dog(confidence=np.nan), dog(confidence=1.5)],
)
def test_invalid_dog_values_are_rejected(bad_dog):
    with pytest.raises(ValueError):
        merge_dog_detections([], [bad_dog], 0, (100, 200, 3), 2)


@pytest.fixture
def encoder_factory(tmp_path, monkeypatch):
    checkpoint = tmp_path / "mock-seg-pf.pt"
    checkpoint.write_bytes(b"fake YOLOE checkpoint")
    dog_checkpoint = tmp_path / "mock-dog.pt"
    dog_checkpoint.write_bytes(b"fake RF-DETR checkpoint")
    dog_calls = []

    class FakeYOLOE:
        def __init__(self, checkpoint):
            self.model = torch.nn.Linear(1, 1)
            self.names = {0: "reindeer", 1: "person", 2: "dog", 3: "bowl"}
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["source"].mode == "RGB"
            boxes = []
            for class_id, confidence, box in (
                (0, 0.95, [0.1, 0.2, 0.5, 0.8]),
                (3, 0.7, [0.25, 0.4, 0.3, 0.45]),
            ):
                boxes.append(
                    SimpleNamespace(
                        cls=torch.tensor([class_id]),
                        conf=torch.tensor([confidence]),
                        xyxyn=torch.tensor([box]),
                    )
                )
            return [SimpleNamespace(names=self.names, boxes=boxes)]

    def fake_detect_dogs(detector, frames):
        dog_calls.append((detector, frames))
        assert len(frames) == 1
        return [[dog()]]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOE=FakeYOLOE))
    monkeypatch.setattr(scene_module, "version", lambda package: "test-version")
    monkeypatch.setattr(scene_module, "detect_dogs", fake_detect_dogs)

    def create(confidence=0.5, overlap_iou=0.5):
        detector = DogDetector(
            SimpleNamespace(model_config=SimpleNamespace(pretrain_weights=str(dog_checkpoint))),
            score_threshold=confidence,
        )
        return scene_module.SceneEncoder(
            checkpoint,
            device="cpu",
            dog_detector=detector,
            dog_overlap_iou=overlap_iou,
        )

    return create, dog_calls


def test_encoder_preserves_raw_results_and_records_dog_replacements(encoder_factory):
    create, dog_calls = encoder_factory
    encoder = create()
    frames = [np.zeros((100, 200, 3), dtype=np.uint8)] * 2

    result = encoder.encode_frames(frames, frame_times_s=[1.2, 1.5])

    assert encoder.dim == 40
    assert len(encoder.model.calls) == len(dog_calls) == 2
    assert result["present"] and result["has_detections"]
    assert result["frame_count"] == 2
    assert [obj["name"] for obj in result["yoloe_objects"]] == [
        "reindeer",
        "bowl",
        "reindeer",
        "bowl",
    ]
    assert [obj["name"] for obj in result["suppressed_objects"]] == ["reindeer"] * 2
    assert [obj["name"] for obj in result["dog_objects"]] == ["dog"] * 2
    assert all(obj["source"] == "rfdetr" for obj in result["dog_objects"])
    assert all(obj["source"] == "yoloe" for obj in result["yoloe_objects"])
    assert all(obj["matched_dog_iou"] > 0.99 for obj in result["suppressed_objects"])
    for key in ("detected_objects", "yoloe_objects", "dog_objects", "suppressed_objects"):
        assert all(obj["time_s"] == [1.2, 1.5][obj["frame"]] for obj in result[key])
    features = result["embedding"].reshape(4, 10)
    assert not features[0].any()
    assert features[2, 1] == 1
    assert features[3, 1] == 1
    assert not any(parameter.requires_grad for parameter in encoder.model.model.parameters())


def test_empty_encoder_input_does_not_run_either_detector(encoder_factory):
    create, dog_calls = encoder_factory
    encoder = create()

    result = encoder.encode_frames([])

    assert encoder.model.calls == [] and dog_calls == []
    assert not result["present"] and not result["has_detections"]
    assert result["frame_count"] == 0
    assert result["embedding"].shape == (encoder.dim,)
    assert not result["embedding"].any()
    assert all(
        result[key] == []
        for key in (
            "detected_objects",
            "yoloe_objects",
            "dog_objects",
            "suppressed_objects",
        )
    )
    assert encoder.embed(BarkSegment(0, 1)) is None
    assert encoder.model.calls == [] and dog_calls == []


def test_hybrid_settings_change_schema_and_invalidate_cached_features(encoder_factory, tmp_path):
    create, _ = encoder_factory
    encoder = create()
    changed_confidence = create(confidence=0.7)
    changed_overlap = create(overlap_iou=0.7)
    assert len({encoder.schema_id, changed_confidence.schema_id, changed_overlap.schema_id}) == 3
    assert encoder.names == changed_confidence.names == changed_overlap.names
    assert encoder.schema["dog_detector"]["checkpoint_sha256"]
    result = encoder.encode_frames([])
    path = tmp_path / "hybrid.npz"
    save_scene_cache(path, ["segment"], [result], encoder)
    loaded, rows = load_scene_cache(path, encoder)
    np.testing.assert_array_equal(loaded, result["embedding"][None])
    assert rows["clip_id"].tolist() == ["segment"]
    for changed in (changed_confidence, changed_overlap):
        with pytest.raises(ValueError, match="settings differ"):
            load_scene_cache(path, changed)

    with np.load(path, allow_pickle=False) as cache:
        old_cache = {key: cache[key].copy() for key in cache.files}
    old_schema = {key: value for key, value in encoder.schema.items() if key != "dog_detector"}
    old_schema["encoder"] = "yoloe_object_pool_v1"
    old_cache["schema_json"] = np.array(json.dumps(old_schema))
    old_path = tmp_path / "old_yoloe_only.npz"
    np.savez_compressed(old_path, **old_cache)
    with pytest.raises(ValueError, match="settings differ"):
        load_scene_cache(old_path, encoder)


def test_scene_factory_forwards_device_and_dog_settings(encoder_factory, monkeypatch, tmp_path):
    create, _ = encoder_factory
    injected_model = create().dog_detector.model
    calls = []

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return DogDetector(injected_model, score_threshold=kwargs["score_threshold"])

    monkeypatch.setattr(scene_module, "load_dog_detector", fake_loader)
    config = {
        "dog_detector": {"model": "small", "score_threshold": 0.65},
        "branches": {
            "scene": {
                "encoder": scene_module.SceneEncoder.name,
                "model": str(tmp_path / "mock-seg-pf.pt"),
                "confidence": 0.6,
                "dog_overlap_iou": 0.7,
            }
        },
    }

    encoder = scene_module.encoder_from_config(config, device="cpu")

    assert calls == [
        {"model_size": "small", "score_threshold": 0.65, "verbose": False, "device": "cpu"}
    ]
    assert encoder.device == "cpu"
    assert encoder.dog_detector.model is injected_model
    assert encoder.schema["dog_detector"]["confidence"] == 0.65
    assert encoder.schema["dog_detector"]["overlap_iou"] == 0.7


def test_scene_constructor_forwards_device_to_default_dog_loader(
    encoder_factory,
    monkeypatch,
    tmp_path,
):
    create, _ = encoder_factory
    detector = create().dog_detector
    calls = []

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return detector

    monkeypatch.setattr(scene_module, "load_dog_detector", fake_loader)

    encoder = scene_module.SceneEncoder(tmp_path / "mock-seg-pf.pt", device="cpu")

    assert calls == [{"verbose": False, "device": "cpu"}]
    assert encoder.dog_detector is detector


@pytest.mark.parametrize("device", ["cpu", None])
def test_existing_dog_loader_forwards_only_an_explicit_device(monkeypatch, device):
    calls = []

    def fake_nano(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name="mock nano")

    monkeypatch.setitem(
        sys.modules,
        "rfdetr",
        SimpleNamespace(
            RFDETRNano=fake_nano,
            RFDETRSmall=fake_nano,
        ),
    )

    detector = load_dog_detector(score_threshold=0.6, verbose=False, device=device)

    assert calls == ([{"device": "cpu"}] if device is not None else [{}])
    assert detector.score_threshold == 0.6
    assert detector.model.name == "mock nano"
