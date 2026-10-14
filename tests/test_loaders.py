"""Tests for tasks, loaders and the PyTorch dataset. A tiny synthetic data folder only."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from woofdata.loader import crop_to_box, load_audio_segment, load_sample
from woofdata.prepare.manifest import MANIFEST_FILENAME, ManifestRow, make_clip_id, rows_to_frame
from woofdata.prepare.sources.stanfordextra import KEYPOINT_NAMES
from woofdata.prepare.utils.audio import TARGET_SAMPLE_RATE, write_wav_16k
from woofdata.prepare.utils.coco import add_coco_annotation, add_coco_image, new_coco, write_coco
from woofdata.prepare.utils.label_files import write_labels, write_sed_classes
from woofdata.prepare.utils.windows import save_frame_targets
from woofdata.tasks import BarkDetectorSample, PoseSample, task_rows
from woofdata.utils_fileio import relative_data_path, write_csv
from woofdecoder.sed_classes import SED_CLASSES

IMAGE_WIDTH: int = 80
IMAGE_HEIGHT: int = 60
POSE_BOX: list[float] = [10.0, 20.0, 40.0, 30.0]


def write_tone(path: Path, seconds: float) -> None:
    """Write a 16 kHz sine tone of the given length."""
    times = np.arange(int(seconds * TARGET_SAMPLE_RATE)) / TARGET_SAMPLE_RATE  # (num_samples,)
    write_wav_16k(path, 0.5 * np.sin(2 * np.pi * 440.0 * times))


def audio_row(source: str, wav_path: Path, split: str, end_s: float, **fields: Any) -> ManifestRow:
    """Return one audio manifest row starting at 0 s."""
    defaults: dict[str, Any] = {"label_type": "call_type", "label_provenance": "expert"}
    return ManifestRow(
        clip_id=make_clip_id(source, wav_path.name),
        source=source,
        split=split,
        modality="audio",
        file_path=relative_data_path(wav_path),
        start_s=0.0,
        end_s=end_s,
        **(defaults | fields),
    )


def make_esc50(data_dir: Path) -> list[ManifestRow]:
    """A 5 s dog clip without targets (train) and a 5 s rain clip with targets (val)."""
    source_dir = data_dir / "processed" / "esc50"
    write_tone(source_dir / "audio" / "dog.wav", 5.0)
    write_tone(source_dir / "audio" / "rain.wav", 5.0)
    target_path = source_dir / "targets" / "rain.npy"
    save_frame_targets(target_path, np.zeros((250, len(SED_CLASSES)), dtype=np.float32))
    write_sed_classes(source_dir, SED_CLASSES)
    dog = audio_row("esc50", source_dir / "audio" / "dog.wav", "train", 10.0, original_label="dog")
    rain = audio_row(
        "esc50",
        source_dir / "audio" / "rain.wav",
        "val",
        10.0,
        original_label="rain",
        label_type="object",
        target_path=relative_data_path(target_path),
    )
    return [dog, rain]


def make_act_env(data_dir: Path) -> list[ManifestRow]:
    """One 15 s train_only clip (longer than the 10 s cap) with a labels.csv row."""
    source_dir = data_dir / "processed" / "act_env"
    wav_path = source_dir / "audio" / "a.wav"
    write_tone(wav_path, 15.0)
    row = audio_row(
        "act_env",
        wav_path,
        "train_only",
        15.0,
        label_type="context",
        label_provenance="vlm_video",
        mapped_subtype="play",
        eval_allowed=False,
    )
    label = {"clip_id": row.clip_id, "environment_normalized": "near door", "activity": "rest"}
    write_labels(source_dir, [label], list(label))
    return [row]


def make_stanfordextra(data_dir: Path) -> list[ManifestRow]:
    """One image with a keypoint COCO JSON (train)."""
    image_path = data_dir / "processed" / "stanford_dogs" / "images" / "breed" / "dog1.png"
    image_path.parent.mkdir(parents=True)
    pixels = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image_path)
    keypoints = [0.0] * (3 * len(KEYPOINT_NAMES))
    keypoints[0:3] = [30.0, 35.0, 2]
    coco = new_coco(["dog"], KEYPOINT_NAMES)
    image_id = add_coco_image(coco, image_path, IMAGE_WIDTH, IMAGE_HEIGHT, data_dir)
    add_coco_annotation(coco, image_id, "dog", POSE_BOX, keypoints)
    coco_path = data_dir / "processed" / "stanfordextra" / "keypoints_train.json"
    write_coco(coco, coco_path)
    row = ManifestRow(
        clip_id=make_clip_id("stanfordextra", "dog1"),
        source="stanfordextra",
        split="train",
        modality="image",
        file_path=relative_data_path(image_path),
        annotation_path=relative_data_path(coco_path),
        label_type="keypoints",
        label_provenance="expert",
    )
    return [row]


@pytest.fixture
def data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the data folder at a tiny synthetic one with a manifest; return it."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(data_dir))
    rows = make_esc50(data_dir) + make_act_env(data_dir) + make_stanfordextra(data_dir)
    write_csv(rows_to_frame(rows), data_dir / "processed" / MANIFEST_FILENAME)
    return data_dir


# --- task_rows ---


def test_train_includes_train_only(data_dir: Path) -> None:
    rows = task_rows("audio_context", "train")
    assert list(rows["split"]) == ["train_only"]


def test_eval_only_keeps_eval_allowed_rows(data_dir: Path) -> None:
    assert len(task_rows("audio_context", "train", eval_only=True)) == 0
    assert list(task_rows("bark_detector", "val", eval_only=True)["original_label"]) == ["rain"]


def test_test_split_prints_reminder(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    task_rows("bark_detector", "test")
    assert "test is reported once" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("task", "split", "sources"),
    [("no_such_task", "train", None), ("pose", "dev", None), ("pose", "train", ["esc50"])],
)
def test_bad_arguments_raise(
    data_dir: Path, task: str, split: str, sources: list[str] | None
) -> None:
    with pytest.raises(ValueError):
        task_rows(task, split, sources=sources)


# --- Audio ---


def test_window_past_end_is_zero_padded(data_dir: Path) -> None:
    row = task_rows("bark_detector", "train").iloc[0]  # 5 s clip, 0-10 s window
    audio = load_audio_segment(row)
    assert audio.shape == (10 * TARGET_SAMPLE_RATE,)
    assert np.abs(audio[: 5 * TARGET_SAMPLE_RATE]).max() > 0.1
    assert not audio[5 * TARGET_SAMPLE_RATE :].any()


def test_max_seconds_takes_the_centre(data_dir: Path) -> None:
    row = task_rows("audio_context", "train").iloc[0]  # 15 s clip
    full = load_audio_segment(row)
    centre = load_audio_segment(row, max_seconds=10.0)
    start = round(2.5 * TARGET_SAMPLE_RATE)
    np.testing.assert_array_equal(centre, full[start : start + 10 * TARGET_SAMPLE_RATE])


def test_audio_context_defaults_to_ten_seconds(data_dir: Path) -> None:
    sample = load_sample("audio_context", task_rows("audio_context", "train").iloc[0])
    assert sample.audio.shape == (10 * TARGET_SAMPLE_RATE,)
    assert sample.mapped_subtype == "play"
    assert sample.labels["environment_normalized"] == "near door"


# --- bark_detector ---


def test_esc50_dog_clip_has_empty_mask_and_is_dog(data_dir: Path) -> None:
    row = task_rows("bark_detector", "train").iloc[0]
    sample = load_sample("bark_detector", row)
    assert isinstance(sample, BarkDetectorSample)
    assert sample.is_dog
    assert sample.targets.shape == (250, len(SED_CLASSES))
    assert not sample.targets.any()
    assert sample.mask.shape == (len(SED_CLASSES),)
    assert not sample.mask.any()


def test_negative_clip_has_full_mask(data_dir: Path) -> None:
    sample = load_sample("bark_detector", task_rows("bark_detector", "val").iloc[0])
    assert not sample.is_dog
    assert sample.mask.all()


def test_load_errors_name_task_clip_and_file(data_dir: Path) -> None:
    row = task_rows("bark_detector", "val").iloc[0].copy()
    row["file_path"] = "processed/esc50/audio/missing.wav"
    with pytest.raises(RuntimeError, match=r"bark_detector.*esc50/rain\.wav.*missing\.wav"):
        load_sample("bark_detector", row)


# --- Images ---


def test_crop_to_box_moves_keypoints() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[35, 30] = 255
    keypoints = np.array([[30.0, 35.0, 2.0], [0.0, 0.0, 0.0]])
    crop, moved = crop_to_box(image, [10, 20, 40, 30], pad=1.0, keypoints=keypoints)
    assert crop.shape == (30, 40, 3)
    assert moved is not None
    np.testing.assert_allclose(moved, [[20.0, 15.0, 2.0], [0.0, 0.0, 0.0]])
    assert crop[15, 20, 0] == 255


def test_crop_to_box_resizes_keypoints() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.array([[30.0, 35.0, 2.0]])
    crop, moved = crop_to_box(
        image, [10, 20, 40, 30], pad=1.0, out_size=(80, 60), keypoints=keypoints
    )
    assert crop.shape == (60, 80, 3)
    assert moved is not None
    np.testing.assert_allclose(moved, [[40.0, 30.0, 2.0]])


def test_crop_to_box_pads_outside_the_image() -> None:
    image = np.full((20, 20, 3), 7, dtype=np.uint8)
    crop, moved = crop_to_box(image, [0, 0, 20, 20], pad=1.5)
    assert crop.shape == (30, 30, 3)
    assert moved is None
    assert crop[0, 0, 0] == 0
    assert crop[15, 15, 0] == 7


# --- PyTorch dataset ---


def test_task_dataset_pickles(data_dir: Path) -> None:
    pytest.importorskip("torch")
    from wooftrainer.torch_datasets import TaskDataset

    dataset = TaskDataset("pose", "train")
    restored = pickle.loads(pickle.dumps(dataset))
    sample = restored[0]
    assert len(restored) == 1
    assert isinstance(sample, PoseSample)
    assert sample.image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    assert sample.keypoints.shape == (len(KEYPOINT_NAMES), 3)
    np.testing.assert_allclose(sample.box, POSE_BOX)
