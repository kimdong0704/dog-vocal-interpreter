"""Load one sample of a task from its manifest row: audio, images, targets and labels.

Every path comes from the manifest, relative to get_data_dir(). Each COCO JSON,
labels.csv and sed_classes.json is read once per process and kept as an index by image
path or clip_id, so loading a sample never re-reads them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile
from PIL import Image

from woofdecoder.config import get_data_dir
from woofdecoder.data.fileio import read_csv
from woofdecoder.data.manifest import processed_dir
from woofdecoder.data.prepare import esc50
from woofdecoder.data.prepare.common import (
    DEFAULT_FRAME_S,
    DEFAULT_WINDOW_S,
    LABELS_FILENAME,
    SED_CLASSES_FILENAME,
    TARGET_SAMPLE_RATE,
)
from woofdecoder.data.prepare.roboflow import TAG_COLUMNS
from woofdecoder.data.prepare.stanfordextra import KEYPOINT_NAMES
from woofdecoder.data.tasks import (
    AudioContextSample,
    AudioEmotionSample,
    BarkDetectorSample,
    PoseSample,
    PostureSample,
    SceneObjectsSample,
    get_task,
)
from woofdecoder.sed_classes import SED_CLASSES

NUM_FRAMES: int = round(DEFAULT_WINDOW_S / DEFAULT_FRAME_S)  # 250 frames per 10 s window
RESIZE_FILTER = Image.Resampling.BILINEAR

Row = pd.Series | dict[str, Any]  # one manifest row


def data_path(relative_path: str) -> Path:
    """Return a manifest path (relative to the data folder) as a full path."""
    return get_data_dir() / relative_path


# --- Cached label files ---


@cache
def read_labels_file(labels_path: Path) -> dict[str, dict[str, str]]:
    """Return {clip_id: labels.csv row without clip_id}, read once per process."""
    labels = read_csv(labels_path).set_index("clip_id")
    return labels.to_dict("index")


def labels_for(row: Row) -> dict[str, str]:
    """Return the labels.csv fields of one row (processed/<source>/labels.csv)."""
    labels_path = processed_dir() / row["source"] / LABELS_FILENAME
    return read_labels_file(labels_path)[row["clip_id"]]


@cache
def read_sed_mask_file(sed_classes_path: Path) -> np.ndarray:
    """Return which SED classes a source labels, shape (len(SED_CLASSES),) bool.

    Raises:
        ValueError: If the file's class order differs from SED_CLASSES.
    """
    with open(sed_classes_path, encoding="utf-8") as sed_classes_file:
        sed_classes = json.load(sed_classes_file)
    if sed_classes["classes"] != SED_CLASSES:
        raise ValueError(f"{sed_classes_path} lists classes {sed_classes['classes']}")
    labelled: list[bool] = []
    for class_name in SED_CLASSES:
        labelled.append(sed_classes["labelled"][class_name])
    return np.array(labelled, dtype=bool)


@cache
def read_coco_file(coco_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[int, str]]:
    """Read a COCO JSON once per process.

    Returns:
        ({image file_name: its annotations}, {category id: category name}). file_name is
        the image path relative to the data folder, the same as the manifest file_path.
    """
    with open(coco_path, encoding="utf-8") as coco_file:
        coco = json.load(coco_file)
    file_name_by_id: dict[int, str] = {}
    annotations_by_file: dict[str, list[dict[str, Any]]] = {}
    for image in coco["images"]:
        file_name_by_id[image["id"]] = image["file_name"]
        annotations_by_file[image["file_name"]] = []
    for annotation in coco["annotations"]:
        annotations_by_file[file_name_by_id[annotation["image_id"]]].append(annotation)
    category_names: dict[int, str] = {}
    for category in coco["categories"]:
        category_names[category["id"]] = category["name"]
    return annotations_by_file, category_names


def annotations_for(row: Row) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Return (the row's COCO annotations, {category id: name}) from its annotation_path."""
    annotations_by_file, category_names = read_coco_file(data_path(row["annotation_path"]))
    return annotations_by_file[row["file_path"]], category_names


def single_annotation(row: Row) -> tuple[dict[str, Any], dict[int, str]]:
    """Return the one COCO annotation of a row's image, and the category names.

    Raises:
        ValueError: If the image has no annotation or more than one.
    """
    annotations, category_names = annotations_for(row)
    if len(annotations) != 1:
        raise ValueError(f"expected one annotation, found {len(annotations)}")
    return annotations[0], category_names


# --- Audio ---


def load_audio_segment(row: Row, max_seconds: float | None = None) -> np.ndarray:
    """Read the row's start_s..end_s from its 16 kHz WAV.

    A window running past the end of the file is padded with zeros. When max_seconds is
    set and the segment is longer, only its centre max_seconds is read (deterministic).

    Returns:
        float32 samples, shape (num_samples,).

    Raises:
        ValueError: If the file is not mono at TARGET_SAMPLE_RATE.
    """
    start_s = float(row["start_s"])
    end_s = float(row["end_s"])
    if max_seconds is not None and end_s - start_s > max_seconds:
        centre_s = (start_s + end_s) / 2
        start_s = centre_s - max_seconds / 2
        end_s = start_s + max_seconds
    start_frame = round(start_s * TARGET_SAMPLE_RATE)
    num_frames = round((end_s - start_s) * TARGET_SAMPLE_RATE)

    samples, sample_rate = soundfile.read(
        data_path(row["file_path"]),
        frames=num_frames,
        start=start_frame,
        dtype="float32",
        fill_value=0.0,  # frames past the end of the file
    )
    if sample_rate != TARGET_SAMPLE_RATE or samples.ndim != 1:
        raise ValueError(f"expected mono {TARGET_SAMPLE_RATE} Hz, got {sample_rate} Hz")
    return samples  # shape (num_frames,)


# --- Images ---


def load_image(row: Row) -> np.ndarray:
    """Read the row's image as RGB, shape (height, width, 3), uint8."""
    with Image.open(data_path(row["file_path"])) as image:
        return np.asarray(image.convert("RGB"))


def padded_bounds(box: np.ndarray | list[float], pad: float) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) in whole pixels of box (x, y, w, h) grown by pad
    around its centre. The bounds may lie outside the image.

    Raises:
        ValueError: If the box has no area.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError(f"box {list(box)} has no area")
    centre_x = x + w / 2
    centre_y = y + h / 2
    left = math.floor(centre_x - w * pad / 2)
    top = math.floor(centre_y - h * pad / 2)
    right = math.ceil(centre_x + w * pad / 2)
    bottom = math.ceil(centre_y + h * pad / 2)
    return left, top, right, bottom


def cut_region(image: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    """Return image[top:bottom, left:right], with zeros where the region leaves the image."""
    height, width = image.shape[:2]
    region = np.zeros((bottom - top, right - left) + image.shape[2:], dtype=image.dtype)
    inside_left = max(left, 0)
    inside_top = max(top, 0)
    inside_right = min(right, width)
    inside_bottom = min(bottom, height)
    if inside_right > inside_left and inside_bottom > inside_top:
        region[inside_top - top : inside_bottom - top, inside_left - left : inside_right - left] = (
            image[inside_top:inside_bottom, inside_left:inside_right]
        )
    return region


def crop_to_box(
    image: np.ndarray,
    box: np.ndarray | list[float],
    pad: float = 1.25,
    out_size: tuple[int, int] | None = None,
    keypoints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Crop image to box (x, y, w, h) grown by pad around its centre (zeros outside).

    Args:
        image: shape (height, width, 3).
        box: x, y, w, h in pixels.
        pad: 1.25 makes the crop 25% wider and taller than the box.
        out_size: (width, height) to resize the crop to; None keeps its size.
        keypoints: shape (num_keypoints, 3) as x, y, visibility; only points with
            visibility > 0 are moved (the others are left as they are).

    Returns:
        (crop, keypoints in crop coordinates, or None if none were given).
    """
    left, top, right, bottom = padded_bounds(box, pad)
    crop = cut_region(image, left, top, right, bottom)  # shape (crop_h, crop_w, 3)
    scale_x = 1.0
    scale_y = 1.0
    if out_size is not None:
        scale_x = out_size[0] / crop.shape[1]
        scale_y = out_size[1] / crop.shape[0]
        crop = np.asarray(Image.fromarray(crop).resize(out_size, RESIZE_FILTER))

    if keypoints is None:
        return crop, None
    moved = np.array(keypoints, dtype=np.float32)  # a copy, shape (num_keypoints, 3)
    labelled = moved[:, 2] > 0  # shape (num_keypoints,)
    moved[labelled, 0] = (moved[labelled, 0] - left) * scale_x
    moved[labelled, 1] = (moved[labelled, 1] - top) * scale_y
    return crop, moved


# --- One loader per task ---


def load_bark_detector(row: Row, max_seconds: float | None) -> BarkDetectorSample:
    """Return a window with its frame targets and the source's labelled-class mask.

    Raises:
        ValueError: If a row without a target file is not an ESC-50 dog clip.
    """
    audio = load_audio_segment(row, max_seconds)
    if row["target_path"]:
        targets = np.load(data_path(row["target_path"]))  # shape (250, 6)
        sed_classes_path = processed_dir() / row["source"] / SED_CLASSES_FILENAME
        mask = read_sed_mask_file(sed_classes_path).copy()  # shape (6,)
        is_dog = bool(targets.any())
    elif row["source"] == esc50.SOURCE and row["original_label"] == esc50.DOG_CATEGORY:
        targets = np.zeros((NUM_FRAMES, len(SED_CLASSES)), dtype=np.float32)
        mask = np.zeros(len(SED_CLASSES), dtype=bool)  # nothing known per frame
        is_dog = True
    else:
        raise ValueError("row has no target_path")
    return BarkDetectorSample(row["clip_id"], audio, targets, mask, is_dog)


def load_audio_context(row: Row, max_seconds: float | None) -> AudioContextSample:
    """Return a clip, its mapped subtype (None if empty) and its labels.csv fields."""
    audio = load_audio_segment(row, max_seconds)
    mapped_subtype = row["mapped_subtype"] or None
    return AudioContextSample(row["clip_id"], audio, mapped_subtype, labels_for(row))


def load_audio_emotion(row: Row, max_seconds: float | None) -> AudioEmotionSample:
    """Return a clip with its arousal and valence from labels.csv."""
    audio = load_audio_segment(row, max_seconds)
    labels = labels_for(row)
    return AudioEmotionSample(row["clip_id"], audio, labels["arousal"], labels["valence"])


def load_pose(row: Row, max_seconds: float | None) -> PoseSample:
    """Return an image, its dog box and its 24 keypoints (max_seconds is unused)."""
    annotation, _ = single_annotation(row)
    box = np.array(annotation["bbox"], dtype=np.float32)  # shape (4,)
    keypoints = np.array(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)  # (24, 3)
    return PoseSample(row["clip_id"], load_image(row), box, keypoints, list(KEYPOINT_NAMES))


def load_posture(row: Row, max_seconds: float | None) -> PostureSample:
    """Return an image, its largest dog box and the posture (max_seconds is unused)."""
    annotation, category_names = single_annotation(row)
    box = np.array(annotation["bbox"], dtype=np.float32)  # shape (4,)
    posture = category_names[annotation["category_id"]]
    return PostureSample(row["clip_id"], load_image(row), box, posture)


def load_scene_objects(row: Row, max_seconds: float | None) -> SceneObjectsSample:
    """Return an image, all its boxes and category ids, and its tags (max_seconds unused)."""
    annotations, _ = annotations_for(row)
    boxes = np.zeros((len(annotations), 4), dtype=np.float32)  # shape (num_boxes, 4)
    category_ids = np.zeros(len(annotations), dtype=np.int64)  # shape (num_boxes,)
    for position, annotation in enumerate(annotations):
        boxes[position] = annotation["bbox"]
        category_ids[position] = annotation["category_id"]
    labels = labels_for(row)
    tags: dict[str, bool] = {}
    for column in TAG_COLUMNS:
        tags[column] = labels[column] == "True"
    return SceneObjectsSample(row["clip_id"], load_image(row), boxes, category_ids, tags)


SampleLoader = Callable[[Row, float | None], Any]
LOADERS: dict[str, SampleLoader] = {
    "bark_detector": load_bark_detector,
    "audio_context": load_audio_context,
    "audio_emotion": load_audio_emotion,
    "pose": load_pose,
    "posture": load_posture,
    "scene_objects": load_scene_objects,
}


def load_sample(task: str, row: Row, max_seconds: float | None = None) -> Any:
    """Load one manifest row as the task's sample dataclass (see tasks.TASKS).

    Args:
        task: A TASKS key.
        row: One row from task_rows(task, ...).
        max_seconds: Audio length cap; None uses the task's default (10 s for the
            audio_context and audio_emotion tasks, no cap otherwise).

    Raises:
        ValueError: If the row's source is not one of the task's sources.
        RuntimeError: If loading fails; the message names the task, clip_id and file.
    """
    task_info = get_task(task)
    if row["source"] not in task_info.sources:
        raise ValueError(f"{task}: {row['clip_id']} is from {row['source']}, not this task")
    if max_seconds is None:
        max_seconds = task_info.default_max_seconds
    try:
        return LOADERS[task](row, max_seconds)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        message = f"{task}: cannot load {row['clip_id']} (file {row['file_path']}): {error!r}"
        raise RuntimeError(message) from error
