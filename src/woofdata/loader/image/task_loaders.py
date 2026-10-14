"""Loaders for the image tasks: pose, posture and scene_objects."""

from __future__ import annotations

import numpy as np
from PIL import Image

from woofdata.loader.annotations import Row, annotations_for, labels_for, single_annotation
from woofdata.path import data_path
from woofdata.prepare.sources.roboflow import TAG_COLUMNS
from woofdata.prepare.sources.stanfordextra import KEYPOINT_NAMES
from woofdata.tasks import PoseSample, PostureSample, SceneObjectsSample


def load_image(row: Row) -> np.ndarray:
    """Read the row's image as RGB, shape (height, width, 3), uint8."""
    with Image.open(data_path(row["file_path"])) as image:
        return np.asarray(image.convert("RGB"))


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
