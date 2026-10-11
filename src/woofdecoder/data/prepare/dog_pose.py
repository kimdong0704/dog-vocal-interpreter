"""Ultralytics Dog-Pose check: the same StanfordExtra labels in YOLO format, not used directly.

Raw file: raw/dog_pose/dog-pose.zip with images/{train,val}/<key>.jpg and
labels/{train,val}/<key>.txt (one line per dog: class, box cx cy w h, then 24 x [x y v],
all normalised to the image size). Its "val" folder is StanfordExtra's TEST split, so tuning
on it would leak the test set; owners use the stanfordextra COCO JSON instead.
This converter only reads the label files, checks them against the shared split and the
StanfordExtra JSON, and writes processed/dog_pose/consistency.txt. No rows, no images.
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from woofdecoder.data.manifest import ManifestRow
from woofdecoder.data.prepare.stanford_shared import image_key, read_index, read_stanfordextra

ARCHIVE_FILENAME: str = "dog-pose.zip"
LABELS_FOLDER: str = "labels"
# Dog-Pose folder -> the StanfordExtra split it holds.
FOLDER_TO_SPLIT: dict[str, str] = {"train": "train", "val": "test"}
STANFORDEXTRA_SOURCE: str = "stanfordextra"
NUM_KEYPOINTS: int = 24
BOX_VALUES: int = 5  # class, cx, cy, w, h before the keypoints
NUM_SPOT_CHECKS: int = 50
SPOT_CHECK_SEED: int = 0
MAX_PIXEL_ERROR: float = 2.0
CONSISTENCY_FILENAME: str = "consistency.txt"


def read_yolo_labels(archive_path: Path) -> dict[str, dict[str, list[float]]]:
    """Return {folder: {image key: the first label line's values}} without touching images."""
    labels: dict[str, dict[str, list[float]]] = {}
    for folder in FOLDER_TO_SPLIT:
        labels[folder] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member_name in sorted(archive.namelist()):
            parts = PurePosixPath(member_name).parts
            if len(parts) != 3 or parts[0] != LABELS_FOLDER or not member_name.endswith(".txt"):
                continue
            first_line = archive.read(member_name).decode("utf-8").splitlines()[0]
            values: list[float] = []
            for value in first_line.split():
                values.append(float(value))
            labels[parts[1]][image_key(member_name)] = values
    return labels


def compare_folders(
    labels: dict[str, dict[str, list[float]]], official_splits: dict[str, str]
) -> list[str]:
    """Return report lines comparing each folder's images with its StanfordExtra split."""
    lines: list[str] = []
    for folder, split in FOLDER_TO_SPLIT.items():
        expected: set[str] = set()
        for key, official_split in official_splits.items():
            if official_split == split:
                expected.add(key)
        found = set(labels[folder])
        missing = len(expected - found)
        extra = len(found - expected)
        status = "OK" if missing == 0 and extra == 0 else "MISMATCH"
        lines.append(
            f"{status}: Dog-Pose {folder} ({len(found)} images) vs our shared {split} images "
            f"from StanfordExtra ({len(expected)}): {missing} missing, {extra} extra"
        )
    return lines


def yolo_keypoints_px(values: list[float], width: int, height: int) -> np.ndarray:
    """Return one YOLO label's keypoints in pixels, shape (24, 3) as [x, y, v]."""
    keypoints = np.array(values[BOX_VALUES:]).reshape(NUM_KEYPOINTS, 3)  # shape (24, 3)
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height
    return keypoints


def spot_check(
    labels: dict[str, dict[str, list[float]]],
    joints_by_key: dict[str, np.ndarray],
    index: dict[str, dict[str, str]],
) -> list[str]:
    """Return report lines: max pixel error and visibility disagreements on 50 random images."""
    all_labels: dict[str, list[float]] = {}
    for folder_labels in labels.values():
        all_labels.update(folder_labels)
    keys = random.Random(SPOT_CHECK_SEED).sample(sorted(all_labels), NUM_SPOT_CHECKS)

    max_error = 0.0
    visibility_disagreements = 0
    for key in keys:
        image = index[key]
        yolo = yolo_keypoints_px(all_labels[key], int(image["width"]), int(image["height"]))
        joints = joints_by_key[key]  # shape (24, 3), visibility 0/1
        yolo_labelled = yolo[:, 2] > 0  # shape (24,)
        joints_labelled = joints[:, 2] > 0  # shape (24,)
        visibility_disagreements += int((yolo_labelled != joints_labelled).sum())
        both = yolo_labelled & joints_labelled
        if both.any():
            errors = np.abs(yolo[both, :2] - joints[both, :2])  # shape (num_both, 2)
            max_error = max(max_error, float(errors.max()))
    status = "MISMATCH"
    if max_error <= MAX_PIXEL_ERROR and visibility_disagreements == 0:
        status = "OK"
    line = (
        f"{status}: {NUM_SPOT_CHECKS} images (seed {SPOT_CHECK_SEED}): max keypoint error "
        f"{max_error:.2f} px (limit {MAX_PIXEL_ERROR}), "
        f"{visibility_disagreements} keypoints labelled in only one of the two"
    )
    return [line]


def prepare(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Check Dog-Pose against the shared split and StanfordExtra; write consistency.txt."""
    entries, official_splits = read_stanfordextra(raw_dir.parent / STANFORDEXTRA_SOURCE)
    joints_by_key: dict[str, np.ndarray] = {}
    for entry in entries:
        joints_by_key[image_key(entry["img_path"])] = np.array(entry["joints"], dtype=float)
    index = read_index()
    labels = read_yolo_labels(raw_dir / ARCHIVE_FILENAME)

    lines = [
        "Dog-Pose (Ultralytics) consistency check. Its val folder is StanfordExtra TEST:",
        "do not tune on it; use processed/stanfordextra/keypoints_*.json instead.",
    ]
    lines.extend(compare_folders(labels, official_splits))
    lines.extend(spot_check(labels, joints_by_key, index))
    consistency_text = "\n".join(lines) + "\n"
    (out_dir / CONSISTENCY_FILENAME).write_text(consistency_text, encoding="utf-8", newline="\n")
    for line in lines:
        print(f"  {line}")
    return []
