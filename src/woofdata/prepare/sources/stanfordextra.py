"""StanfordExtra converter: 24 dog keypoints per image -> COCO keypoint JSON per split.

Raw file: raw/stanfordextra/stanford-dogs-dataset-for-animal-pose-estimation.zip (Kaggle
mirror) with StanfordExtra_V12/StanfordExtra_v12.json (12,538 entries: img_path, img_width,
img_height, img_bbox as [x, y, w, h], is_multiple_dogs, joints as 24 x [x, y, visible 0/1],
seg as a COCO RLE string) and the three official split .npy files. The mirror has no
keypoint_definitions file, so KEYPOINT_NAMES follows the agreed StanfordExtra order.
Images are not copied: COCO file names point to processed/stanford_dogs/images/, and the
split comes from processed/stanford_dogs/index.csv (prepare stanford_dogs first).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from woofdata.download.datasets import dataset_config
from woofdata.path import get_data_dir
from woofdata.prepare.manifest import ManifestRow, make_clip_id
from woofdata.prepare.sources.stanford_shared import (
    STANFORD_DOGS,
    image_key,
    read_index,
    read_stanfordextra,
)
from woofdata.prepare.utils.coco import add_coco_annotation, add_coco_image, new_coco, write_coco
from woofdata.prepare.utils.label_files import write_labels
from woofdata.utils_fileio import relative_data_path

SOURCE: str = "stanfordextra"
SPLITS: tuple[str, ...] = ("train", "val", "test")
CATEGORY: str = "dog"
KEYPOINT_NAMES: list[str] = [
    "left_front_paw",
    "left_front_knee",
    "left_front_elbow",
    "left_rear_paw",
    "left_rear_knee",
    "left_rear_elbow",
    "right_front_paw",
    "right_front_knee",
    "right_front_elbow",
    "right_rear_paw",
    "right_rear_knee",
    "right_rear_elbow",
    "tail_start",
    "tail_end",
    "left_ear_base",
    "right_ear_base",
    "nose",
    "chin",
    "left_ear_tip",
    "right_ear_tip",
    "left_eye",
    "right_eye",
    "withers",
    "throat",
]
# Eyes, withers and throat are empty in this dataset: always visibility 0.
EMPTY_KEYPOINTS: tuple[int, ...] = (20, 21, 22, 23)
LABELLED_VISIBILITY: int = 2  # COCO "labelled and visible"
SOURCE_LICENSE: str = "MIT"
NOTES: str = "annotations MIT; images under ImageNet terms (non-commercial research)"
LABEL_COLUMNS: list[str] = [
    "clip_id",
    "image_key",
    "img_path",
    "img_width",
    "img_height",
    "img_bbox",
    "is_multiple_dogs",
    "joints",
]


def keypoints_path(out_dir: Path, split: str) -> Path:
    """Return out_dir/keypoints_<split>.json."""
    return out_dir / f"keypoints_{split}.json"


def coco_keypoints(joints: list[list[float]]) -> list[float]:
    """Return flat COCO keypoints [x, y, v] * 24: v = 2 where labelled, else 0 (x = y = 0).

    The EMPTY_KEYPOINTS are always 0, whatever the JSON says.
    """
    flat: list[float] = []
    for position, (x, y, visible) in enumerate(joints):
        if visible > 0 and position not in EMPTY_KEYPOINTS:
            flat.extend([float(x), float(y), LABELLED_VISIBILITY])
        else:
            flat.extend([0.0, 0.0, 0])
    return flat


def label_row(clip_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Return one labels.csv row: the entry's own fields (seg left out: it is in the JSON)."""
    return {
        "clip_id": clip_id,
        "image_key": image_key(entry["img_path"]),
        "img_path": entry["img_path"],
        "img_width": entry["img_width"],
        "img_height": entry["img_height"],
        "img_bbox": json.dumps(entry["img_bbox"]),
        "is_multiple_dogs": entry["is_multiple_dogs"],
        "joints": json.dumps(entry["joints"]),
    }


def make_row(key: str, image: dict[str, str], out_dir: Path, source_version: str) -> ManifestRow:
    """Return the manifest row of one image (split from the shared index)."""
    return ManifestRow(
        clip_id=make_clip_id(SOURCE, key),
        source=SOURCE,
        source_version=source_version,
        source_license=SOURCE_LICENSE,
        dog_id=key,
        leakage_group=f"{STANFORD_DOGS}/{key}",
        split=image["split"],
        modality="image",
        file_path=image["path"],
        annotation_path=relative_data_path(keypoints_path(out_dir, image["split"])),
        label_type="keypoints",
        label_provenance="expert",
        notes=NOTES,
    )


def add_entry(coco: dict[str, Any], entry: dict[str, Any], image: dict[str, str]) -> None:
    """Add one entry's image (from the shared index) and its dog annotation to coco."""
    width = int(image["width"])
    height = int(image["height"])
    image_id = add_coco_image(coco, get_data_dir() / image["path"], width, height)
    segmentation = {"size": [height, width], "counts": entry["seg"]}
    keypoints = coco_keypoints(entry["joints"])
    add_coco_annotation(coco, image_id, CATEGORY, entry["img_bbox"], keypoints, segmentation)


def print_report(rows: list[ManifestRow], cocos: dict[str, dict[str, Any]]) -> None:
    """Print images per split and the % of images where each keypoint is labelled."""
    split_counts: Counter[str] = Counter()
    for row in rows:
        split_counts[row.split] += 1
    print(f"  images per split: {dict(sorted(split_counts.items()))}")

    keypoint_lists: list[list[float]] = []
    for coco in cocos.values():
        for annotation in coco["annotations"]:
            keypoint_lists.append(annotation["keypoints"])
    visibility = np.array(keypoint_lists)[:, 2::3]  # shape (num_images, 24)
    labelled_percent = (visibility > 0).mean(axis=0) * 100  # shape (24,)
    for position, name in enumerate(KEYPOINT_NAMES):
        print(f"  {position:>2} {name:<18} {labelled_percent[position]:5.1f}%")


def prepare(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Write keypoints_{train,val,test}.json, labels.csv and one row per image."""
    entries, _ = read_stanfordextra(raw_dir)
    index = read_index()
    source_version = f"kaggle_v{dataset_config(SOURCE)['args']['version']}"
    cocos: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        cocos[split] = new_coco([CATEGORY], KEYPOINT_NAMES)

    rows: list[ManifestRow] = []
    label_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    size_mismatches = 0
    for entry in entries:
        key = image_key(entry["img_path"])
        if key not in index:
            missing.append(key)
            continue
        image = index[key]
        jpeg_size = (int(image["width"]), int(image["height"]))
        if jpeg_size != (entry["img_width"], entry["img_height"]):
            size_mismatches += 1
        add_entry(cocos[image["split"]], entry, image)
        rows.append(make_row(key, image, out_dir, source_version))
        label_rows.append(label_row(rows[-1].clip_id, entry))

    for split in SPLITS:
        write_coco(cocos[split], keypoints_path(out_dir, split))
    write_labels(out_dir, label_rows, LABEL_COLUMNS)
    print(f"  entries not in the stanford_dogs index: {len(missing)} {missing[:10]}")
    print(f"  entries whose width/height differ from the JPEG: {size_mismatches}")
    print_report(rows, cocos)
    return rows
