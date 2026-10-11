"""DogPoseCV converter: one posture label per Stanford Dogs image -> COCO JSON per split.

Raw files in raw/dogposecv/data/ (pinned Hugging Face commit):
    labels.tar.gz   "labels/<wnid>-<breed>.csv" with columns id ("<image key>.jpg"), label;
                    119 breeds (no file for n02097209 standard schnauzer), plus macOS "._"
                    copies that are skipped
    images.tar.gz   byte-for-byte the Stanford Dogs images (checked on 50 random files),
                    so it is not extracted: rows point to processed/stanford_dogs/images/.
Each image gets one annotation, on its largest dog box from processed/stanford_dogs/
boxes.csv. Prepare stanford_dogs first.
"""

from __future__ import annotations

import io
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from woofdecoder.config import get_data_dir
from woofdecoder.data.fileio import read_csv, relative_data_path, write_csv
from woofdecoder.data.manifest import ManifestRow, make_clip_id, processed_dir
from woofdecoder.data.prepare.common import (
    add_coco_annotation,
    add_coco_image,
    dataset_config,
    new_coco,
    write_coco,
    write_labels,
)
from woofdecoder.data.prepare.stanford_shared import (
    BOXES_FILENAME,
    STANFORD_DOGS,
    image_key,
    read_index,
)

SOURCE: str = "dogposecv"
LABELS_ARCHIVE: str = "data/labels.tar.gz"
APPLE_DOUBLE_PREFIX: str = "._"
LABEL_SUFFIX: str = ".csv"
SPLITS: tuple[str, ...] = ("train", "val", "test")
# The dataset's own posture names, used as the COCO categories.
POSTURES: list[str] = ["standing", "sitting", "lying", "undefined"]
SOURCE_LICENSE: str = "Apache-2.0"
NOTES: str = (
    "labelled by the dataset's two curators; images under ImageNet terms (non-commercial research)"
)
LABEL_COLUMNS: list[str] = ["clip_id", "image_key", "posture", "num_dogs"]
SKIPPED_FILENAME: str = "skipped.csv"
SKIPPED_COLUMNS: list[str] = ["image_key", "breed", "reason"]
NO_LABEL: str = "no posture label in labels.tar.gz"

Box = tuple[float, float, float, float]  # x, y, w, h in pixels


def read_postures(raw_dir: Path) -> dict[str, str]:
    """Return {image key: posture} from every breed CSV in labels.tar.gz.

    Raises:
        ValueError: If a posture is not in POSTURES or an image is labelled twice.
    """
    postures: dict[str, str] = {}
    with tarfile.open(raw_dir / LABELS_ARCHIVE) as archive:
        for member in archive.getmembers():  # order does not matter: callers sort by key
            file_name = PurePosixPath(member.name).name
            is_label_file = member.isfile() and file_name.endswith(LABEL_SUFFIX)
            if not is_label_file or file_name.startswith(APPLE_DOUBLE_PREFIX):
                continue
            labels = pd.read_csv(io.BytesIO(archive.extractfile(member).read()), dtype=str)
            for image_id, posture in zip(labels["id"], labels["label"], strict=True):
                if posture not in POSTURES:
                    raise ValueError(f"{file_name}: unknown posture {posture!r}")
                key = image_key(image_id)
                if key in postures:
                    raise ValueError(f"{key} is labelled twice")
                postures[key] = posture
    return postures


def check_keys_in_index(postures: dict[str, str], index: dict[str, dict[str, str]]) -> None:
    """Raise ValueError naming the labelled images that are not in the stanford_dogs index."""
    missing: list[str] = []
    for key in sorted(postures):
        if key not in index:
            missing.append(key)
    if missing:
        raise ValueError(f"{len(missing)} labelled images are not in the index: {missing[:10]}")


def largest_boxes() -> dict[str, Box]:
    """Return {image key: its largest dog box (by area)} from stanford_dogs boxes.csv."""
    boxes = read_csv(processed_dir() / STANFORD_DOGS / BOXES_FILENAME)
    largest: dict[str, Box] = {}
    for key, x, y, w, h in boxes[["image_key", "x", "y", "w", "h"]].itertuples(index=False):
        box = (float(x), float(y), float(w), float(h))
        if key not in largest or box[2] * box[3] > largest[key][2] * largest[key][3]:
            largest[key] = box
    return largest


def write_skipped(out_dir: Path, postures: dict[str, str], index: dict[str, dict[str, str]]) -> int:
    """Write skipped.csv: the index images with no posture label (no rows); return the count."""
    skipped_rows: list[dict[str, str]] = []
    for key, image in sorted(index.items()):
        if key not in postures:
            skipped_rows.append({"image_key": key, "breed": image["breed"], "reason": NO_LABEL})
    write_csv(pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS), out_dir / SKIPPED_FILENAME)
    return len(skipped_rows)


def posture_path(out_dir: Path, split: str) -> Path:
    """Return out_dir/posture_<split>.json."""
    return out_dir / f"posture_{split}.json"


def make_row(
    key: str, posture: str, image: dict[str, str], out_dir: Path, source_version: str
) -> ManifestRow:
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
        annotation_path=relative_data_path(posture_path(out_dir, image["split"])),
        original_label=posture,
        label_type="posture",
        label_provenance="uploader",
        notes=NOTES,
    )


def print_report(rows: list[ManifestRow], label_rows: list[dict[str, Any]]) -> None:
    """Print posture counts per split and the number of images with more than one dog."""
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counts[(row.split, row.original_label)] += 1
    for split in SPLITS:
        split_counts: dict[str, int] = {}
        for posture in POSTURES:
            split_counts[posture] = counts[(split, posture)]
        print(f"  {split}: {split_counts}")
    multi_dog_count = 0
    for label in label_rows:
        if int(label["num_dogs"]) > 1:
            multi_dog_count += 1
    print(f"  images with more than one dog: {multi_dog_count}")


def prepare(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Write posture_{train,val,test}.json, labels.csv and one row per labelled image."""
    postures = read_postures(raw_dir)
    index = read_index()
    check_keys_in_index(postures, index)
    boxes = largest_boxes()
    source_version = dataset_config(SOURCE)["revision"]
    cocos: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        cocos[split] = new_coco(POSTURES)

    rows: list[ManifestRow] = []
    label_rows: list[dict[str, Any]] = []
    for key, posture in sorted(postures.items()):
        image = index[key]
        coco = cocos[image["split"]]
        image_path = get_data_dir() / image["path"]
        image_id = add_coco_image(coco, image_path, int(image["width"]), int(image["height"]))
        add_coco_annotation(coco, image_id, posture, list(boxes[key]))
        rows.append(make_row(key, posture, image, out_dir, source_version))
        label_rows.append(
            {
                "clip_id": rows[-1].clip_id,
                "image_key": key,
                "posture": posture,
                "num_dogs": image["num_dogs"],
            }
        )

    for split in SPLITS:
        write_coco(cocos[split], posture_path(out_dir, split))
    write_labels(out_dir, label_rows, LABEL_COLUMNS)
    skipped = write_skipped(out_dir, postures, index)
    print(f"  {skipped} stanford_dogs images have no posture label: listed in {SKIPPED_FILENAME}")
    print_report(rows, label_rows)
    return rows
