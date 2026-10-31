"""Stanford Dogs converter: the shared image index, dog boxes and split for pose and body.

Raw file: raw/stanford_dogs/stanford-dogs-dataset.zip (Kaggle mirror of the official
images.tar and annotation.tar), holding:
    images/Images/<wnid>-<breed>/<image key>.jpg
        20,580 images, keys unique
    annotations/Annotation/<wnid>-<breed>/<image key>
        PASCAL VOC XML: <size> (equal to the JPEG size for all 20,580 images, so images
        are never opened) and one <object><bndbox> per dog (xmin, ymin, xmax, ymax in pixels)
The split also needs raw/stanfordextra/ (its official split files), see stanford_shared.
This source writes no manifest rows: it provides images/, index.csv and boxes.csv, and
must be prepared before stanfordextra, dogposecv and dog_pose.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd

from woofdata.prepare.manifest import ManifestRow
from woofdata.prepare.sources.stanford_shared import (
    BOXES_FILENAME,
    INDEX_FILENAME,
    image_key,
    read_stanfordextra,
    shared_split,
)
from woofdata.prepare.utils.archives import (
    extract_members,
    list_archive_members,
    open_archive,
    open_member,
    select_members,
)
from woofdata.utils_fileio import relative_data_path, write_csv

ARCHIVE: str = "stanford-dogs-dataset.zip"
IMAGES_PATTERN: str = "images/Images/*.jpg"
ANNOTATIONS_PATTERN: str = "annotations/Annotation/*"
ARCHIVE_IMAGES_FOLDER: str = "images/Images"
IMAGES_DIRNAME: str = "images"
STANFORDEXTRA_SOURCE: str = "stanfordextra"
INDEX_COLUMNS: list[str] = [
    "image_key",
    "path",
    "wnid",
    "breed",
    "width",
    "height",
    "num_dogs",
    "split",
]
BOX_COLUMNS: list[str] = ["image_key", "x", "y", "w", "h"]

Box = tuple[int, int, int, int]  # COCO x, y, w, h in pixels
VocAnnotation = tuple[int, int, list[Box]]  # image width, height, dog boxes


def extract_images(raw_dir: Path, out_dir: Path) -> list[Path]:
    """Extract every image to out_dir/images/<wnid>-<breed>/<key>.jpg; return the paths.

    The archive folder is "images/Images"; it is stripped and replaced by lower case
    "images" so paths read the same on case-sensitive and case-insensitive file systems.
    """
    extract_members(
        raw_dir / ARCHIVE,
        IMAGES_PATTERN,
        out_dir / IMAGES_DIRNAME,
        strip_prefix=ARCHIVE_IMAGES_FOLDER + "/",
    )
    return sorted((out_dir / IMAGES_DIRNAME).glob("*/*.jpg"))


def find_int(element: ElementTree.Element, path: str) -> int:
    """Return the integer text of the element at path.

    Raises:
        ValueError: If there is no element at path, or it has no text.
    """
    text = element.findtext(path)
    if text is None:
        raise ValueError(f"VOC XML has no <{path}>")
    return int(text)


def parse_voc(xml_bytes: bytes) -> VocAnnotation:
    """Return (width, height, dog boxes as COCO (x, y, w, h)) from one PASCAL VOC XML file.

    w = xmax - xmin and h = ymax - ymin, the usual VOC -> COCO conversion.
    """
    root = ElementTree.fromstring(xml_bytes)
    width = find_int(root, "size/width")
    height = find_int(root, "size/height")
    boxes: list[Box] = []
    for box in root.iter("bndbox"):
        x_min = find_int(box, "xmin")
        y_min = find_int(box, "ymin")
        x_max = find_int(box, "xmax")
        y_max = find_int(box, "ymax")
        boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    return width, height, boxes


def read_annotations(raw_dir: Path) -> dict[str, VocAnnotation]:
    """Return {image key: (width, height, dog boxes)} from the archive's annotations/."""
    archive_path = raw_dir / ARCHIVE
    member_names = select_members(sorted(list_archive_members(archive_path)), ANNOTATIONS_PATTERN)
    annotations: dict[str, VocAnnotation] = {}
    with open_archive(archive_path) as archive:
        for member_name in member_names:
            with open_member(archive, member_name) as member_file:
                annotations[image_key(member_name)] = parse_voc(member_file.read())
    return annotations


def index_row(
    image_path: Path, annotation: VocAnnotation, official_splits: dict[str, str]
) -> dict[str, Any]:
    """Return one index.csv row (width and height from the VOC XML)."""
    key = image_key(image_path.name)
    wnid, _, breed = image_path.parent.name.partition("-")
    width, height, boxes = annotation
    return {
        "image_key": key,
        "path": relative_data_path(image_path),
        "wnid": wnid,
        "breed": breed,
        "width": width,
        "height": height,
        "num_dogs": len(boxes),
        "split": shared_split(key, official_splits),
    }


def print_report(index_rows: list[dict[str, Any]], official_splits: dict[str, str]) -> None:
    """Print images, multi-dog images, and images per split (official vs hashed)."""
    split_counts: Counter[tuple[str, str]] = Counter()
    multi_dog_count = 0
    for row in index_rows:
        origin = "hashed"
        if row["image_key"] in official_splits:
            origin = "StanfordExtra"
        split_counts[(row["split"], origin)] += 1
        if row["num_dogs"] > 1:
            multi_dog_count += 1
    print(f"  {len(index_rows)} images, {multi_dog_count} with more than one dog")
    print(f"  images per (split, origin): {dict(sorted(split_counts.items()))}")


def prepare(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Extract the images and write index.csv and boxes.csv; return no manifest rows.

    Raises:
        ValueError: If an image has no box file, or the official split names an image
            that is not in the archive.
    """
    _, official_splits = read_stanfordextra(raw_dir.parent / STANFORDEXTRA_SOURCE)
    print(f"  StanfordExtra official split sizes: {dict(Counter(official_splits.values()))}")
    annotations = read_annotations(raw_dir)
    image_paths = extract_images(raw_dir, out_dir)

    index_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    for image_path in image_paths:
        key = image_key(image_path.name)
        if key not in annotations:
            raise ValueError(f"{key} has no annotation in {ARCHIVE}")
        index_rows.append(index_row(image_path, annotations[key], official_splits))
        for x, y, w, h in annotations[key][2]:
            box_rows.append({"image_key": key, "x": x, "y": y, "w": w, "h": h})

    indexed_keys: set[str] = set()
    for row in index_rows:
        indexed_keys.add(row["image_key"])
    missing = sorted(set(official_splits) - indexed_keys)
    if missing:
        raise ValueError(f"StanfordExtra images not in {ARCHIVE}: {missing[:10]}")

    write_csv(pd.DataFrame(index_rows, columns=INDEX_COLUMNS), out_dir / INDEX_FILENAME)
    write_csv(pd.DataFrame(box_rows, columns=BOX_COLUMNS), out_dir / BOXES_FILENAME)
    print_report(index_rows, official_splits)
    return []
