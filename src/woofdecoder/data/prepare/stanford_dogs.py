"""Stanford Dogs converter: the shared image index, dog boxes and split for pose and body.

Raw files in raw/stanford_dogs/:
    images.tar      "Images/<wnid>-<breed>/<image key>.jpg", 20,580 images, keys unique
    annotation.tar  "Annotation/<wnid>-<breed>/<image key>", PASCAL VOC XML: <size> (equal to
                    the JPEG size for all 20,580 images, so images are never opened) and
                    one <object><bndbox> per dog (xmin, ymin, xmax, ymax in pixels)
    lists.tar       the authors' breed-classification lists (not used here)
The split also needs raw/stanfordextra/ (its official split files), see stanford_shared.
This source writes no manifest rows: it provides images/, index.csv and boxes.csv, and
must be prepared before stanfordextra, dogposecv and dog_pose.
"""

from __future__ import annotations

import tarfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd

from woofdecoder.data.fileio import relative_data_path, write_csv
from woofdecoder.data.manifest import ManifestRow
from woofdecoder.data.prepare.common import extract_members
from woofdecoder.data.prepare.stanford_shared import (
    BOXES_FILENAME,
    INDEX_FILENAME,
    image_key,
    read_stanfordextra,
    shared_split,
)

IMAGES_ARCHIVE: str = "images.tar"
ANNOTATION_ARCHIVE: str = "annotation.tar"
IMAGES_PATTERN: str = "Images/*.jpg"
ARCHIVE_IMAGES_FOLDER: str = "Images"
IMAGES_DIRNAME: str = "images"
EXTRACT_DIRNAME: str = "extract"
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

    The archive folder is "Images"; it is renamed to lower case so paths read the same on
    case-sensitive and case-insensitive file systems.
    """
    extract_dir = out_dir / EXTRACT_DIRNAME
    extract_members(raw_dir / IMAGES_ARCHIVE, IMAGES_PATTERN, extract_dir)
    (extract_dir / ARCHIVE_IMAGES_FOLDER).rename(out_dir / IMAGES_DIRNAME)
    extract_dir.rmdir()
    return sorted((out_dir / IMAGES_DIRNAME).glob("*/*.jpg"))


def parse_voc(xml_bytes: bytes) -> VocAnnotation:
    """Return (width, height, dog boxes as COCO (x, y, w, h)) from one PASCAL VOC XML file.

    w = xmax - xmin and h = ymax - ymin, the usual VOC -> COCO conversion.
    """
    root = ElementTree.fromstring(xml_bytes)
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))
    boxes: list[Box] = []
    for box in root.iter("bndbox"):
        x_min = int(box.findtext("xmin"))
        y_min = int(box.findtext("ymin"))
        x_max = int(box.findtext("xmax"))
        y_max = int(box.findtext("ymax"))
        boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    return width, height, boxes


def read_annotations(raw_dir: Path) -> dict[str, VocAnnotation]:
    """Return {image key: (width, height, dog boxes)} from annotation.tar."""
    annotations: dict[str, VocAnnotation] = {}
    with tarfile.open(raw_dir / ANNOTATION_ARCHIVE) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_file = archive.extractfile(member)
            annotations[image_key(member.name)] = parse_voc(member_file.read())
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
            that is not in images.tar.
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
            raise ValueError(f"{key} has no file in {ANNOTATION_ARCHIVE}")
        index_rows.append(index_row(image_path, annotations[key], official_splits))
        for x, y, w, h in annotations[key][2]:
            box_rows.append({"image_key": key, "x": x, "y": y, "w": w, "h": h})

    indexed_keys: set[str] = set()
    for row in index_rows:
        indexed_keys.add(row["image_key"])
    missing = sorted(set(official_splits) - indexed_keys)
    if missing:
        raise ValueError(f"StanfordExtra images not in {IMAGES_ARCHIVE}: {missing[:10]}")

    write_csv(pd.DataFrame(index_rows, columns=INDEX_COLUMNS), out_dir / INDEX_FILENAME)
    write_csv(pd.DataFrame(box_rows, columns=BOX_COLUMNS), out_dir / BOXES_FILENAME)
    print_report(index_rows, official_splits)
    return []
