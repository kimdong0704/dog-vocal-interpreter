"""Pieces shared by the four Stanford Dogs-based converters: image keys, the split, the index.

Every pose and body source uses the one split in processed/stanford_dogs/index.csv:
images in StanfordExtra's official split files keep that split; every other image gets a
split from the sha1 of its image key, the same on every machine.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from woofdata.prepare.manifest import processed_dir
from woofdata.prepare.utils.hash_splits import split_from_hash
from woofdata.utils_fileio import read_csv

STANFORD_DOGS: str = "stanford_dogs"
INDEX_FILENAME: str = "index.csv"
BOXES_FILENAME: str = "boxes.csv"

STANFORDEXTRA_ARCHIVE: str = "stanford-dogs-dataset-for-animal-pose-estimation.zip"
STANFORDEXTRA_JSON: str = "StanfordExtra_V12/StanfordExtra_v12.json"
# Official split -> its index file inside the archive (indices into the JSON list).
STANFORDEXTRA_SPLIT_FILES: dict[str, str] = {
    "train": "StanfordExtra_V12/train_stanford_StanfordExtra_v12.npy",
    "val": "StanfordExtra_V12/val_stanford_StanfordExtra_v12.npy",
    "test": "StanfordExtra_V12/test_stanford_StanfordExtra_v12.npy",
}

# Images outside StanfordExtra: sha1(image key) mod 100 below these bounds -> the split.
TRAIN_BELOW: int = 70
VAL_BELOW: int = 85


def image_key(path: str) -> str:
    """Return the image key, the file name without folder or suffix: "n02085620_10074"."""
    return PurePosixPath(path).stem


def hash_split(key: str) -> str:
    """Return train / val / test from sha1(key) mod 100 (<70 train, <85 val, else test)."""
    return split_from_hash(key, TRAIN_BELOW, VAL_BELOW)


def shared_split(key: str, official_splits: dict[str, str]) -> str:
    """Return the official StanfordExtra split of an image if it has one, else hash_split."""
    if key in official_splits:
        return official_splits[key]
    return hash_split(key)


def read_stanfordextra(raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read raw/stanfordextra/: the JSON entries and {image key: official split}.

    Raises:
        ValueError: If an image is listed in more than one split file.
    """
    with zipfile.ZipFile(raw_dir / STANFORDEXTRA_ARCHIVE) as archive:
        entries = json.loads(archive.read(STANFORDEXTRA_JSON))
        official_splits: dict[str, str] = {}
        for split, split_file in STANFORDEXTRA_SPLIT_FILES.items():
            positions = np.load(io.BytesIO(archive.read(split_file)))  # shape (num_images,)
            for position in positions.tolist():
                key = image_key(entries[position]["img_path"])
                if key in official_splits:
                    raise ValueError(f"{key} is in two StanfordExtra split files")
                official_splits[key] = split
    return entries, official_splits


def read_index(data_dir: Path | None = None) -> dict[str, dict[str, str]]:
    """Return {image key: index.csv row} from processed/stanford_dogs/.

    Raises:
        ValueError: If stanford_dogs has not been prepared yet.
    """
    index_path = processed_dir(data_dir) / STANFORD_DOGS / INDEX_FILENAME
    if not index_path.is_file():
        raise ValueError(f"{index_path} is missing; prepare stanford_dogs first")
    index: dict[str, dict[str, str]] = {}
    for record in read_csv(index_path).to_dict("records"):
        index[record["image_key"]] = record
    return index
