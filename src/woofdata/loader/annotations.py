"""Cached label files: labels.csv, sed_classes.json and COCO JSON.

Each file is read once per process and kept as an index by clip_id or image path, so
loading a sample never re-reads them.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from woofdata.path import data_path
from woofdata.prepare.manifest import processed_dir
from woofdata.prepare.utils.label_files import LABELS_FILENAME
from woofdata.utils_fileio import read_csv
from woofdecoder.sed_classes import SED_CLASSES


class Row(Protocol):
    """One manifest row, read by column name: a pandas Series or a plain dict."""

    def __getitem__(self, column: str, /) -> Any: ...


@cache
def read_labels_file(labels_path: Path) -> dict[str, dict[str, str]]:
    """Return {clip_id: labels.csv row without clip_id}, read once per process."""
    labels = read_csv(labels_path).set_index("clip_id")
    # read_csv reads every column as str, so the index and the values are all str.
    return cast("dict[str, dict[str, str]]", labels.to_dict("index"))


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
