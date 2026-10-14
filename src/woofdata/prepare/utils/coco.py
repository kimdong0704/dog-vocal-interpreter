"""Build COCO JSON (images, boxes, keypoints) with paths relative to the data folder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from woofdata.utils_fileio import relative_data_path, write_json


def new_coco(category_names: list[str], keypoint_names: list[str] | None = None) -> dict[str, Any]:
    """Return an empty COCO dict with categories numbered 1.. in the given order.

    keypoint_names, if given, are attached to every category (COCO keypoint format).
    """
    categories: list[dict[str, Any]] = []
    for position, category_name in enumerate(category_names):
        category: dict[str, Any] = {"id": position + 1, "name": category_name}
        if keypoint_names is not None:
            category["keypoints"] = list(keypoint_names)
        categories.append(category)
    return {"images": [], "annotations": [], "categories": categories}


def add_coco_image(
    coco: dict[str, Any],
    image_path: Path,
    width: int,
    height: int,
    data_dir: Path | None = None,
) -> int:
    """Add an image (stored relative to the data folder) and return its new image id."""
    image_id = len(coco["images"]) + 1
    file_name = relative_data_path(image_path, data_dir)
    coco["images"].append(
        {"id": image_id, "file_name": file_name, "width": width, "height": height}
    )
    return image_id


def add_coco_annotation(
    coco: dict[str, Any],
    image_id: int,
    category_name: str,
    bbox: list[float],
    keypoints: list[float] | None = None,
    segmentation: dict[str, Any] | None = None,
) -> int:
    """Add a box [x, y, width, height], optional flat [x, y, v, ...] keypoints and an
    optional segmentation (COCO RLE {"size", "counts"}).

    Returns:
        The new annotation id.

    Raises:
        ValueError: If category_name is not one of the COCO categories.
    """
    category_id = None
    for category in coco["categories"]:
        if category["name"] == category_name:
            category_id = category["id"]
    if category_id is None:
        raise ValueError(f"unknown COCO category {category_name!r}")

    annotation_id = len(coco["annotations"]) + 1
    annotation: dict[str, Any] = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
    }
    if keypoints is not None:
        visibility_flags = keypoints[2::3]  # every third value is v (0 = not labelled)
        labelled_count = 0
        for visibility in visibility_flags:
            if visibility > 0:
                labelled_count += 1
        annotation["keypoints"] = list(keypoints)
        annotation["num_keypoints"] = labelled_count
    if segmentation is not None:
        annotation["segmentation"] = segmentation
    coco["annotations"].append(annotation)
    return annotation_id


def write_coco(coco: dict[str, Any], path: Path) -> None:
    """Write a COCO dict built with new_coco / add_coco_image / add_coco_annotation."""
    write_json(path, coco)
