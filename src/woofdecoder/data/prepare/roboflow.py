"""Roboflow COCO exports (rf_leash, rf_bowl) -> unified-category COCO boxes per split.

Raw file: one "<project>-v<n>-coco.zip" in raw/<source>/ with train/, valid/, test/, each
holding the images and _annotations.coco.json. Roboflow adds a dataset-name category at
id 0 (supercategory "none") with no boxes; it is dropped. Every other category maps,
ignoring case, onto UNIFIED_CATEGORIES, so both sources share category ids.
Roboflow file names look like "<original>_jpg.rf.<hash>.jpg"; the original key strips
both parts, so augmented copies of one photo share a key (and a leakage group).

rf_leash keeps Roboflow's split. rf_bowl is mostly video frames, and Roboflow put frames of
one recording in different splits, so rf_bowl is re-split by recording: the leakage group
is the recording (see RECORDING_PATTERNS), and groups fill test and val to about 10% of
the images each, in sha1(group) order (balanced_group_splits).
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path, PurePosixPath
from typing import Any

from woofdecoder.data.fileio import relative_data_path
from woofdecoder.data.manifest import ManifestRow, make_clip_id
from woofdecoder.data.prepare.common import (
    add_coco_annotation,
    add_coco_image,
    balanced_group_splits,
    dataset_config,
    new_coco,
    write_coco,
    write_labels,
)

# Unified categories, numbered 1.. in this order in every output (1 dog ... 5 cat).
UNIFIED_CATEGORIES: list[str] = ["dog", "person", "leash", "bowl", "cat"]
# Category -> its count column in labels.csv; has_<category> is derived from it.
COUNT_COLUMNS: dict[str, str] = {
    "dog": "n_dogs",
    "person": "n_people",
    "leash": "n_leashes",
    "bowl": "n_bowls",
    "cat": "n_cats",
}
OTHER_DOG_MIN_DOGS: int = 2  # other_dog: a second dog is in the picture
EMPTY_SUPERCATEGORY: str = "none"  # Roboflow's dataset-name category has this parent
SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_BY_FOLDER: dict[str, str] = {"train": "train", "valid": "val", "test": "test"}
ANNOTATIONS_FILENAME: str = "_annotations.coco.json"
ARCHIVE_PATTERN: str = "*-coco.zip"
IMAGES_DIRNAME: str = "images"
ROBOFLOW_HASH = re.compile(r"\.rf\.[0-9a-f]+\.[a-z]+$", re.IGNORECASE)
ORIGINAL_SUFFIX = re.compile(r"_(jpg|jpeg|png|webp)$", re.IGNORECASE)
# Recording name formats; the first match wins and keys matching none are their own group.
RECORDING_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "-2025-11-21_111443_758-frame_000145" -> "-2025-11-21_111443_758"
    re.compile(r"^(?P<recording>.+)-frame_\d+$"),
    # "20251204083406_20251204085312_0019" (start time, end time, frame) ->
    # "20251204083406_20251204085312"
    re.compile(r"^(?P<recording>\d{14}_\d{14})_\d+$"),
)
BALANCED_TEST_SHARE: float = 0.10
BALANCED_VAL_SHARE: float = 0.10
MIN_EVAL_RECORDINGS: int = 5  # report rf_bowl val / test with fewer recordings
MIN_EVAL_BOWL_BOXES: int = 50  # report rf_bowl val / test with fewer bowl boxes
SOURCE_LICENSE: str = "CC-BY-4.0"
LICENSE_NOTE: str = (
    "declared by the Roboflow uploader; image provenance unknown; do not redistribute"
)
LEASH_TRAIN_NOTE: str = "train is 3x augmented"
TAG_COLUMNS: list[str] = ["has_dog", "has_person", "has_leash", "has_bowl", "has_cat", "other_dog"]


@dataclass(frozen=True)
class RoboflowSource:
    """How one Roboflow export is converted."""

    name: str
    train_note: str  # added to the notes of train rows ("" for none)
    resplit_by_recording: bool  # False: keep Roboflow's train / valid / test


RF_LEASH = RoboflowSource("rf_leash", LEASH_TRAIN_NOTE, resplit_by_recording=False)
RF_BOWL = RoboflowSource("rf_bowl", "", resplit_by_recording=True)

Box = list[float]  # COCO x, y, w, h in pixels


# --- Names, groups and splits ---


def original_key(file_name: str) -> str:
    """Return the photo's original key: "photo_12_jpg.rf.abc123.jpg" -> "photo_12"."""
    stem = ROBOFLOW_HASH.sub("", file_name)
    return ORIGINAL_SUFFIX.sub("", stem)


def assign_keys(split_cocos: dict[str, dict[str, Any]]) -> None:
    """Set image["key"] on every image: its original key, plus "_<w>x<h>" when the files
    sharing that key do not all have the same size.

    Different photos can share a name. rf_bowl has two "frame_0012" (train 796x966 exported
    from a PNG, valid 1280x720 from a JPG) and two "10045" (train 640x853 without boxes, test
    1100x1100 with two). Augmented copies of one photo keep its size, so they keep one key.
    """
    sizes_by_key: dict[str, set[tuple[int, int]]] = {}
    for coco in split_cocos.values():
        for image in coco["images"]:
            size = (image["width"], image["height"])
            sizes_by_key.setdefault(original_key(image["file_name"]), set()).add(size)
    for coco in split_cocos.values():
        for image in coco["images"]:
            key = original_key(image["file_name"])
            if len(sizes_by_key[key]) > 1:
                key = f"{key}_{image['width']}x{image['height']}"
            image["key"] = key


def recording_group(key: str) -> str:
    """Return the recording a key belongs to (RECORDING_PATTERNS), else the key itself."""
    for pattern in RECORDING_PATTERNS:
        match = pattern.match(key)
        if match is not None:
            return match["recording"]
    return key


def label_columns(source: RoboflowSource) -> list[str]:
    """Return the labels.csv columns; re-split sources also keep "roboflow_split"."""
    columns = ["clip_id", "original_key", "split"]
    if source.resplit_by_recording:
        columns.append("roboflow_split")
    columns.extend(["width", "height"])
    columns.extend(COUNT_COLUMNS.values())
    columns.extend(TAG_COLUMNS)
    return columns


def map_categories(categories: list[dict[str, Any]], box_counts: Counter[int]) -> dict[int, str]:
    """Return {Roboflow category id: unified name}, dropping the empty supercategory.

    Raises:
        ValueError: If a category name is not in UNIFIED_CATEGORIES (case ignored), or the
            dataset-name supercategory has boxes.
    """
    unified_by_id: dict[int, str] = {}
    for category in categories:
        if category.get("supercategory") == EMPTY_SUPERCATEGORY:
            if box_counts[category["id"]] > 0:
                raise ValueError(f"supercategory {category['name']!r} has boxes")
            continue
        name = category["name"].strip().lower()
        if name not in UNIFIED_CATEGORIES:
            raise ValueError(f"unknown Roboflow category {category['name']!r}")
        unified_by_id[category["id"]] = name
    return unified_by_id


# --- Boxes and tags ---


def clean_box(bbox: Box, width: int, height: int) -> tuple[Box | None, str]:
    """Return (box clipped to the image, "kept" / "clipped") or (None, "dropped").

    A box with w or h <= 0, or lying completely outside the image, is dropped.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None, "dropped"
    if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
        return list(bbox), "kept"  # untouched, so float rounding cannot alter it
    left = max(0.0, x)
    top = max(0.0, y)
    right = min(float(width), x + w)
    bottom = min(float(height), y + h)
    if right <= left or bottom <= top:
        return None, "dropped"
    return [left, top, right - left, bottom - top], "clipped"


def image_tags(category_counts: Counter[str]) -> dict[str, Any]:
    """Return the labels.csv count and tag columns for one image's kept boxes."""
    tags: dict[str, Any] = {}
    for category, column in COUNT_COLUMNS.items():
        tags[column] = category_counts[category]
        tags[f"has_{category}"] = category_counts[category] > 0
    tags["other_dog"] = category_counts["dog"] >= OTHER_DOG_MIN_DOGS
    return tags


# --- Reading and planning ---


def read_split(archive: zipfile.ZipFile, folder: str) -> dict[str, Any]:
    """Return one folder's COCO JSON, with its category map and boxes grouped per image."""
    coco = json.loads(archive.read(f"{folder}/{ANNOTATIONS_FILENAME}"))
    box_counts: Counter[int] = Counter()
    boxes_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        box_counts[annotation["category_id"]] += 1
        boxes_by_image.setdefault(annotation["image_id"], []).append(annotation)
    coco["unified_by_id"] = map_categories(coco["categories"], box_counts)
    coco["boxes_by_image"] = boxes_by_image
    return coco


def check_keys_in_one_split(split_cocos: dict[str, dict[str, Any]]) -> None:
    """Raise ValueError if one key (same name and size: a real duplicate) has images in
    more than one Roboflow split. Run assign_keys first."""
    splits_by_key: dict[str, set[str]] = {}
    for folder, coco in split_cocos.items():
        for image in coco["images"]:
            splits_by_key.setdefault(image["key"], set()).add(folder)
    crossing: dict[str, list[str]] = {}
    for key, folders in sorted(splits_by_key.items()):
        if len(folders) > 1:
            crossing[key] = sorted(folders)
    if crossing:
        raise ValueError(f"{len(crossing)} duplicate images are in more than one split: {crossing}")


def plan_images(
    source: RoboflowSource, split_cocos: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return one {folder, image, key, group, split} per image, in folder then name order.

    Without re-splitting, the group is the key and the split is Roboflow's. With it, the
    group is the recording and balanced_group_splits decides each group's split; val and
    test each take MIN_EVAL_RECORDINGS recordings first. Run assign_keys first.
    """
    plans: list[dict[str, Any]] = []
    images_per_group: Counter[str] = Counter()
    for folder, coco_in in split_cocos.items():
        for image in sorted(coco_in["images"], key=itemgetter("file_name")):
            key = image["key"]
            group = key
            if source.resplit_by_recording:
                group = recording_group(key)
            images_per_group[group] += 1
            split = SPLIT_BY_FOLDER[folder]
            plan = {"folder": folder, "image": image, "key": key, "group": group, "split": split}
            plans.append(plan)

    if source.resplit_by_recording:
        recordings: set[str] = set()
        for plan in plans:
            if plan["group"] != plan["key"]:
                recordings.add(plan["group"])
        split_by_group = balanced_group_splits(
            images_per_group,
            BALANCED_TEST_SHARE,
            BALANCED_VAL_SHARE,
            priority_groups=recordings,
            min_priority_groups=MIN_EVAL_RECORDINGS,
        )
        for plan in plans:
            plan["split"] = split_by_group[plan["group"]]
    return plans


# --- Converting ---


def boxes_path(out_dir: Path, split: str) -> Path:
    """Return out_dir/boxes_<split>.json."""
    return out_dir / f"boxes_{split}.json"


def add_boxes(
    coco_out: dict[str, Any],
    image_id: int,
    image: dict[str, Any],
    coco_in: dict[str, Any],
    stats: Counter[str],
) -> Counter[str]:
    """Add one image's cleaned boxes to coco_out; count kept boxes per category and
    clipped / dropped boxes in stats.

    Returns:
        The number of kept boxes per unified category.
    """
    category_counts: Counter[str] = Counter()
    for annotation in coco_in["boxes_by_image"].get(image["id"], []):
        category = coco_in["unified_by_id"][annotation["category_id"]]
        box, status = clean_box(annotation["bbox"], image["width"], image["height"])
        stats[status] += 1
        if box is None:
            continue
        add_coco_annotation(coco_out, image_id, category, box)
        category_counts[category] += 1
        stats[category] += 1
    return category_counts


def make_row(
    source: RoboflowSource,
    plan: dict[str, Any],
    image_path: Path,
    out_dir: Path,
    source_version: str,
) -> ManifestRow:
    """Return the manifest row of one image (train rows carry the source's train note)."""
    split = plan["split"]
    notes = LICENSE_NOTE
    if split == "train" and source.train_note:
        notes = f"{LICENSE_NOTE}; {source.train_note}"
    clip_path = f"{IMAGES_DIRNAME}/{split}/{plan['image']['file_name']}"
    return ManifestRow(
        clip_id=make_clip_id(source.name, clip_path),
        source=source.name,
        source_version=source_version,
        source_license=SOURCE_LICENSE,
        leakage_group=f"{source.name}/{plan['group']}",
        split=split,
        modality="image",
        file_path=relative_data_path(image_path),
        annotation_path=relative_data_path(boxes_path(out_dir, split)),
        label_type="object",
        label_provenance="uploader",
        eval_allowed=split != "train",
        notes=notes,
    )


def convert_image(
    archive: zipfile.ZipFile,
    source: RoboflowSource,
    plan: dict[str, Any],
    coco_in: dict[str, Any],
    coco_out: dict[str, Any],
    stats: Counter[str],
    out_dir: Path,
    source_version: str,
) -> tuple[ManifestRow, dict[str, Any]]:
    """Extract one image into images/<split>/, add its boxes; return its row and labels."""
    image = plan["image"]
    image_path = out_dir / IMAGES_DIRNAME / plan["split"] / image["file_name"]
    if image_path.exists():
        raise ValueError(f"two images would both be written to {image_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(archive.read(f"{plan['folder']}/{image['file_name']}"))
    image_id = add_coco_image(coco_out, image_path, image["width"], image["height"])
    category_counts = add_boxes(coco_out, image_id, image, coco_in, stats)

    row = make_row(source, plan, image_path, out_dir, source_version)
    label_row: dict[str, Any] = {
        "clip_id": row.clip_id,
        "original_key": plan["key"],
        "split": plan["split"],
        "roboflow_split": SPLIT_BY_FOLDER[plan["folder"]],
        "width": image["width"],
        "height": image["height"],
    }
    return row, label_row | image_tags(category_counts)


# --- Report ---


def print_split_report(split: str, num_images: int, stats: Counter[str]) -> None:
    """Print one split's images, kept boxes per category, and clipped / dropped boxes."""
    box_counts: dict[str, int] = {}
    for category in UNIFIED_CATEGORIES:
        box_counts[category] = stats[category]
    print(f"  {split}: {num_images} images, boxes {box_counts}")
    print(f"  {split}: clipped {stats['clipped']}, dropped {stats['dropped']}")


def print_augmentation(plans: list[dict[str, Any]]) -> None:
    """Print how many train images exist per original photo (augmented copies)."""
    copies_per_key: Counter[str] = Counter()
    for plan in plans:
        if plan["split"] == "train":
            copies_per_key[plan["key"]] += 1
    originals_per_copy_count = dict(sorted(Counter(copies_per_key.values()).items()))
    print(f"  train: copies per original -> number of originals: {originals_per_copy_count}")


def print_resplit(plans: list[dict[str, Any]], stats: dict[str, Counter[str]]) -> None:
    """Print recordings per new split, images that changed split, and thin bowl splits."""
    recordings: dict[str, set[str]] = {}
    for split in SPLITS:
        recordings[split] = set()
    changed_count = 0
    for plan in plans:
        if plan["group"] != plan["key"]:
            recordings[plan["split"]].add(plan["group"])
        if plan["split"] != SPLIT_BY_FOLDER[plan["folder"]]:
            changed_count += 1
    recording_counts: dict[str, int] = {}
    for split in SPLITS:
        recording_counts[split] = len(recordings[split])
    print(f"  recordings per split: {recording_counts}")
    print(f"  images whose split differs from Roboflow's: {changed_count} of {len(plans)}")
    for split in ("val", "test"):
        if recording_counts[split] < MIN_EVAL_RECORDINGS:
            print(f"  WARNING: {split} has only {recording_counts[split]} recordings")
        if stats[split]["bowl"] < MIN_EVAL_BOWL_BOXES:
            print(f"  WARNING: {split} has only {stats[split]['bowl']} bowl boxes")


# --- Entry points ---


def prepare_roboflow(raw_dir: Path, out_dir: Path, source: RoboflowSource) -> list[ManifestRow]:
    """Convert one Roboflow COCO export: images, boxes_{train,val,test}.json, labels.csv, rows.

    Raises:
        ValueError: If raw_dir does not hold exactly one export zip, a category is unknown,
            or an original key is in more than one Roboflow split (checked before any
            extraction).
    """
    archives = sorted(raw_dir.glob(ARCHIVE_PATTERN))
    if len(archives) != 1:
        raise ValueError(f"expected one {ARCHIVE_PATTERN} in {raw_dir}, found {len(archives)}")
    source_version = f"roboflow_v{PurePosixPath(dataset_config(source.name)['link']).name}"
    coco_outs: dict[str, dict[str, Any]] = {}
    stats: dict[str, Counter[str]] = {}
    for split in SPLITS:
        coco_outs[split] = new_coco(UNIFIED_CATEGORIES)
        stats[split] = Counter()

    rows: list[ManifestRow] = []
    label_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archives[0]) as archive:
        split_cocos: dict[str, dict[str, Any]] = {}
        for folder in SPLIT_BY_FOLDER:
            split_cocos[folder] = read_split(archive, folder)
        assign_keys(split_cocos)
        check_keys_in_one_split(split_cocos)
        plans = plan_images(source, split_cocos)
        for plan in plans:
            split = plan["split"]
            coco_in = split_cocos[plan["folder"]]
            row, label_row = convert_image(
                archive,
                source,
                plan,
                coco_in,
                coco_outs[split],
                stats[split],
                out_dir,
                source_version,
            )
            rows.append(row)
            label_rows.append(label_row)

    for split in SPLITS:
        write_coco(coco_outs[split], boxes_path(out_dir, split))
        print_split_report(split, len(coco_outs[split]["images"]), stats[split])
    write_labels(out_dir, label_rows, label_columns(source))
    if source.resplit_by_recording:
        print_resplit(plans, stats)
    else:
        print_augmentation(plans)
    return rows


def prepare_rf_leash(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Convert rf_leash (dog, leash, person); Roboflow's split, train 3x augmented."""
    return prepare_roboflow(raw_dir, out_dir, RF_LEASH)


def prepare_rf_bowl(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Convert rf_bowl (bowl, cat, dog), re-split by recording; cat-only images stay in as
    negatives for "dog"."""
    return prepare_roboflow(raw_dir, out_dir, RF_BOWL)
