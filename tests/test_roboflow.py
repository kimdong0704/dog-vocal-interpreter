"""Tests for the Roboflow converters (rf_leash, rf_bowl). No real data."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from woofdata.prepare.sources import roboflow
from woofdata.prepare.utils.hash_splits import balanced_group_splits


def test_category_mapping() -> None:
    """Test: Names map ignoring case, the empty supercategory is dropped, unknown stops."""
    categories = [
        {"id": 0, "name": "objects", "supercategory": "none"},
        {"id": 1, "name": "DOG", "supercategory": "objects"},
        {"id": 2, "name": "Leash", "supercategory": "objects"},
    ]
    box_counts: Counter[int] = Counter({1: 5, 2: 1})

    assert roboflow.map_categories(categories, box_counts) == {1: "dog", 2: "leash"}
    unknown = categories + [{"id": 3, "name": "squirrel", "supercategory": "objects"}]
    with pytest.raises(ValueError, match="squirrel"):
        roboflow.map_categories(unknown, box_counts)
    with pytest.raises(ValueError, match="objects"):
        roboflow.map_categories(categories, Counter({0: 1}))


def test_original_key_strips_roboflow_parts() -> None:
    """Test: ".rf.<hash>" and the "_jpg" / "_png" suffix are removed, nothing else."""
    assert roboflow.original_key("photo_12_jpg.rf.abc123.jpg") == "photo_12"
    assert roboflow.original_key("scan_3_png.rf.0f9e8d.jpg") == "scan_3"
    assert roboflow.original_key("-2025-11-21_111443_758-frame_000006_jpg.rf.567a.jpg") == (
        "-2025-11-21_111443_758-frame_000006"
    )


def test_boxes_dropped_or_clipped() -> None:
    """Test: Zero width or fully outside -> dropped; past the edge -> clipped; inside -> kept."""
    assert roboflow.clean_box([10, 10, 0, 5], 100, 100) == (None, "dropped")
    assert roboflow.clean_box([120, 10, 5, 5], 100, 100) == (None, "dropped")
    assert roboflow.clean_box([90, -5, 20, 20], 100, 100) == ([90.0, 0.0, 10.0, 15.0], "clipped")
    assert roboflow.clean_box([10, 10, 20, 20], 100, 100) == ([10, 10, 20, 20], "kept")
    # (10.3 + 20.1) - 10.3 is not exactly 20.1: an inside box must come back unchanged.
    assert roboflow.clean_box([10.3, 0.0, 20.1, 5.0], 100, 100) == ([10.3, 0.0, 20.1, 5.0], "kept")


def test_other_dog_needs_two_dog_boxes() -> None:
    """Test: other_dog is true only with 2 or more dog boxes; has_* follow the counts."""
    one_dog = roboflow.image_tags(Counter({"dog": 1, "leash": 1}))
    two_dogs = roboflow.image_tags(Counter({"dog": 2}))
    cat_only = roboflow.image_tags(Counter({"cat": 1}))

    assert one_dog["other_dog"] is False
    assert one_dog["has_leash"] is True
    assert one_dog["n_dogs"] == 1
    assert two_dogs["other_dog"] is True
    assert cat_only["has_dog"] is False
    assert cat_only["has_cat"] is True


def roboflow_image(file_name: str, width: int, height: int) -> dict[str, object]:
    """Return one Roboflow COCO image entry."""
    return {"file_name": file_name, "width": width, "height": height}


def test_same_name_different_size_gets_size_suffixed_keys() -> None:
    """Test: frame_0012 and 10045 (different photos, different sizes) get distinct keys,
    so they no longer stop the converter; other keys stay as they are."""
    split_cocos: dict[str, dict[str, Any]] = {
        "train": {
            "images": [
                roboflow_image("frame_0012_png.rf.b922.jpg", 796, 966),
                roboflow_image("10045_webp.rf.ad5d.jpg", 640, 853),
            ]
        },
        "valid": {"images": [roboflow_image("frame_0012_jpg.rf.4e83.jpg", 1280, 720)]},
        "test": {
            "images": [
                roboflow_image("10045_jpg.rf.c716.jpg", 1100, 1100),
                roboflow_image("other_jpg.rf.1a2b.jpg", 640, 480),
            ]
        },
    }

    roboflow.assign_keys(split_cocos)
    roboflow.check_keys_in_one_split(split_cocos)

    keys: list[str] = []
    for coco in split_cocos.values():
        for image in coco["images"]:
            keys.append(image["key"])
    assert sorted(keys) == [
        "10045_1100x1100",
        "10045_640x853",
        "frame_0012_1280x720",
        "frame_0012_796x966",
        "other",
    ]


def test_same_name_same_size_in_two_splits_stops() -> None:
    """Test: Same key and same size in two splits is a real duplicate: stop and name both."""
    split_cocos = {
        "train": {"images": [roboflow_image("photo_7_jpg.rf.aaaa.jpg", 512, 512)]},
        "valid": {"images": [roboflow_image("photo_7_jpg.rf.bbbb.jpg", 512, 512)]},
        "test": {"images": []},
    }

    roboflow.assign_keys(split_cocos)
    with pytest.raises(ValueError, match=r"photo_7': \['train', 'valid'\]"):
        roboflow.check_keys_in_one_split(split_cocos)


def test_augmented_copies_keep_one_key() -> None:
    """Test: Augmented copies (same name and size, same split) share one unsuffixed key."""
    split_cocos = {
        "train": {
            "images": [
                roboflow_image("001020_jpg.rf.0406.jpg", 512, 512),
                roboflow_image("001020_jpg.rf.c3fc.jpg", 512, 512),
                roboflow_image("001020_jpg.rf.9a1b.jpg", 512, 512),
            ]
        },
        "valid": {"images": []},
        "test": {"images": []},
    }

    roboflow.assign_keys(split_cocos)
    roboflow.check_keys_in_one_split(split_cocos)

    for image in split_cocos["train"]["images"]:
        assert image["key"] == "001020"


def test_recording_group_both_formats() -> None:
    """Test: "-frame_N" and "<start>_<end>_<frame>" keys group by recording; others alone."""
    first = roboflow.recording_group("-2025-11-21_111443_758-frame_000006")
    second = roboflow.recording_group("-2025-11-21_111443_758-frame_000145")

    assert first == second == "-2025-11-21_111443_758"
    assert roboflow.recording_group("20251204083406_20251204085312_0019") == (
        "20251204083406_20251204085312"
    )
    assert roboflow.recording_group("202512040834_photo") == "202512040834_photo"
    assert roboflow.recording_group("001020") == "001020"


def test_webp_suffix_is_stripped() -> None:
    """Test: "_webp" is removed from the key like "_jpg", "_jpeg" and "_png"."""
    assert roboflow.original_key("123456789012_webp.rf.abc123.jpg") == "123456789012"


def images_per_split(splits: dict[str, str], images_per_group: dict[str, int]) -> dict[str, int]:
    """Return the number of images in each split."""
    counts = {"train": 0, "val": 0, "test": 0}
    for group, split in splits.items():
        counts[split] += images_per_group[group]
    return counts


def test_balanced_group_splits() -> None:
    """Test: Same result on every run and dict order; test and val each reach ~10% of the
    images; every group is assigned once, whole."""
    images_per_group: dict[str, int] = {}
    for number in range(60):
        images_per_group[f"group_{number}"] = 1 + number % 7

    splits = balanced_group_splits(images_per_group, 0.10, 0.10)
    reversed_groups = dict(reversed(list(images_per_group.items())))

    assert splits == balanced_group_splits(reversed_groups, 0.10, 0.10)
    assert set(splits) == set(images_per_group)
    counts = images_per_split(splits, images_per_group)
    total = sum(images_per_group.values())
    largest_group = max(images_per_group.values())
    for split in ("test", "val"):
        assert 0.10 * total <= counts[split] < 0.10 * total + largest_group


def test_balanced_group_splits_takes_recordings_first() -> None:
    """Test: With many single images and few large recordings, val and test still get 5
    recordings each (option B), and the result is the same on every run."""
    images_per_group: dict[str, int] = {}
    recordings: set[str] = set()
    for number in range(12):
        images_per_group[f"recording_{number}"] = 30
        recordings.add(f"recording_{number}")
    for number in range(2000):
        images_per_group[f"single_{number}"] = 1

    splits = balanced_group_splits(
        images_per_group, 0.10, 0.10, priority_groups=recordings, min_priority_groups=5
    )

    assert splits == balanced_group_splits(
        images_per_group, 0.10, 0.10, priority_groups=recordings, min_priority_groups=5
    )
    for split in ("test", "val"):
        recordings_in_split = 0
        for recording in recordings:
            if splits[recording] == split:
                recordings_in_split += 1
        assert recordings_in_split >= 5
    counts = images_per_split(splits, images_per_group)
    total = sum(images_per_group.values())
    assert counts["test"] >= 0.10 * total
    assert counts["val"] >= 0.10 * total


def test_roboflow_split_column_only_when_resplit() -> None:
    """Test: labels.csv gets roboflow_split for rf_bowl only; rf_leash columns are unchanged."""
    assert "roboflow_split" in roboflow.label_columns(roboflow.RF_BOWL)
    assert "roboflow_split" not in roboflow.label_columns(roboflow.RF_LEASH)
