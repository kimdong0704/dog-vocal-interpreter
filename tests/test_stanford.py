"""Tests for the Stanford Dogs-based converters and the shared image split. No real data."""

from __future__ import annotations

import pytest

from woofdecoder.data.download import load_datasets
from woofdecoder.data.prepare import dogposecv, stanford_dogs, stanfordextra
from woofdecoder.data.prepare.stanford_shared import hash_split, shared_split


def test_official_split_wins_over_the_hash() -> None:
    """Test: A StanfordExtra image keeps its official split, whatever its hash says."""
    key = "n02085620_10074"
    for official in ("train", "val", "test"):
        assert shared_split(key, {key: official}) == official


def test_hash_split_is_fixed() -> None:
    """Test: Other images get the same split from the hash on every run, about 70/15/15."""
    keys: list[str] = []
    for number in range(2000):
        keys.append(f"n02085620_{number}")

    first: list[str] = []
    again: list[str] = []
    for key in keys:
        first.append(shared_split(key, {}))
        again.append(hash_split(key))

    assert first == again
    assert hash_split("n02085620_10074") == hash_split("n02085620_10074")
    assert 0.65 < first.count("train") / len(keys) < 0.75
    assert 0.12 < first.count("val") / len(keys) < 0.18


def test_empty_keypoints_always_have_visibility_zero() -> None:
    """Test: Eyes, withers and throat (20-23) are 0 even if marked; others follow the JSON."""
    joints: list[list[float]] = []
    for position in range(24):
        joints.append([10.0 + position, 20.0, 1])
    joints[3] = [0.0, 0.0, 0]

    keypoints = stanfordextra.coco_keypoints(joints)

    visibility = keypoints[2::3]
    assert len(keypoints) == 72
    assert visibility[20:24] == [0, 0, 0, 0]
    assert keypoints[60:72] == [0.0] * 12
    assert visibility[3] == 0
    assert visibility[16] == 2
    assert keypoints[48:50] == [26.0, 20.0]  # nose x, y kept as given


def test_voc_box_becomes_coco_xywh() -> None:
    """Test: A VOC (xmin, ymin, xmax, ymax) box becomes COCO (x, y, w, h); size is read."""
    xml = (
        b"<annotation><size><width>333</width><height>500</height></size>"
        b"<object><bndbox><xmin>25</xmin><ymin>10</ymin><xmax>276</xmax>"
        b"<ymax>498</ymax></bndbox></object><object><bndbox><xmin>1</xmin><ymin>2</ymin>"
        b"<xmax>11</xmax><ymax>22</ymax></bndbox></object></annotation>"
    )

    assert stanford_dogs.parse_voc(xml) == (333, 500, [(25, 10, 251, 488), (1, 2, 10, 20)])


def test_dogposecv_image_not_in_index_stops() -> None:
    """Test: A labelled DogPoseCV image whose key is not in the index raises."""
    index = {"n02085620_10074": {"split": "train"}}

    dogposecv.check_keys_in_index({"n02085620_10074": "sitting"}, index)
    with pytest.raises(ValueError, match="n02099999_1"):
        dogposecv.check_keys_in_index({"n02099999_1": "lying"}, index)


def test_stanford_dogs_is_prepared_first() -> None:
    """Test: In datasets.yaml order, stanford_dogs comes before the sources that read it."""
    dataset_ids: list[str] = []
    for dataset in load_datasets():
        dataset_ids.append(dataset["id"])

    for dependent in ("stanfordextra", "dog_pose", "dogposecv"):
        assert dataset_ids.index("stanford_dogs") < dataset_ids.index(dependent)
