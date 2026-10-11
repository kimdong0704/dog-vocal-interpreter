"""Tests for the per-source converters and the SED mapping tables. Synthetic data only."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile

from woofdecoder.data.prepare import audioset_strong_eval, esc50, playpant
from woofdecoder.sed_classes import AUDIOSET_TO_SED, PLAYPANT_TO_SED, SED_CLASSES

TONE_RATE: int = 44_100
LAYER1_HEADER: str = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tAnnotation\n"
LAYER2_HEADER: str = "tmin\ttier\ttext\ttmax\r\n"


def tone_bytes(seconds: float, file_format: str) -> bytes:
    """Return a mono sine tone of the given length as WAV or FLAC file bytes."""
    times = np.arange(int(seconds * TONE_RATE)) / TONE_RATE  # shape (num_samples,)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * times)  # shape (num_samples,)
    buffer = io.BytesIO()
    soundfile.write(buffer, tone, TONE_RATE, format=file_format)
    return buffer.getvalue()


def processed_source_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str) -> Path:
    """Point the data folder at tmp_path/data and return its empty processed/<source>/."""
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(tmp_path / "data"))
    out_dir = tmp_path / "data" / "processed" / source
    out_dir.mkdir(parents=True)
    return out_dir


# --- Mapping tables ---


def test_audioset_mapping() -> None:
    """Test: AudioSet names map as agreed, and human "Pant" maps to nothing."""
    assert AUDIOSET_TO_SED == {
        "Bark": "bark",
        "Yip": "bark",
        "Bow-wow": "bark",
        "Growling": "growl",
        "Whimper (dog)": "whine",
        "Howl": "howl",
        "Bay": "howl",
        "Dog": "other_vocal",
        "Canidae, dogs, wolves": "other_vocal",
    }
    assert "Pant" not in AUDIOSET_TO_SED


def test_playpant_mapping() -> None:
    """Test: PlayPanT codes map as agreed; J, K and L (no dog vocalization) map to nothing."""
    assert PLAYPANT_TO_SED == {
        "C": "bark",
        "A": "growl",
        "B": "whine",
        "G": "howl",
        "E": "pant",
        "D": "other_vocal",
        "F": "other_vocal",
        "H": "other_vocal",
        "I": "other_vocal",
    }
    for code in ("J", "K", "L"):
        assert code not in PLAYPANT_TO_SED


def test_mappings_only_use_sed_classes() -> None:
    """Test: Every mapped class is in SED_CLASSES, and Layer 2 names cover codes A-L."""
    mapped_classes = list(AUDIOSET_TO_SED.values()) + list(PLAYPANT_TO_SED.values())
    for sed_class in mapped_classes:
        assert sed_class in SED_CLASSES
    assert sorted(playpant.LAYER2_TEXT_TO_CODE.values()) == list("ABCDEFGHIJKL")


# --- PlayPanT ---


def make_annotation_zip(path: Path) -> None:
    """Write an annotations zip: session P-01 has both layers, P-02 only Layer 1."""
    with zipfile.ZipFile(path, "w") as zip_archive:
        layer1 = (
            LAYER1_HEADER + "1\tSpectrogram 1\t1\t1.0\t2.0\tC \n2\tSpectrogram 1\t1\t3.0\t4.0\t\n"
        )
        zip_archive.writestr("04 Annotations/FirstLayer/P-01-S-P_1.Table.1.selections.txt", layer1)
        zip_archive.writestr("04 Annotations/FirstLayer/P-02-S-P_1.Table.1.selections.txt", layer1)
        layer2 = LAYER2_HEADER + "0.5\tcalls\tPant\t1.5\r\n2.0\tnoise\tHuman\t2.5\r\n"
        zip_archive.writestr("04 Annotations/SecondLayer/P-01-S-P_1.Table", layer2)
        zip_archive.writestr("__MACOSX/04 Annotations/SecondLayer/._P-02-S-P_1.Table", "junk")


def test_layer2_preferred_over_layer1(tmp_path: Path) -> None:
    """Test: Layer 2 is used when it exists, else Layer 1; macOS copies are ignored."""
    archive_path = tmp_path / "annotations.zip"
    make_annotation_zip(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(archive.namelist())
        layer1_members = playpant.session_members(names, playpant.LAYER1_SUFFIX)
        layer2_members = playpant.session_members(names, playpant.LAYER2_SUFFIX)

        both = playpant.read_session_events(archive, "P-01-S-P_1", layer1_members, layer2_members)
        only1 = playpant.read_session_events(archive, "P-02-S-P_1", layer1_members, layer2_members)

    assert sorted(layer2_members) == ["P-01-S-P_1"]
    assert both == ([(0.5, 1.5, "E"), (2.0, 2.5, "L")], "layer 2")
    assert only1 == ([(1.0, 2.0, "C")], "layer 1")  # stray space stripped, blank code dropped


def test_playpant_dog_split_is_fixed() -> None:
    """Test: The 17 dogs always split the same way, 9 to val and 8 to test."""
    dog_ids: list[str] = []
    for number in range(1, 18):
        dog_ids.append(f"{number:02d}")

    first = playpant.assign_dog_splits(dog_ids)
    again = playpant.assign_dog_splits(list(reversed(dog_ids)) + dog_ids)

    assert first == again
    assert list(first.values()).count("val") == 9
    assert list(first.values()).count("test") == 8


def test_playpant_windows_rows_events_and_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test: A 12 s session gives two windows; events land in the windows they overlap."""
    out_dir = processed_source_dir(monkeypatch, tmp_path, "playpant")
    events = [(9.0, 11.0, "C"), (1.0, 2.0, "L")]

    rows, event_rows = playpant.session_windows(
        "P-01-S-P_1", "01", "val", "layer 2", 12.0, events, out_dir
    )

    assert (rows[0].start_s, rows[1].start_s) == (0.0, 10.0)
    assert rows[0].dog_id == "playpant_01"
    assert rows[0].mapped_category == "playful_excited"
    first_targets = np.load(tmp_path / "data" / rows[0].target_path)
    second_targets = np.load(tmp_path / "data" / rows[1].target_path)
    assert first_targets.shape == (250, len(SED_CLASSES))
    assert first_targets[:, SED_CLASSES.index("bark")].sum() == 25  # 9-10 s
    assert second_targets[:, SED_CLASSES.index("bark")].sum() == 25  # 10-11 s
    assert first_targets.sum() == 25  # the human event (L) sets no frames
    assert len(event_rows) == 3  # the bark twice (two windows), the human once
    human_rows = pd.DataFrame(event_rows).query("original_label == 'L'")
    assert list(human_rows["sed_class"]) == [""]


# --- AudioSet ---


def test_audioset_clip_row_and_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: dog_id gets one "<source>_" prefix; only mapped events set frame targets."""
    out_dir = processed_source_dir(monkeypatch, tmp_path, "audioset_strong_eval")
    clip = {
        "video_id": "-21_SXelVNo",
        "audio": {"bytes": tone_bytes(4.0, "FLAC"), "path": "-21_SXelVNo.flac"},
        "events": [
            {"start": 0.0, "end": 1.0, "event_name": "Bark"},
            {"start": 2.0, "end": 3.0, "event_name": "Pant"},
        ],
    }

    row, event_rows = audioset_strong_eval.convert_clip(clip, out_dir, "abc123")

    assert row.dog_id == "audioset_strong_eval_-21_SXelVNo"
    assert row.leakage_group == "audioset"
    assert row.source_version == "abc123"
    assert row.source_license == "CC-BY-4.0"
    assert (row.start_s, row.end_s) == (0.0, 10.0)
    assert row.original_label == "Bark"
    targets = np.load(tmp_path / "data" / row.target_path)
    assert targets.sum() == 25  # the bark only; human "Pant" sets nothing
    assert (event_rows[0]["sed_class"], event_rows[1]["sed_class"]) == ("bark", "")


# --- ESC-50 ---


def test_esc50_fold_to_split_and_license() -> None:
    """Test: Folds 1-3 -> train, 4 -> val, 5 -> test; esc10 clips are CC BY, others CC BY-NC."""
    expected = {"1": "train", "2": "train", "3": "train", "4": "val", "5": "test"}
    for fold, split in expected.items():
        assert esc50.clip_split(fold) == split
    with pytest.raises(ValueError, match="fold"):
        esc50.clip_split("6")
    assert esc50.clip_license("True") == "CC-BY-3.0"
    assert esc50.clip_license("False") == "CC-BY-NC-3.0"


def test_esc50_source_follows_its_lowest_fold() -> None:
    """Test: Clips of one source recording in folds 4 and 5 all take fold 4 (val)."""
    clips = [
        {"src_file": "209698", "fold": "5"},
        {"src_file": "209698", "fold": "4"},
        {"src_file": "131943", "fold": "3"},
        {"src_file": "131943", "fold": "2"},
        {"src_file": "100032", "fold": "1"},
    ]

    lowest_fold = esc50.lowest_fold_by_source(clips)

    assert lowest_fold == {"209698": "4", "131943": "2", "100032": "1"}
    assert esc50.clip_split(lowest_fold["209698"]) == "val"


def test_source_versions_come_from_datasets_yaml() -> None:
    """Test: ESC-50 uses the commit in its archive url; AudioSet its pinned revision."""
    assert esc50.archive_commit("esc50") == "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
    revision = audioset_strong_eval.dataset_config("audioset_strong_eval")["revision"]
    assert revision == "804fa9ccfa8251edb08c5721471bdde9638adc62"


@pytest.mark.parametrize(
    ("category", "label_type", "has_target"),
    [("dog", "call_type", False), ("rain", "object", True)],
)
def test_esc50_clip_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    category: str,
    label_type: str,
    has_target: bool,
) -> None:
    """Test: Dog clips have no target file; other clips get all-zero targets; one prefix."""
    out_dir = processed_source_dir(monkeypatch, tmp_path, "esc50")
    audio_path = tmp_path / "1-100032-A-0.wav"
    audio_path.write_bytes(tone_bytes(5.0, "WAV"))
    clip = {
        "filename": "1-100032-A-0.wav",
        "fold": "4",
        "category": category,
        "esc10": "True",
        "src_file": "100032",
    }

    row = esc50.convert_clip(clip, audio_path, out_dir, "val", "33c8ce9")

    assert row.dog_id == "esc50_100032"
    assert row.split == "val"
    assert row.source_version == "33c8ce9"
    assert row.source_license == "CC-BY-3.0"
    assert row.label_type == label_type
    assert (row.start_s, row.end_s) == (0.0, 10.0)
    assert bool(row.target_path) == has_target
    if has_target:
        assert np.load(tmp_path / "data" / row.target_path).sum() == 0
    else:
        assert row.notes == "clip-level dog label, no frame targets"
