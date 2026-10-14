"""Tests for the shared data-preparation core. Synthetic files only: no network, no real data."""

from __future__ import annotations

import io
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pytest
import soundfile

from woofdata.download.datasets import load_datasets
from woofdata.prepare import run, splits
from woofdata.prepare.fingerprints import (
    check_fingerprints,
    fingerprint_source,
    load_fingerprints,
    record_fingerprint,
)
from woofdata.prepare.manifest import ManifestRow, make_clip_id, rows_to_frame, write_source_rows
from woofdata.prepare.sources import registry
from woofdata.prepare.splits import check_splits
from woofdata.prepare.utils.archives import extract_members
from woofdata.prepare.utils.audio import load_audio, to_mono_16k, write_wav_16k
from woofdata.prepare.utils.windows import events_to_frame_targets, make_windows
from woofdata.utils_fileio import relative_data_path

SOURCE_RATE: int = 44_100
CLIP_SECONDS: float = 0.5


def write_test_tone(path: Path, num_channels: int, sample_rate: int, subtype: str) -> None:
    """Write a CLIP_SECONDS sine tone with num_channels channels."""
    times = np.arange(int(CLIP_SECONDS * sample_rate)) / sample_rate  # shape (num_samples,)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * times)  # shape (num_samples,)
    channels = np.tile(tone[:, None], (1, num_channels))  # shape (num_samples, num_channels)
    soundfile.write(path, channels, sample_rate, subtype=subtype)


def make_row(clip_name: str, split: str, dog_id: str = "", leakage_group: str = "") -> ManifestRow:
    """Return a minimal audio row of the esc50 source."""
    return ManifestRow(
        clip_id=make_clip_id("esc50", clip_name),
        source="esc50",
        dog_id=dog_id,
        leakage_group=leakage_group,
        split=split,
        modality="audio",
        file_path=f"processed/esc50/{clip_name}",
        label_type="context",
        label_provenance="expert",
    )


def use_temp_folders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the data folder and data/splits/ at tmp_path; return the data folder."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(data_dir))
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path / "splits")
    return data_dir


# --- Audio ---


@pytest.mark.parametrize(("num_channels", "subtype"), [(2, "PCM_16"), (1, "PCM_32"), (2, "FLOAT")])
def test_audio_comes_out_16k_mono_16bit(tmp_path: Path, num_channels: int, subtype: str) -> None:
    """Test: Stereo and 32-bit inputs come out as 16 kHz mono 16-bit PCM WAV."""
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "out" / "output.wav"
    write_test_tone(input_path, num_channels, SOURCE_RATE, subtype)

    samples, sample_rate = load_audio(input_path)
    write_wav_16k(output_path, to_mono_16k(samples, sample_rate))

    info = soundfile.info(output_path)
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert info.frames == int(CLIP_SECONDS * 16_000)


def test_load_audio_reads_bytes(tmp_path: Path) -> None:
    """Test: load_audio accepts raw file bytes (as stored in parquet files)."""
    input_path = tmp_path / "input.wav"
    write_test_tone(input_path, 2, SOURCE_RATE, "PCM_16")

    samples, sample_rate = load_audio(input_path.read_bytes())

    assert sample_rate == SOURCE_RATE
    assert samples.shape == (int(CLIP_SECONDS * SOURCE_RATE), 2)
    assert samples.dtype == np.float32


def test_make_windows_pads_last_window() -> None:
    """Test: The last window runs past the end instead of being dropped."""
    assert make_windows(25.0) == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    assert make_windows(20.0) == [(0.0, 10.0), (10.0, 20.0)]
    assert make_windows(3.0) == [(0.0, 10.0)]
    assert make_windows(25.0, window_s=10.0, hop_s=5.0)[-1] == (15.0, 25.0)
    assert make_windows(0.0) == []


def test_frame_targets_shape_and_boundaries() -> None:
    """Test: (250, C) targets; frames touching an event's edge only are not set."""
    class_names = ["bark", "whine"]
    window_start_s = 10.0
    events = [
        (10.4, 10.8, "bark"),  # exactly frames 10..19
        (9.5, 10.02, "whine"),  # starts before the window: frame 0 only
        (30.0, 31.0, "bark"),  # outside the window: ignored
    ]

    targets = events_to_frame_targets(events, class_names, window_start_s)

    assert targets.shape == (250, 2)
    assert targets.dtype == np.float32
    assert targets[9, 0] == 0
    assert targets[10, 0] == 1
    assert targets[19, 0] == 1
    assert targets[20, 0] == 0
    assert targets[:, 0].sum() == 10
    assert targets[0, 1] == 1
    assert targets[:, 1].sum() == 1


def test_frame_targets_reject_unknown_class() -> None:
    """Test: An event class missing from class_names raises."""
    with pytest.raises(ValueError, match="growl"):
        events_to_frame_targets([(0.0, 1.0, "growl")], ["bark"], 0.0)


# --- Archives ---


def make_zip(path: Path, member_names: list[str]) -> None:
    """Create a zip whose members contain their own names."""
    with zipfile.ZipFile(path, "w") as zip_archive:
        for member_name in member_names:
            zip_archive.writestr(member_name, member_name)


def make_tar(path: Path, member_names: list[str]) -> None:
    """Create a tar whose members contain their own names."""
    with tarfile.open(path, "w") as tar_archive:
        for member_name in member_names:
            content = member_name.encode()
            info = tarfile.TarInfo(member_name)
            info.size = len(content)
            tar_archive.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("make_archive", [make_zip, make_tar])
def test_extract_members_only_requested(
    tmp_path: Path, make_archive: Callable[[Path, list[str]], None]
) -> None:
    """Test: Only the named members (or pattern matches) are extracted, in sorted order."""
    archive_path = tmp_path / "archive"
    out_dir = tmp_path / "out"
    make_archive(archive_path, ["Images/b.jpg", "Images/a.jpg", "labels.json", "notes.txt"])

    by_pattern = extract_members(archive_path, "Images/*.jpg", out_dir)
    by_name = extract_members(archive_path, ["labels.json"], out_dir)

    assert by_pattern == [out_dir / "Images" / "a.jpg", out_dir / "Images" / "b.jpg"]
    assert by_name == [out_dir / "labels.json"]
    assert not (out_dir / "notes.txt").exists()
    assert (out_dir / "labels.json").read_text() == "labels.json"


def test_extract_members_skips_files_already_extracted(tmp_path: Path) -> None:
    """Test: A file already present with the right size is not extracted again."""
    archive_path = tmp_path / "archive.zip"
    out_dir = tmp_path / "out"
    make_zip(archive_path, ["labels.json"])
    extract_members(archive_path, ["labels.json"], out_dir)
    (out_dir / "labels.json").write_text("same length")  # same size as "labels.json"

    extract_members(archive_path, ["labels.json"], out_dir)

    assert (out_dir / "labels.json").read_text() == "same length"


def test_extract_members_rejects_missing_member(tmp_path: Path) -> None:
    """Test: Asking for a member that is not in the archive raises."""
    archive_path = tmp_path / "archive.zip"
    make_zip(archive_path, ["labels.json"])

    with pytest.raises(ValueError, match="missing.json"):
        extract_members(archive_path, ["missing.json"], tmp_path / "out")


# --- Manifest ---


def test_clip_id_and_paths_ignore_windows_style() -> None:
    """Test: Windows-style input paths give the same clip_id and manifest path as POSIX ones."""
    windows_clip_id = make_clip_id("esc50", PureWindowsPath("audio\\1-100.wav"), 10.0)
    posix_clip_id = make_clip_id("esc50", "audio/1-100.wav", 10.0)
    windows_path = relative_data_path("D:\\woof\\processed\\esc50\\a.wav", "D:\\woof")
    posix_path = relative_data_path("D:/woof/processed/esc50/a.wav", "D:/woof")

    assert windows_clip_id == posix_clip_id == "esc50/audio/1-100.wav@10.000"
    assert windows_path == posix_path == "processed/esc50/a.wav"


def test_manifest_row_derives_category_and_groups() -> None:
    """Test: mapped_category comes from mapped_subtype; dog_id and leakage_group are filled."""
    row = ManifestRow(
        clip_id="act_env/a.wav",
        source="act_env",
        dog_id="17",
        split="train",
        modality="audio",
        file_path="processed/act_env/a.wav",
        label_type="context",
        label_provenance="uploader",
        mapped_subtype="play",
    )
    no_dog_row = make_row("b.wav", "train")

    assert row.mapped_category == "playful_excited"
    assert row.dog_id == "act_env_17"
    assert row.leakage_group == "act_env_17"
    assert no_dog_row.dog_id == "esc50_esc50/b.wav"


def test_manifest_row_rejects_unknown_subtype() -> None:
    """Test: A mapped_subtype outside taxonomy.SUBTYPES raises."""
    with pytest.raises(ValueError, match="mapped_subtype"):
        ManifestRow(
            clip_id="x",
            source="esc50",
            split="train",
            modality="audio",
            file_path="x.wav",
            label_type="context",
            label_provenance="expert",
            mapped_subtype="not_a_subtype",
        )


# --- Splits ---


def test_check_splits_catches_dog_in_two_splits() -> None:
    """Test: The same dog in train and train_only is reported."""
    manifest = rows_to_frame(
        [make_row("a.wav", "train", "7"), make_row("b.wav", "train_only", "7")]
    )

    with pytest.raises(ValueError, match="esc50_7: train, train_only"):
        check_splits(manifest)


def test_check_splits_catches_leakage_group_across_splits() -> None:
    """Test: One leakage group in train and test is reported; train_only does not count."""
    leaking = rows_to_frame(
        [make_row("a.wav", "train", "1", "session_1"), make_row("b.wav", "test", "2", "session_1")]
    )
    train_only = rows_to_frame(
        [make_row("a.wav", "train", "1", "s2"), make_row("b.wav", "train_only", "2", "s2")]
    )

    with pytest.raises(ValueError, match="session_1: test, train"):
        check_splits(leaking)
    check_splits(train_only)


def test_check_splits_allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test: A leakage group in ALLOWED_CROSS_SPLIT_GROUPS may cross train/val/test."""
    manifest = rows_to_frame(
        [make_row("a.wav", "train", "1", "shared"), make_row("b.wav", "val", "2", "shared")]
    )
    monkeypatch.setattr(splits, "ALLOWED_CROSS_SPLIT_GROUPS", ("shared",))

    check_splits(manifest)


# --- Fingerprints ---


def test_fingerprint_changes_only_when_a_label_changes(tmp_path: Path) -> None:
    """Test: Same rows give the same fingerprint; one changed label changes it."""
    rows = [make_row("a.wav", "train"), make_row("b.wav", "test")]
    write_source_rows("esc50", rows, tmp_path)
    first = fingerprint_source("esc50", tmp_path)

    write_source_rows("esc50", list(reversed(rows)), tmp_path)
    same = fingerprint_source("esc50", tmp_path)

    rows[0].mapped_subtype = "play"
    write_source_rows("esc50", rows, tmp_path)
    changed = fingerprint_source("esc50", tmp_path)

    assert same == first
    assert changed["rows"] == first["rows"]
    assert changed["sha1"] != first["sha1"]


@pytest.mark.parametrize("filename", ["events.csv", "labels.csv"])
def test_fingerprint_changes_when_one_text_row_changes(tmp_path: Path, filename: str) -> None:
    """Test: One changed events.csv or labels.csv row changes the fingerprint; order does not."""
    write_source_rows("esc50", [make_row("a.wav", "train")], tmp_path)
    text_path = tmp_path / "processed" / "esc50" / filename
    text_path.write_text("clip_id,label\nesc50/a.wav,x\nesc50/b.wav,y\n", encoding="utf-8")
    first = fingerprint_source("esc50", tmp_path)

    text_path.write_text("clip_id,label\nesc50/b.wav,y\nesc50/a.wav,x\n", encoding="utf-8")
    reordered = fingerprint_source("esc50", tmp_path)
    text_path.write_text("clip_id,label\nesc50/a.wav,x\nesc50/b.wav,z\n", encoding="utf-8")
    changed = fingerprint_source("esc50", tmp_path)

    key = filename.replace(".csv", "_sha1")
    assert reordered == first
    assert changed[key] != first[key]
    assert changed["sha1"] == first["sha1"]


# --- Prepare runner ---


def fake_converter(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Converter stand-in: writes one file into out_dir and returns one row."""
    (out_dir / "a.wav").write_bytes(b"audio")
    return [make_row("a.wav", "train", "1")]


def two_row_converter(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Converter stand-in whose output differs from fake_converter's (two rows)."""
    (out_dir / "a.wav").write_bytes(b"audio")
    return [make_row("a.wav", "train", "1"), make_row("b.wav", "train", "1")]


def test_prepare_skips_missing_raw_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: A source with no raw folder is skipped with a message, not an error."""
    use_temp_folders(monkeypatch, tmp_path)
    monkeypatch.setitem(registry.CONVERTERS, "esc50", fake_converter)

    result = run.prepare_one("esc50")[0]

    assert result.status == "skipped"
    assert "raw folder missing" in result.message


def test_prepare_rebuilds_only_its_own_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test: Rerunning a source replaces processed/<source>/ only; manifest, splits and
    fingerprints are written, and --check then prints "match"."""
    data_dir = use_temp_folders(monkeypatch, tmp_path)
    monkeypatch.setitem(registry.CONVERTERS, "esc50", fake_converter)
    (data_dir / "raw" / "esc50").mkdir(parents=True)
    (data_dir / "processed" / "esc50").mkdir(parents=True)
    (data_dir / "processed" / "esc50" / "stale.wav").write_bytes(b"old")
    (data_dir / "processed" / "act_env").mkdir(parents=True)
    (data_dir / "processed" / "act_env" / "keep.wav").write_bytes(b"keep")

    result = run.prepare_one("esc50")[0]

    assert result.status == "done"
    assert result.rows_written == 1
    assert not (data_dir / "processed" / "esc50" / "stale.wav").exists()
    assert (data_dir / "processed" / "act_env" / "keep.wav").exists()
    manifest = pd.read_csv(data_dir / "processed" / "manifest.csv")
    assert list(manifest["clip_id"]) == ["esc50/a.wav"]
    assert (tmp_path / "splits" / "split_v1.json").is_file()
    assert (tmp_path / "splits" / "dog_groups.csv").is_file()
    assert check_fingerprints() == []


def test_check_reports_the_differing_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test: --check prints "match", then names the source whose rows changed."""
    data_dir = use_temp_folders(monkeypatch, tmp_path)
    monkeypatch.setitem(registry.CONVERTERS, "esc50", fake_converter)
    (data_dir / "raw" / "esc50").mkdir(parents=True)
    run.prepare_one("esc50")

    assert run.main(["--check"]) == 0
    assert capsys.readouterr().out == "match\n"

    write_source_rows("esc50", [make_row("a.wav", "test", "1")], data_dir)
    assert run.main(["--check"]) == 1
    assert capsys.readouterr().out.startswith("esc50: rows 1 -> 1, files 2 -> 2, sha1 differs")


def test_record_fingerprint_keeps_a_differing_entry(tmp_path: Path) -> None:
    """Test: New -> recorded, same -> match, different -> WARNING and kept, update -> overwritten."""
    first = {"rows": 1, "files": 2, "sha1": "aaa"}
    changed = {"rows": 3, "files": 4, "sha1": "bbb"}

    assert record_fingerprint("esc50", first, splits_dir=tmp_path) == "fingerprint recorded"
    assert record_fingerprint("esc50", first, splits_dir=tmp_path) == "fingerprint match"
    warning = record_fingerprint("esc50", changed, splits_dir=tmp_path)
    assert warning.startswith("WARNING")
    assert "rows 1 -> 3, files 2 -> 4, sha1 differs" in warning
    assert load_fingerprints(tmp_path)["esc50"] == first

    updated = record_fingerprint("esc50", changed, update=True, splits_dir=tmp_path)
    assert updated.startswith("fingerprint updated")
    assert load_fingerprints(tmp_path)["esc50"] == changed


def test_prepare_warns_and_update_flag_overwrites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test: A changed source prints a WARNING line; --update-fingerprints records it."""
    data_dir = use_temp_folders(monkeypatch, tmp_path)
    monkeypatch.setitem(registry.CONVERTERS, "esc50", fake_converter)
    (data_dir / "raw" / "esc50").mkdir(parents=True)
    run.main(["esc50"])
    assert "fingerprint recorded" in capsys.readouterr().out
    run.main(["esc50"])
    assert "fingerprint match" in capsys.readouterr().out

    monkeypatch.setitem(registry.CONVERTERS, "esc50", two_row_converter)
    run.main(["esc50"])
    assert "\n  WARNING: fingerprint differs" in capsys.readouterr().out
    assert load_fingerprints(tmp_path / "splits")["esc50"]["rows"] == 1

    run.main(["esc50", "--update-fingerprints"])
    assert "fingerprint updated" in capsys.readouterr().out
    assert load_fingerprints(tmp_path / "splits")["esc50"]["rows"] == 2


def test_check_with_nothing_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test: --check prints "nothing recorded" (not "match") when nothing is recorded."""
    use_temp_folders(monkeypatch, tmp_path)

    assert run.main(["--check"]) == 1
    assert capsys.readouterr().out == "nothing recorded\n"


def test_check_names_a_prepared_source_with_nothing_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test: A source prepared here but missing from fingerprints.json is named."""
    data_dir = use_temp_folders(monkeypatch, tmp_path)
    write_source_rows("esc50", [make_row("a.wav", "train", "1")], data_dir)

    assert run.main(["--check"]) == 1
    assert capsys.readouterr().out == "esc50: nothing recorded\n"


def test_dry_run_lists_every_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: --dry-run lists every datasets.yaml id and writes nothing."""
    data_dir = use_temp_folders(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "CONVERTERS", {})
    monkeypatch.setattr(run, "CONVERTERS", {})

    table = run.dry_run()

    dataset_ids: list[str] = []
    for dataset in load_datasets():
        dataset_ids.append(dataset["id"])
    assert list(table["id"]) == dataset_ids
    assert set(table["outputs"]) == {"no converter yet"}
    assert not data_dir.exists()
