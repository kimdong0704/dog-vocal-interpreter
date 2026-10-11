"""EmotionalCanines converter: husky and shiba clips labelled with arousal and valence.

Raw files in raw/emotional_canines/ (pinned Hugging Face commit):
    {train,validation,test}/{husky,shiba}/<audio_id>.wav   e.g. train/husky/husky_train_00000.wav
    {husky,shiba}_{train,validation}_labels.csv           columns audio_id, arousal, valence
The test split has no labels and is skipped. One row per whole clip, no windows.
Emotion feeds an auxiliary head, so mapped_subtype stays empty.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd

from woofdecoder.data.fileio import read_csv, relative_data_path
from woofdecoder.data.manifest import ManifestRow, make_clip_id
from woofdecoder.data.prepare.common import convert_audio, dataset_config, write_labels

SOURCE: str = "emotional_canines"
BREEDS: tuple[str, ...] = ("husky", "shiba")
# Author split folder -> manifest split. "test" is left out on purpose: it has no labels.
SPLIT_BY_FOLDER: dict[str, str] = {"train": "train_only", "validation": "val"}
UNLABELLED_FOLDER: str = "test"
AUDIO_DIRNAME: str = "audio"
LEAKAGE_GROUP: str = "barkopedia_pool"
SOURCE_LICENSE: str = "CC-BY-NC-SA-4.0"
NOTES: str = "annotation method not documented; relicensed from MIT on 2026-09-16"
LABEL_COLUMNS: list[str] = ["clip_id", "audio_id", "arousal", "valence", "breed", "original_split"]


def labels_path(raw_dir: Path, breed: str, folder: str) -> Path:
    """Return raw_dir/<breed>_<folder>_labels.csv."""
    return raw_dir / f"{breed}_{folder}_labels.csv"


def read_split_labels(raw_dir: Path, folder: str) -> list[dict[str, str]]:
    """Return one author split's label rows (both breeds), with breed and original_split.

    Raises:
        ValueError: If a labelled clip has no WAV file.
    """
    clips: list[dict[str, str]] = []
    for breed in BREEDS:
        breed_labels_path = labels_path(raw_dir, breed, folder)
        for record in read_csv(breed_labels_path).to_dict("records"):
            audio_path = raw_dir / folder / breed / f"{record['audio_id']}.wav"
            if not audio_path.is_file():
                raise ValueError(f"{breed_labels_path.name} lists {audio_path.name}, not on disk")
            clips.append(record | {"breed": breed, "original_split": folder})
    return clips


def count_unlabelled(raw_dir: Path) -> int:
    """Return the number of WAVs in the unlabelled test folder (skipped, not converted)."""
    return len(list((raw_dir / UNLABELLED_FOLDER).glob("*/*.wav")))


def convert_clip(
    clip: dict[str, str], raw_dir: Path, out_dir: Path, source_version: str
) -> tuple[ManifestRow, float]:
    """Write one clip's 16 kHz WAV; return its manifest row and duration in seconds."""
    member_path = f"{clip['original_split']}/{clip['breed']}/{clip['audio_id']}.wav"
    wav_path = out_dir / AUDIO_DIRNAME / member_path
    duration_s = convert_audio(raw_dir / member_path, wav_path)
    split = SPLIT_BY_FOLDER[clip["original_split"]]
    row = ManifestRow(
        clip_id=make_clip_id(SOURCE, member_path),
        source=SOURCE,
        source_version=source_version,
        source_license=SOURCE_LICENSE,
        leakage_group=LEAKAGE_GROUP,
        split=split,
        modality="audio",
        file_path=relative_data_path(wav_path),
        start_s=0.0,
        end_s=round(duration_s, 6),
        original_label=f"{clip['arousal']}/{clip['valence']}",
        label_type="emotion",
        label_provenance="uploader",
        eval_allowed=split == "val",
        notes=NOTES,
    )
    return row, duration_s


def print_report(label_rows: list[dict[str, str]], durations: list[float], skipped: int) -> None:
    """Print skipped clips, clips per split, arousal x valence per split, durations."""
    labels = pd.DataFrame(label_rows)
    print(f"  skipped {skipped} unlabelled {UNLABELLED_FOLDER} clips")
    print(f"  clips per split: {labels['original_split'].value_counts().sort_index().to_dict()}")
    for folder in SPLIT_BY_FOLDER:
        folder_labels = labels[labels["original_split"] == folder]
        table = pd.crosstab(folder_labels["arousal"], folder_labels["valence"], margins=True)
        print(f"  arousal x valence, {folder}:\n{table.to_string()}")
    median_s = statistics.median(durations)
    print(
        f"  duration s: min {min(durations):.2f}, median {median_s:.2f}, max {max(durations):.2f}"
    )


def prepare(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Convert the labelled EmotionalCanines clips into WAVs, labels.csv and rows."""
    source_version = dataset_config(SOURCE)["revision"]
    rows: list[ManifestRow] = []
    label_rows: list[dict[str, str]] = []
    durations: list[float] = []
    for folder in SPLIT_BY_FOLDER:
        for clip in read_split_labels(raw_dir, folder):
            row, duration_s = convert_clip(clip, raw_dir, out_dir, source_version)
            rows.append(row)
            label_rows.append({"clip_id": row.clip_id} | clip)
            durations.append(duration_s)

    write_labels(out_dir, label_rows, LABEL_COLUMNS)
    print_report(label_rows, durations, count_unlabelled(raw_dir))
    return rows
