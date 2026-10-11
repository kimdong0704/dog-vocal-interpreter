"""The manifest: one row per audio clip/window or image, across all prepared sources.

Each source writes its own rows to <data_dir>/processed/<source>/rows.csv;
build_manifest() joins them into <data_dir>/processed/manifest.csv.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path, PurePath

import pandas as pd

from woofdecoder.config import get_data_dir
from woofdecoder.data.download import load_datasets
from woofdecoder.data.fileio import read_csv, to_posix_path, write_csv
from woofdecoder.taxonomy import SUBTYPE_TO_CATEGORY, SUBTYPES

PROCESSED_DIRNAME: str = "processed"
MANIFEST_FILENAME: str = "manifest.csv"
ROWS_FILENAME: str = "rows.csv"

SPLITS: tuple[str, ...] = ("train", "val", "test", "train_only")
MODALITIES: tuple[str, ...] = ("audio", "image")
LABEL_TYPES: tuple[str, ...] = (
    "context",
    "emotion",
    "call_type",
    "identity",
    "posture",
    "object",
    "keypoints",
)
LABEL_PROVENANCES: tuple[str, ...] = (
    "protocol",
    "expert",
    "single_annotator",
    "vlm_video",
    "uploader",
)


def check_choice(column: str, value: str, choices: tuple[str, ...] | list[str]) -> None:
    """Raise ValueError if value is not one of choices."""
    if value not in choices:
        raise ValueError(f"{column} must be one of {list(choices)}, not {value!r}")


@dataclass(kw_only=True)
class ManifestRow:
    """One manifest row. Field order is the column order of manifest.csv.

    dog_id is stored as "<source>_<dog id>" so ids from different sources never collide;
    without a dog id it becomes "<source>_<clip_id>" (each clip is its own group).
    leakage_group defaults to dog_id. mapped_category is derived from mapped_subtype.
    """

    clip_id: str
    source: str
    source_version: str = ""
    source_license: str = ""
    dog_id: str = ""
    leakage_group: str = ""
    split: str
    modality: str
    file_path: str
    start_s: float | None = None
    end_s: float | None = None
    target_path: str = ""
    annotation_path: str = ""
    original_label: str = ""
    label_type: str
    label_provenance: str
    mapped_subtype: str = ""
    mapped_category: str = field(default="", init=False)
    eval_allowed: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        """Check the fixed-choice columns, fill the group keys and derive mapped_category."""
        check_choice("split", self.split, SPLITS)
        check_choice("modality", self.modality, MODALITIES)
        check_choice("label_type", self.label_type, LABEL_TYPES)
        check_choice("label_provenance", self.label_provenance, LABEL_PROVENANCES)
        if self.mapped_subtype:
            check_choice("mapped_subtype", self.mapped_subtype, SUBTYPES)
            self.mapped_category = SUBTYPE_TO_CATEGORY[self.mapped_subtype]

        if self.dog_id:
            self.dog_id = f"{self.source}_{self.dog_id}"
        else:
            self.dog_id = f"{self.source}_{self.clip_id}"
        if not self.leakage_group:
            self.leakage_group = self.dog_id


MANIFEST_COLUMNS: list[str] = []
for manifest_field in fields(ManifestRow):
    MANIFEST_COLUMNS.append(manifest_field.name)


def make_clip_id(source: str, member_path: str | PurePath, start_s: float | None = None) -> str:
    """Return a clip id that is the same on every machine, e.g. "esc50/audio/1-100.wav@10.000".

    Args:
        source: Dataset id from configs/datasets.yaml.
        member_path: Path of the clip inside the source (Windows or POSIX style).
        start_s: Window start for clips cut into windows, else None.
    """
    clip_id = f"{source}/{to_posix_path(member_path)}"
    if start_s is not None:
        clip_id += f"@{start_s:.3f}"
    return clip_id


def processed_dir(data_dir: Path | None = None) -> Path:
    """Return <data_dir>/processed/."""
    if data_dir is None:
        data_dir = get_data_dir()
    return data_dir / PROCESSED_DIRNAME


def rows_to_frame(rows: list[ManifestRow]) -> pd.DataFrame:
    """Return rows as a DataFrame in MANIFEST_COLUMNS order, sorted by clip_id.

    Raises:
        ValueError: If two rows share a clip_id.
    """
    records: list[dict] = []
    for row in rows:
        records.append(asdict(row))
    frame = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    duplicated = frame["clip_id"][frame["clip_id"].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"duplicate clip_id values: {sorted(set(duplicated))[:10]}")
    return frame.sort_values("clip_id", kind="stable").reset_index(drop=True)


def write_source_rows(source: str, rows: list[ManifestRow], data_dir: Path | None = None) -> Path:
    """Write one source's rows to processed/<source>/rows.csv and return that path."""
    rows_path = processed_dir(data_dir) / source / ROWS_FILENAME
    write_csv(rows_to_frame(rows), rows_path)
    return rows_path


def build_manifest(data_dir: Path | None = None) -> pd.DataFrame:
    """Join every prepared source's rows.csv (datasets.yaml order), sort by clip_id, save.

    Returns:
        The manifest; also written to processed/manifest.csv. Empty if nothing is prepared.
    """
    source_frames: list[pd.DataFrame] = []
    for dataset in load_datasets():
        rows_path = processed_dir(data_dir) / dataset["id"] / ROWS_FILENAME
        if rows_path.is_file():
            source_frames.append(read_csv(rows_path))
    if not source_frames:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    manifest = pd.concat(source_frames, ignore_index=True)
    manifest = manifest.sort_values("clip_id", kind="stable").reset_index(drop=True)
    write_csv(manifest, processed_dir(data_dir) / MANIFEST_FILENAME)
    return manifest
