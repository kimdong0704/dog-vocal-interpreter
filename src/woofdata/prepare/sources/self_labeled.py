"""Self-labeled converter: the team's own labelled YouTube / phone videos.
 
Raw files (added by hand; configs/datasets.yaml "provider: manual"):
    raw/self_labeled/<collection>/labels.csv              one row per labelled span
    raw/self_labeled/<collection>/videos/<video_id>.mp4   the full video
labels.csv columns: video_id, source_url, dog_id, start_s, end_s, category, labeller.
 
Every collection has a role (COLLECTION_ROLES), which sets its splits (SPLIT_SHARES):
    branch   fills a branch's missing data. Split train / val by dog (no test: branches are
             finally tested through fusion). Each span is cut into 10 s windows.
    fusion   the fusion task's own data. Split train / val / test by dog; one row per span.
             Kept out of every branch task by woofdata.tasks (collections are opt-in there).
 
A dog may appear in one collection only, so no branch ever trains on a fusion dog.
 
Audio is taken from the video with the app's own load_audio_16k and cut_audio, so training
audio and app audio go through the same code. One 16 kHz WAV is written per span; window
rows point into it with start_s / end_s.
"""
 
from __future__ import annotations
 
from collections import Counter
from pathlib import Path
from typing import Any
 
import numpy as np
import pandas as pd
 
from woofdata.prepare.manifest import ManifestRow, make_clip_id
from woofdata.prepare.utils.audio import write_wav_16k
from woofdata.prepare.utils.hash_splits import balanced_group_splits
from woofdata.prepare.utils.label_files import write_labels
from woofdata.prepare.utils.windows import DEFAULT_WINDOW_S, make_windows
from woofdata.utils_fileio import relative_data_path
from woofdecoder.preprocess.media import TARGET_SAMPLE_RATE, cut_audio, load_audio_16k
from woofdecoder.taxonomy import CATEGORIES
 
SOURCE: str = "self_labeled"
RAW_LABELS_FILENAME: str = "labels.csv"
VIDEOS_DIRNAME: str = "videos"
VIDEO_SUFFIX: str = ".mp4"
AUDIO_DIRNAME: str = "audio"
 
BRANCH_ROLE: str = "branch"
FUSION_ROLE: str = "fusion"
# Every collection folder must be listed here, so a new one is added on purpose.
COLLECTION_ROLES: dict[str, str] = {
    "audio": BRANCH_ROLE,
    "fusion": FUSION_ROLE,
}
# Role -> (test share, val share) of its dogs; the rest is train.
SPLIT_SHARES: dict[str, tuple[float, float]] = {
    BRANCH_ROLE: (0.0, 0.2),
    FUSION_ROLE: (0.2, 0.2),
}
MIN_EVAL_DOGS: int = 4  # a val or test split with fewer dogs prints a warning
 
REQUIRED_COLUMNS: list[str] = [
    "video_id",
    "source_url",
    "dog_id",
    "start_s",
    "end_s",
    "category",
    "labeller",
]
 
WINDOW_S: float = DEFAULT_WINDOW_S
MIN_WINDOW_S: float = 1.0  # a leftover piece shorter than this is dropped
# How much shorter than its label a cut span may be before end_s counts as past the video.
SPAN_TOLERANCE_S: float = 0.1
 
SOURCE_LICENSE: str = "YouTube terms; team research use only"
NOTES: str = "team-labelled online video; do not redistribute"
LABEL_COLUMNS: list[str] = [
    "clip_id",
    "collection",
    "video_id",
    "source_url",
    "dog_id",
    "category",
    "labeller",
    "video_start_s",
    "video_end_s",
    "window_index",
    "num_windows",
]
 
 
def find_collections(raw_dir: Path) -> list[str]:
    """Return the collection folder names in raw_dir, sorted.
 
    Raises:
        ValueError: If a folder is not in COLLECTION_ROLES.
    """
    collections: list[str] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_dir():
            continue
        if path.name not in COLLECTION_ROLES:
            raise ValueError(
                f"unknown collection folder {path.name!r}; add it to COLLECTION_ROLES "
                f"({list(COLLECTION_ROLES)})"
            )
        collections.append(path.name)
    return collections
 
 
def read_collection(raw_dir: Path, collection: str) -> list[dict[str, Any]]:
    """Return one dict per labels.csv row, with collection, role and float times added.
 
    Raises:
        ValueError: If labels.csv is missing or lacks a required column.
    """
    labels_path = raw_dir / collection / RAW_LABELS_FILENAME
    if not labels_path.is_file():
        raise ValueError(f"{labels_path} is missing")
    # utf-8-sig: a CSV saved by Excel or Google Sheets may start with a byte-order mark.
    frame = pd.read_csv(labels_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing_columns: list[str] = []
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            missing_columns.append(column)
    if missing_columns:
        raise ValueError(f"{labels_path} lacks columns {missing_columns}")
 
    frame = frame.sort_values("video_id", kind="stable")
    spans: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        span: dict[str, Any] = {}
        for column in REQUIRED_COLUMNS:
            span[column] = record[column].strip()
        span["start_s"] = float(span["start_s"])
        span["end_s"] = float(span["end_s"])
        span["collection"] = collection
        span["role"] = COLLECTION_ROLES[collection]
        spans.append(span)
    return spans
 
 
def video_path_for(raw_dir: Path, span: dict[str, Any]) -> Path:
    """Return raw/self_labeled/<collection>/videos/<video_id>.mp4."""
    return raw_dir / span["collection"] / VIDEOS_DIRNAME / f"{span['video_id']}{VIDEO_SUFFIX}"
 
 
def check_spans(raw_dir: Path, spans: list[dict[str, Any]]) -> None:
    """Raise ValueError listing every problem in the label rows (before any audio is read).
 
    Checks: known category, end_s > start_s, a dog_id, the video file exists, video_id
    unique across all collections, and each dog_id in one collection only.
    """
    problems: list[str] = []
    collections_by_video: dict[str, list[str]] = {}
    collections_by_dog: dict[str, set[str]] = {}
    for span in spans:
        name = f"{span['collection']}/{span['video_id']}"
        if span["category"] not in CATEGORIES:
            problems.append(f"{name}: category {span['category']!r} is not in {CATEGORIES}")
        if span["end_s"] <= span["start_s"]:
            problems.append(f"{name}: end_s {span['end_s']} is not after start_s")
        if not span["dog_id"]:
            problems.append(f"{name}: dog_id is empty")
        if not video_path_for(raw_dir, span).is_file():
            problems.append(f"{name}: video file missing ({video_path_for(raw_dir, span)})")
        collections_by_video.setdefault(span["video_id"], []).append(span["collection"])
        collections_by_dog.setdefault(span["dog_id"], set()).add(span["collection"])
 
    for video_id, video_collections in sorted(collections_by_video.items()):
        if len(video_collections) > 1:
            problems.append(f"video_id {video_id!r} is used {len(video_collections)} times")
    for dog_id, dog_collections in sorted(collections_by_dog.items()):
        if len(dog_collections) > 1:
            problems.append(
                f"dog_id {dog_id!r} is in several collections {sorted(dog_collections)}"
            )
 
    if problems:
        raise ValueError("self_labeled label problems:\n  " + "\n  ".join(problems))
 
 
def assign_splits(spans: list[dict[str, Any]]) -> dict[str, str]:
    """Return {dog_id: split} for every dog, split within its collection by SPLIT_SHARES.
 
    Each dog counts once (not per window or video), so one long video cannot fill a split
    by itself. The order is sha1 of the dog id, the same on every machine.
    """
    dogs_by_collection: dict[str, dict[str, int]] = {}
    for span in spans:
        dogs_by_collection.setdefault(span["collection"], {})[span["dog_id"]] = 1
 
    split_by_dog: dict[str, str] = {}
    for collection, dogs_per_group in sorted(dogs_by_collection.items()):
        test_share, val_share = SPLIT_SHARES[COLLECTION_ROLES[collection]]
        collection_splits = balanced_group_splits(dogs_per_group, test_share, val_share)
        dogs_per_split: Counter[str] = Counter(collection_splits.values())
        for split, share in (("val", val_share), ("test", test_share)):
            if share > 0 and dogs_per_split[split] < MIN_EVAL_DOGS:
                print(
                    f"  WARNING: {collection} has {dogs_per_split[split]} {split} dogs "
                    f"(want {MIN_EVAL_DOGS}+); collect more dogs"
                )
        split_by_dog.update(collection_splits)
    return split_by_dog
 
 
def window_bounds(span_s: float) -> list[tuple[float, float]]:
    """Return (start_s, end_s) windows of at most WINDOW_S inside a span of span_s seconds.
 
    The last window ends at the span's end (no padding). A leftover piece shorter than
    MIN_WINDOW_S is dropped, unless it is the only window (a short clip is kept whole).
    """
    bounds: list[tuple[float, float]] = []
    for start_s, end_s in make_windows(span_s, WINDOW_S):
        end_s = round(min(end_s, span_s), 6)
        if end_s - start_s < MIN_WINDOW_S and bounds:
            continue
        bounds.append((start_s, end_s))
    return bounds
 
 
def extract_span(raw_dir: Path, span: dict[str, Any]) -> np.ndarray:
    """Return the span's audio, 16 kHz mono float32, shape (num_samples,).
 
    Raises:
        ValueError: If the video has no audio track or end_s is past the end of the video.
    """
    audio = load_audio_16k(video_path_for(raw_dir, span), verbose=False)
    name = f"{span['collection']}/{span['video_id']}"
    if audio is None:
        raise ValueError(f"{name}: the video has no audio track")
    span_audio = cut_audio(audio, span["start_s"], span["end_s"])  # shape (num_samples,)
    cut_s = len(span_audio) / TARGET_SAMPLE_RATE
    if cut_s < span["end_s"] - span["start_s"] - SPAN_TOLERANCE_S:
        video_s = len(audio) / TARGET_SAMPLE_RATE
        raise ValueError(f"{name}: end_s {span['end_s']} is past the video end ({video_s:.1f} s)")
    return span_audio
 
 
def print_report(rows: list[ManifestRow], label_rows: list[dict[str, Any]]) -> None:
    """Print rows, videos and dogs per collection and split, and rows per category."""
    rows_by_group: Counter[str] = Counter()
    videos_by_group: dict[str, set[str]] = {}
    dogs_by_group: dict[str, set[str]] = {}
    categories: Counter[str] = Counter()
    for row, label_row in zip(rows, label_rows, strict=True):
        group = f"{label_row['collection']} / {row.split}"
        rows_by_group[group] += 1
        videos_by_group.setdefault(group, set()).add(label_row["video_id"])
        dogs_by_group.setdefault(group, set()).add(label_row["dog_id"])
        categories[f"{label_row['collection']} / {row.mapped_category}"] += 1
    for group in sorted(rows_by_group):
        print(
            f"  {group}: {rows_by_group[group]} rows, {len(videos_by_group[group])} videos, "
            f"{len(dogs_by_group[group])} dogs"
        )
    print(f"  rows per category: {dict(sorted(categories.items()))}")
 
 
def prepare(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Convert every collection into span WAVs, labels.csv and manifest rows.
 
    Raises:
        ValueError: On any label problem (checked before audio is read), an unknown
            collection folder, a video without audio, or a span past the video's end.
    """
    spans: list[dict[str, Any]] = []
    for collection in find_collections(raw_dir):
        spans.extend(read_collection(raw_dir, collection))
    check_spans(raw_dir, spans)
    split_by_dog = assign_splits(spans)
 
    rows: list[ManifestRow] = []
    label_rows: list[dict[str, Any]] = []
    for span in spans:
        span_audio = extract_span(raw_dir, span)  # shape (num_samples,)
        wav_path = out_dir / span["collection"] / AUDIO_DIRNAME / f"{span['video_id']}.wav"
        write_wav_16k(wav_path, span_audio)
        span_s = len(span_audio) / TARGET_SAMPLE_RATE
 
        if span["role"] == FUSION_ROLE:
            bounds = [(0.0, round(span_s, 6))]  # fusion works on the whole labelled span
        else:
            bounds = window_bounds(span_s)
        split = split_by_dog[span["dog_id"]]
 
        member_path = f"{span['collection']}/{span['video_id']}{VIDEO_SUFFIX}"
        for window_index, (start_s, end_s) in enumerate(bounds):
            clip_id = make_clip_id(SOURCE, member_path, span["start_s"] + start_s)
            rows.append(
                ManifestRow(
                    clip_id=clip_id,
                    source=SOURCE,
                    source_license=SOURCE_LICENSE,
                    dog_id=span["dog_id"],
                    split=split,
                    modality="audio",
                    file_path=relative_data_path(wav_path),
                    start_s=start_s,
                    end_s=end_s,
                    original_label=span["category"],
                    label_type="context",
                    label_provenance="single_annotator",
                    mapped_category=span["category"],
                    eval_allowed=True,
                    notes=NOTES,
                )
            )
            label_rows.append(
                {
                    "clip_id": clip_id,
                    "collection": span["collection"],
                    "video_id": span["video_id"],
                    "source_url": span["source_url"],
                    "dog_id": span["dog_id"],
                    "category": span["category"],
                    "labeller": span["labeller"],
                    "video_start_s": span["start_s"],
                    "video_end_s": span["end_s"],
                    "window_index": window_index,
                    "num_windows": len(bounds),
                }
            )
 
    write_labels(out_dir, label_rows, LABEL_COLUMNS)
    print_report(rows, label_rows)
    return rows