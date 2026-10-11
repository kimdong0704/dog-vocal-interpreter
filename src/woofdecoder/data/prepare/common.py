"""Shared helpers for the per-source converters: audio, archives, COCO JSON and paths.

Path, JSON and CSV helpers live in woofdecoder.data.fileio.
Everything here is pure Python (no ffmpeg, no shell) so prepare gives identical results
on Windows and macOS. File lists are always sorted, and every path written to disk is
relative to the data folder with forward slashes.
"""

from __future__ import annotations

import fnmatch
import io
import math
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Any

import numpy as np
import pandas as pd
import soundfile
import soxr

from woofdecoder.config import load_config
from woofdecoder.data.download import load_datasets, select_dataset
from woofdecoder.data.fileio import relative_data_path, to_posix_path, write_csv, write_json
from woofdecoder.sed_classes import SED_CLASSES

TARGET_SAMPLE_RATE: int = load_config()["media"]["sample_rate"]
WAV_SUBTYPE: str = "PCM_16"
# soxr "HQ" quality: deterministic and good enough for 16 kHz speech-band audio.
RESAMPLE_QUALITY: str = "HQ"

DEFAULT_WINDOW_S: float = 10.0
DEFAULT_FRAME_S: float = 0.04  # 25 frames per second, 250 frames per 10 s window
# Tolerance for float rounding when an event boundary falls exactly on a frame boundary.
FRAME_BOUNDARY_EPSILON: float = 1e-6

SED_CLASSES_FILENAME: str = "sed_classes.json"
EVENTS_FILENAME: str = "events.csv"
# clip_id plus every original label column of the source, for labels the manifest has no
# column for (environment, arousal, the authors' split, ...).
LABELS_FILENAME: str = "labels.csv"
# One row per annotated event; sed_class is "" when the label maps to no SED class.
EVENT_COLUMNS: list[str] = ["clip_id", "start_s", "end_s", "sed_class", "original_label"]


# --- Audio ---


def load_audio(source: str | Path | bytes) -> tuple[np.ndarray, int]:
    """Read an audio file (path or raw bytes) of any bit depth or channel count.

    Returns:
        (samples, sample_rate): float32 samples in [-1, 1], shape (num_samples, num_channels).
    """
    audio_file: str | Path | io.BytesIO
    if isinstance(source, bytes):
        audio_file = io.BytesIO(source)
    else:
        audio_file = source
    samples, sample_rate = soundfile.read(audio_file, dtype="float32", always_2d=True)
    return samples, sample_rate  # samples shape (num_samples, num_channels)


def to_mono_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Average the channels and resample to TARGET_SAMPLE_RATE with soxr.

    Args:
        samples: shape (num_samples,) or (num_samples, num_channels).
        sample_rate: Sample rate of samples in Hz.

    Returns:
        float32 samples, shape (num_samples_16k,).
    """
    if samples.ndim == 2:
        mono = samples.mean(axis=1)  # shape (num_samples,)
    else:
        mono = samples
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = soxr.resample(mono, sample_rate, TARGET_SAMPLE_RATE, quality=RESAMPLE_QUALITY)
    return mono.astype(np.float32, copy=False)  # shape (num_samples_16k,)


def write_wav_16k(path: Path, samples: np.ndarray) -> None:
    """Write mono samples (shape (num_samples,)) as a TARGET_SAMPLE_RATE 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)  # values beyond full scale would wrap around
    soundfile.write(path, clipped, TARGET_SAMPLE_RATE, subtype=WAV_SUBTYPE)


def dataset_config(source: str) -> dict[str, Any]:
    """Return the configs/datasets.yaml entry of one source (for its pinned revision)."""
    return select_dataset(load_datasets(), source)[0]


def convert_audio(source: str | Path | bytes, out_path: Path) -> float:
    """Read any audio file (path or bytes), write it as 16 kHz mono WAV, return its seconds."""
    samples, sample_rate = load_audio(source)
    mono = to_mono_16k(samples, sample_rate)  # shape (num_samples_16k,)
    write_wav_16k(out_path, mono)
    return len(mono) / TARGET_SAMPLE_RATE


def make_windows(
    duration_s: float, window_s: float = DEFAULT_WINDOW_S, hop_s: float = DEFAULT_WINDOW_S
) -> list[tuple[float, float]]:
    """Cut a clip into (start_s, end_s) windows; the last one runs past the end (padded).

    Windows stop as soon as one reaches the end of the clip, so nothing is dropped and no
    window starts after the last sample.
    """
    windows: list[tuple[float, float]] = []
    window_index = 0
    while duration_s > 0:
        start_s = round(window_index * hop_s, 6)  # multiply, not add, so rounding can't drift
        end_s = round(start_s + window_s, 6)
        windows.append((start_s, end_s))
        if end_s >= duration_s:
            break
        window_index += 1
    return windows


def events_to_frame_targets(
    events: list[tuple[float, float, str]],
    class_names: list[str],
    window_start_s: float,
    window_s: float = DEFAULT_WINDOW_S,
    frame_s: float = DEFAULT_FRAME_S,
) -> np.ndarray:
    """Turn (start_s, end_s, class_name) events into per-frame 0/1 targets for one window.

    A frame is 1 for a class if it overlaps one of that class's events; an event that only
    touches a frame's edge does not count.

    Returns:
        float32 array, shape (num_frames, num_classes), e.g. (250, C) for 10 s at 0.04 s.

    Raises:
        ValueError: If an event's class is not in class_names.
    """
    num_frames = round(window_s / frame_s)
    targets = np.zeros((num_frames, len(class_names)), dtype=np.float32)
    for event_start_s, event_end_s, class_name in events:
        if class_name not in class_names:
            raise ValueError(f"event class {class_name!r} is not in {class_names}")
        class_index = class_names.index(class_name)
        # Event edges in frame units, relative to the window start.
        first_frame_float = (event_start_s - window_start_s) / frame_s
        end_frame_float = (event_end_s - window_start_s) / frame_s
        first_frame = max(0, math.floor(first_frame_float + FRAME_BOUNDARY_EPSILON))
        end_frame = min(num_frames, math.ceil(end_frame_float - FRAME_BOUNDARY_EPSILON))
        if first_frame < end_frame:
            targets[first_frame:end_frame, class_index] = 1.0
    return targets


def save_frame_targets(path: Path, targets: np.ndarray) -> None:
    """Save one window's frame targets (shape (num_frames, len(SED_CLASSES))) as .npy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, targets)


# --- Sound-event files ---


def write_sed_classes(out_dir: Path, labelled_classes: list[str]) -> None:
    """Write out_dir/sed_classes.json: SED_CLASSES and whether this source labels each one.

    Evaluation masks the classes a source does not label, so a missing pant label in
    AudioSet is not counted as "no pant here".
    """
    labelled: dict[str, bool] = {}
    for class_name in SED_CLASSES:
        labelled[class_name] = class_name in labelled_classes
    write_json(out_dir / SED_CLASSES_FILENAME, {"classes": SED_CLASSES, "labelled": labelled})


def write_events(out_dir: Path, events: list[dict[str, Any]]) -> None:
    """Write out_dir/events.csv (EVENT_COLUMNS), sorted by every column so it is the same
    on every machine."""
    frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    frame = frame.sort_values(EVENT_COLUMNS, kind="stable")
    write_csv(frame, out_dir / EVENTS_FILENAME)


def write_labels(out_dir: Path, label_rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write out_dir/labels.csv with the given columns (clip_id first), sorted by clip_id."""
    frame = pd.DataFrame(label_rows, columns=columns)
    frame = frame.sort_values("clip_id", kind="stable")
    write_csv(frame, out_dir / LABELS_FILENAME)


# --- Archives ---


def list_archive_members(archive_path: Path) -> dict[str, int]:
    """Return {member name: size in bytes} for every file (not folder) in a zip or tar."""
    member_sizes: dict[str, int] = {}
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zip_archive:
            for info in zip_archive.infolist():
                if not info.is_dir():
                    member_sizes[info.filename] = info.file_size
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tar_archive:
            for member in tar_archive.getmembers():
                if member.isfile():
                    member_sizes[member.name] = member.size
    else:
        raise ValueError(f"{archive_path} is neither a zip nor a tar archive")
    return member_sizes


def select_members(member_names: list[str], wanted: list[str] | str) -> list[str]:
    """Pick members by exact names (list) or one glob pattern (str), sorted.

    Patterns match case-sensitively on every OS (fnmatchcase), so results are the same
    on Windows and macOS.

    Raises:
        ValueError: If a named member is missing, or the pattern matches nothing.
    """
    selected: list[str] = []
    if isinstance(wanted, str):
        for member_name in member_names:
            if fnmatch.fnmatchcase(member_name, wanted):
                selected.append(member_name)
        if not selected:
            raise ValueError(f"no archive member matches {wanted!r}")
    else:
        for member_name in wanted:
            if member_name not in member_names:
                raise ValueError(f"archive has no member {member_name!r}")
            selected.append(member_name)
    return sorted(selected)


def safe_member_path(out_dir: Path, member_name: str) -> Path:
    """Return where member_name goes inside out_dir; refuse names that would escape it."""
    member_path = PurePosixPath(to_posix_path(member_name))
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"unsafe archive member path: {member_name}")
    return out_dir.joinpath(*member_path.parts)


def open_archive(archive_path: Path) -> zipfile.ZipFile | tarfile.TarFile:
    """Open a zip or tar (use it in a with statement)."""
    if zipfile.is_zipfile(archive_path):
        return zipfile.ZipFile(archive_path)
    return tarfile.open(archive_path)


def open_member(archive: zipfile.ZipFile | tarfile.TarFile, member_name: str) -> IO[bytes]:
    """Open one member of an already opened zip or tar for reading."""
    if isinstance(archive, zipfile.ZipFile):
        return archive.open(member_name)
    member_file = archive.extractfile(member_name)
    if member_file is None:
        raise ValueError(f"{member_name} is not a regular file")
    return member_file


def extract_members(archive_path: Path, wanted: list[str] | str, out_dir: Path) -> list[Path]:
    """Extract only the wanted members of a zip or tar into out_dir, keeping their folders.

    Members already extracted with the right size are skipped, so reruns are fast.

    Args:
        archive_path: The .zip, .tar or .tar.gz file.
        wanted: Exact member names, or one glob pattern such as "Images/*.jpg".
        out_dir: Folder to extract into.

    Returns:
        The extracted file paths, sorted by member name.
    """
    member_sizes = list_archive_members(archive_path)
    selected = select_members(sorted(member_sizes), wanted)
    extracted_paths: list[Path] = []
    with open_archive(archive_path) as archive:
        for member_name in selected:
            target_path = safe_member_path(out_dir, member_name)
            extracted_paths.append(target_path)
            if target_path.is_file() and target_path.stat().st_size == member_sizes[member_name]:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                open_member(archive, member_name) as source_file,
                open(target_path, "wb") as target_file,
            ):
                shutil.copyfileobj(source_file, target_file)
    return extracted_paths


# --- COCO ---


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
