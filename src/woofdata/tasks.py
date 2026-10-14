"""Training tasks: which sources feed each model, and what one loaded sample holds.

A teammate working on one model needs only:
    rows = task_rows("pose", "train")
    sample = load_sample("pose", rows.iloc[0])      # woofdata.loader
and never opens manifest.csv, labels.csv or a COCO JSON by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from woofdata.path import load_data_dir_from_env_file
from woofdata.prepare.manifest import MANIFEST_FILENAME, processed_dir
from woofdata.prepare.utils.windows import DEFAULT_WINDOW_S
from woofdata.utils_fileio import read_csv

# Manifest split values that count as training for every task. train_only rows (labels
# not trusted enough to report on) are trained on but never evaluated.
TRAIN_SPLITS: tuple[str, ...] = ("train", "train_only")
EVAL_SPLITS: tuple[str, ...] = ("val", "test")
TEST_REMINDER: str = "Reminder: test is reported once, at the end. Tune on val."
# Act_Env clips run up to 93 s; the clip-level audio tasks keep the centre of each clip.
CLIP_MAX_SECONDS: float = DEFAULT_WINDOW_S


# --- Samples: what load_sample returns for each task ---


@dataclass
class BarkDetectorSample:
    """One 10 s window with sound-event frame targets.

    mask says which SED classes the source labels; a False class tells nothing about
    its frames (ESC-50 dog clips have only a clip-level label: all False, is_dog True).
    """

    clip_id: str
    audio: np.ndarray  # shape (num_samples,), float32 at 16 kHz
    targets: np.ndarray  # shape (num_frames, len(SED_CLASSES)), float32 0/1
    mask: np.ndarray  # shape (len(SED_CLASSES),), bool
    is_dog: bool  # a dog vocalization somewhere in the window


@dataclass
class AudioContextSample:
    """One clip with its bark context. labels holds the source's labels.csv fields."""

    clip_id: str
    audio: np.ndarray  # shape (num_samples,), float32 at 16 kHz
    mapped_subtype: str | None  # a taxonomy subtype, None if the clip maps to none
    labels: dict[str, str]  # e.g. environment_normalized, activity, original_split


@dataclass
class AudioEmotionSample:
    """One clip with its arousal and valence labels (as written by the authors)."""

    clip_id: str
    audio: np.ndarray  # shape (num_samples,), float32 at 16 kHz
    arousal: str
    valence: str


@dataclass
class PoseSample:
    """One image with one dog box and its 24 keypoints (v = 0 means not labelled)."""

    clip_id: str
    image: np.ndarray  # shape (height, width, 3), uint8 RGB
    box: np.ndarray  # shape (4,), float32 x, y, w, h in pixels
    keypoints: np.ndarray  # shape (24, 3), float32 x, y, visibility
    keypoint_names: list[str]


@dataclass
class PostureSample:
    """One image with its largest dog box and that dog's posture."""

    clip_id: str
    image: np.ndarray  # shape (height, width, 3), uint8 RGB
    box: np.ndarray  # shape (4,), float32 x, y, w, h in pixels
    posture: str


@dataclass
class SceneObjectsSample:
    """One image with every object box. Category ids: 1 dog, 2 person, 3 leash, 4 bowl,
    5 cat (UNIFIED_CATEGORIES order). tags holds the labels.csv has_* / other_dog flags."""

    clip_id: str
    image: np.ndarray  # shape (height, width, 3), uint8 RGB
    boxes: np.ndarray  # shape (num_boxes, 4), float32 x, y, w, h in pixels
    category_ids: np.ndarray  # shape (num_boxes,), int64
    tags: dict[str, bool]


# --- The registry ---


@dataclass(frozen=True)
class Task:
    """One training task: its sources, its training splits and its sample type."""

    sources: tuple[str, ...]
    sample_type: type
    default_max_seconds: float | None = None  # audio is centre-cropped to this length
    train_splits: tuple[str, ...] = TRAIN_SPLITS


TASKS: dict[str, Task] = {
    "bark_detector": Task(("playpant", "audioset_strong_eval", "esc50"), BarkDetectorSample),
    # Act_Env trains (all train_only); PlayPanT (subtype play) is the val / test set.
    "audio_context": Task(("act_env", "playpant"), AudioContextSample, CLIP_MAX_SECONDS),
    "audio_emotion": Task(("emotional_canines",), AudioEmotionSample, CLIP_MAX_SECONDS),
    "pose": Task(("stanfordextra",), PoseSample),
    "posture": Task(("dogposecv",), PostureSample),
    "scene_objects": Task(("rf_leash", "rf_bowl"), SceneObjectsSample),
}


def get_task(task: str) -> Task:
    """Return TASKS[task].

    Raises:
        ValueError: If task is not in TASKS.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose one of {list(TASKS)}")
    return TASKS[task]


# --- Rows ---


@cache
def read_manifest_file(manifest_path: Path) -> pd.DataFrame:
    """Read manifest.csv once per process, with start_s / end_s as floats (NaN for images)
    and eval_allowed as bool."""
    manifest = read_csv(manifest_path)
    manifest["start_s"] = pd.to_numeric(manifest["start_s"])
    manifest["end_s"] = pd.to_numeric(manifest["end_s"])
    manifest["eval_allowed"] = manifest["eval_allowed"] == "True"
    return manifest


def read_manifest() -> pd.DataFrame:
    """Return the cached manifest of the current data folder (do not modify it in place).

    Raises:
        FileNotFoundError: If processed/manifest.csv does not exist (run prepare first).
    """
    load_data_dir_from_env_file()
    manifest_path = processed_dir() / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} is missing; run woofdata.prepare")
    return read_manifest_file(manifest_path)


def task_rows(
    task: str, split: str, eval_only: bool = False, sources: list[str] | None = None
) -> pd.DataFrame:
    """Return the manifest rows of one task and split, indexed 0..n-1.

    Args:
        task: A TASKS key, e.g. "pose".
        split: "train" (rows in train or train_only), "val" or "test".
        eval_only: Keep only eval_allowed rows (for reporting scores).
        sources: Keep only these of the task's sources (default all of them).

    Raises:
        ValueError: If the task, split or a source is unknown.
    """
    task_info = get_task(task)
    if split == "train":
        wanted_splits = task_info.train_splits
    elif split in EVAL_SPLITS:
        wanted_splits = (split,)
    else:
        raise ValueError(f"split must be 'train', 'val' or 'test', not {split!r}")
    if sources is None:
        sources = list(task_info.sources)
    for source in sources:
        if source not in task_info.sources:
            raise ValueError(f"{task} uses sources {list(task_info.sources)}, not {source!r}")
    if split == "test":
        print(TEST_REMINDER)

    manifest = read_manifest()
    keep = manifest["source"].isin(sources) & manifest["split"].isin(wanted_splits)
    if eval_only:
        keep = keep & manifest["eval_allowed"]
    return manifest.loc[keep].reset_index(drop=True)
