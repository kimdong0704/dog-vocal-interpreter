"""Loaders for the audio tasks: bark_detector, audio_context and audio_emotion."""

from __future__ import annotations

import numpy as np

from woofdata.loader.annotations import Row, labels_for, read_sed_mask_file
from woofdata.loader.audio.segment import load_audio_segment
from woofdata.path import data_path
from woofdata.prepare.manifest import processed_dir
from woofdata.prepare.sources import esc50
from woofdata.prepare.utils.label_files import SED_CLASSES_FILENAME
from woofdata.prepare.utils.windows import DEFAULT_FRAME_S, DEFAULT_WINDOW_S
from woofdata.tasks import AudioContextSample, AudioEmotionSample, BarkDetectorSample
from woofdecoder.sed_classes import SED_CLASSES

NUM_FRAMES: int = round(DEFAULT_WINDOW_S / DEFAULT_FRAME_S)  # 250 frames per 10 s window


def load_bark_detector(row: Row, max_seconds: float | None) -> BarkDetectorSample:
    """Return a window with its frame targets and the source's labelled-class mask.

    Raises:
        ValueError: If a row without a target file is not an ESC-50 dog clip.
    """
    audio = load_audio_segment(row, max_seconds)
    if row["target_path"]:
        targets = np.load(data_path(row["target_path"]))  # shape (250, 6)
        sed_classes_path = processed_dir() / row["source"] / SED_CLASSES_FILENAME
        mask = read_sed_mask_file(sed_classes_path).copy()  # shape (6,)
        is_dog = bool(targets.any())
    elif row["source"] == esc50.SOURCE and row["original_label"] == esc50.DOG_CATEGORY:
        targets = np.zeros((NUM_FRAMES, len(SED_CLASSES)), dtype=np.float32)
        mask = np.zeros(len(SED_CLASSES), dtype=bool)  # nothing known per frame
        is_dog = True
    else:
        raise ValueError("row has no target_path")
    return BarkDetectorSample(row["clip_id"], audio, targets, mask, is_dog)


def load_audio_context(row: Row, max_seconds: float | None) -> AudioContextSample:
    """Return a clip, its mapped subtype (None if empty) and its labels.csv fields."""
    audio = load_audio_segment(row, max_seconds)
    mapped_subtype = row["mapped_subtype"] or None
    return AudioContextSample(row["clip_id"], audio, mapped_subtype, labels_for(row))


def load_audio_emotion(row: Row, max_seconds: float | None) -> AudioEmotionSample:
    """Return a clip with its arousal and valence from labels.csv."""
    audio = load_audio_segment(row, max_seconds)
    labels = labels_for(row)
    return AudioEmotionSample(row["clip_id"], audio, labels["arousal"], labels["valence"])
