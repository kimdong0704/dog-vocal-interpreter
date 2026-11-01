from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from woofdecoder.preprocess.bark_detector.timing import FRAME_S
from woofdecoder.sed_classes import SED_CLASSES


Event = tuple[float, float] # (start_s, end_s)

EVENT_CLASSES: list[str] = ["bark", "growl", "whine", "howl", "other_vocal"]
EVENT_COLUMNS: list[int] = [SED_CLASSES.index(name) for name in EVENT_CLASSES]
SETTING_NAMES: list[str] = ["threshold", "median_frames", "max_gap_s", "min_event_s"]

@dataclass(frozen=True)
class DetectionSettings:
    threshold: float = 0.5 # a frame counts as dog sound at or above this score
    median_frames: int = 1 # smooth over this many frames (odd); 1 = no smoothing
    max_gap_s: float = 0.0 # join two events whose gap is at most this long
    min_event_s: float = 0.0 # after joining, drop events shorter than this


def frames_to_events(
    active_frames: np.ndarray, frame_s: float = FRAME_S
) -> list[tuple[float, float]]:
    
    events: list[tuple[float, float]] = []
    run_start: int | None = None
    for index, active in enumerate(active_frames):
        if active and run_start is None:
            run_start = index
        elif not active and run_start is not None:
            events.append((run_start * frame_s, index * frame_s))
            run_start = None
    
    if run_start is not None:
        events.append((run_start * frame_s, len(active_frames) * frame_s))
    return events

def events_to_frames(
    events: list[Event], num_frames: int, frame_s: float = FRAME_S
) -> np.ndarray:
    # Opposite of frames_to_events: [(0.04, 0.12)] with 4 frames -> [F, T, T, F]
    # lets frame metrics score the cleaned events
    active_frames = np.zeros(num_frames, dtype=bool)
    for start_s, end_s in events:
        active_frames[round(start_s / frame_s): round(end_s / frame_s)] = True
    return active_frames

def smooth_probs(probs: np.ndarray, median_frames: int) -> np.ndarray:
    # [0.1, 0.9, 0.1, 0.1] -> [0.1, 0.1, 0.1, 0.1]
    # single high 40 ms frame = click/door/cough...
    # Removes few false alarms
    
    if median_frames <= 1:
        return probs
    
    if median_frames % 2 == 0:
        raise ValueError(f"median_frames must be odd, got {median_frames}")

    # median removes one odd value completely
    
    half = median_frames // 2
    padding = [(0, 0)] * (probs.ndim - 1) + [(half, half)]
    padded = np.pad(probs, padding, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, median_frames, axis=-1)

    return np.median(windows, axis=-1)

def merge_close_events(events: list[Event], max_gap_s: float) -> list[Event]:
    #Ex. max_gap_s=0.1 makes [(0.0, 0.2), (0.28, 0.4)] -> [(0.0, 0.4)]
    # middle of the doge bark can get quieter for moment
    # prevents events looking choppy

    merged: list[Event] = []
    for start_s, end_s in events:
        if merged and start_s - merged[-1][1] <= max_gap_s + 1e-9:
            merged[-1] = (merged[-1][0], end_s)
        else:
            merged.append((start_s, end_s))
    
    return merged

def drop_short_events(events: list[Event], min_event_s: float) -> list[Event]:
    # Keep only events at least min_event_s long
    # real dog sounds have a minimum length
    # less wasted work for three branches
    return [(start_s, end_s) for start_s, end_s in events if end_s - start_s >= min_event_s - 1e-9]

def find_dog_events(dog_prob: np.ndarray, settings: DetectionSettings) -> list[Event]:
    smoothed = smooth_probs(dog_prob, settings.median_frames)
    events = frames_to_events(smoothed >= settings.threshold)
    events = merge_close_events(events, settings.max_gap_s)
    return drop_short_events(events, settings.min_event_s)

def vocal_score(sed_probs: np.ndarray) -> np.ndarray:
    return sed_probs[..., EVENT_COLUMNS].max(axis=-1)


def settings_from_config(config: dict) -> DetectionSettings:
    section = config["bark_detector"]
    for name in SETTING_NAMES:
        if section.get(name) is None:
            raise ValueError(f"bark_detector.{name} is not set in configs/pipeline.yaml")
    
    return DetectionSettings(
        threshold=float(section["threshold"]),
        median_frames=int(section["median_frames"]),
        max_gap_s=float(section["max_gap_s"]),
        min_event_s=float(section["min_event_s"]),
    )