"""Cut clips into fixed windows and turn sound events into per-frame targets."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

DEFAULT_WINDOW_S: float = 10.0
DEFAULT_FRAME_S: float = 0.04  # 25 frames per second, 250 frames per 10 s window
# Tolerance for float rounding when an event boundary falls exactly on a frame boundary.
FRAME_BOUNDARY_EPSILON: float = 1e-6


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
