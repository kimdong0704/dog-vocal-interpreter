"""Load one sample of a task from its manifest row (see tasks.TASKS for the sample types)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from woofdata.loader.annotations import Row
from woofdata.loader.audio.task_loaders import (
    load_audio_context,
    load_audio_emotion,
    load_bark_detector,
)
from woofdata.loader.image.task_loaders import load_pose, load_posture, load_scene_objects
from woofdata.tasks import get_task

SampleLoader = Callable[[Row, float | None], Any]
LOADERS: dict[str, SampleLoader] = {
    "bark_detector": load_bark_detector,
    "audio_context": load_audio_context,
    "audio_emotion": load_audio_emotion,
    "pose": load_pose,
    "posture": load_posture,
    "scene_objects": load_scene_objects,
}


def load_sample(task: str, row: Row, max_seconds: float | None = None) -> Any:
    """Load one manifest row as the task's sample dataclass (see tasks.TASKS).

    Args:
        task: A TASKS key.
        row: One row from task_rows(task, ...).
        max_seconds: Audio length cap; None uses the task's default (10 s for the
            audio_context and audio_emotion tasks, no cap otherwise).

    Raises:
        ValueError: If the row's source is not one of the task's sources.
        RuntimeError: If loading fails; the message names the task, clip_id and file.
    """
    task_info = get_task(task)
    if row["source"] not in task_info.sources:
        raise ValueError(f"{task}: {row['clip_id']} is from {row['source']}, not this task")
    if max_seconds is None:
        max_seconds = task_info.default_max_seconds
    try:
        return LOADERS[task](row, max_seconds)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        message = f"{task}: cannot load {row['clip_id']} (file {row['file_path']}): {error!r}"
        raise RuntimeError(message) from error
