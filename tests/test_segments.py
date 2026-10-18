"""Tests for BarkSegment: the new detector and dog-box fields and their checks."""

from __future__ import annotations

import numpy as np
import pytest

from woofdecoder.preprocess.segments import BarkSegment

NUM_FRAMES: int = 4
FRAME_HEIGHT: int = 48
FRAME_WIDTH: int = 64


def make_frames(num_frames: int) -> list[np.ndarray]:
    """Return num_frames black RGB frames, each of shape (FRAME_HEIGHT, FRAME_WIDTH, 3)."""
    frames: list[np.ndarray] = []
    for _ in range(num_frames):
        frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
    return frames


def make_frame_times(num_frames: int) -> list[float]:
    """Return num_frames timestamps in seconds, 0.25 s apart."""
    frame_times_s: list[float] = []
    for frame_index in range(num_frames):
        frame_times_s.append(frame_index * 0.25)
    return frame_times_s


def make_dog_boxes() -> list[np.ndarray | None]:
    """Return NUM_FRAMES boxes as x, y, w, h in pixels, with no dog in the third frame."""
    return [
        np.array([10.0, 5.0, 20.0, 30.0]),
        np.array([12.0, 6.0, 20.0, 30.0]),
        None,
        np.array([14.0, 7.0, 20.0, 30.0]),
    ]


def test_valid_segment_with_dog_boxes() -> None:
    segment = BarkSegment(
        start_s=1.0,
        end_s=2.0,
        frames=make_frames(NUM_FRAMES),
        frame_times_s=make_frame_times(NUM_FRAMES),
        sed_class="bark",
        confidence=0.9,
        dog_boxes=make_dog_boxes(),
    )

    assert segment.has_dog_boxes
    assert segment.dog_visible
    assert segment.dog_boxes[2] is None
    first_box = segment.dog_boxes[0]
    assert first_box is not None
    assert first_box.dtype == np.float32
    assert first_box.shape == (4,)


def test_boxes_given_as_lists_become_float32_arrays() -> None:
    segment = BarkSegment(
        start_s=0.0,
        end_s=1.0,
        frames=make_frames(1),
        frame_times_s=make_frame_times(1),
        dog_boxes=[[1, 2, 3, 4]],
    )

    first_box = segment.dog_boxes[0]
    assert isinstance(first_box, np.ndarray)
    assert first_box.dtype == np.float32


def test_no_dog_boxes_means_detection_not_run() -> None:
    segment = BarkSegment(
        start_s=0.0,
        end_s=1.0,
        frames=make_frames(NUM_FRAMES),
        frame_times_s=make_frame_times(NUM_FRAMES),
    )

    assert not segment.has_dog_boxes
    assert not segment.dog_visible


def test_all_boxes_none_means_no_dog_found() -> None:
    segment = BarkSegment(
        start_s=0.0,
        end_s=1.0,
        frames=make_frames(NUM_FRAMES),
        frame_times_s=make_frame_times(NUM_FRAMES),
        dog_boxes=[None, None, None, None],
    )

    assert segment.has_dog_boxes
    assert not segment.dog_visible


def test_minimal_segment_still_works() -> None:
    segment = BarkSegment(start_s=0.0, end_s=1.0)

    assert segment.duration_s == 1.0
    assert segment.sed_class == ""
    assert segment.confidence is None
    assert not segment.has_dog_boxes


def test_end_before_start_raises() -> None:
    with pytest.raises(ValueError, match="end_s .* is before start_s"):
        BarkSegment(start_s=2.0, end_s=1.0)


def test_frame_times_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="frame_times_s has 3 entries but frames has 4"):
        BarkSegment(
            start_s=0.0,
            end_s=1.0,
            frames=make_frames(NUM_FRAMES),
            frame_times_s=make_frame_times(3),
        )


def test_dog_boxes_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dog_boxes has 3 entries but frames has 4"):
        BarkSegment(
            start_s=0.0,
            end_s=1.0,
            frames=make_frames(NUM_FRAMES),
            frame_times_s=make_frame_times(NUM_FRAMES),
            dog_boxes=make_dog_boxes()[:3],
        )


def test_box_with_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"frame 0 has shape \(3,\)"):
        BarkSegment(
            start_s=0.0,
            end_s=1.0,
            frames=make_frames(1),
            frame_times_s=make_frame_times(1),
            dog_boxes=[np.array([1.0, 2.0, 3.0])],
        )


def test_box_with_zero_width_raises() -> None:
    with pytest.raises(ValueError, match="frame 0 has w=0.0"):
        BarkSegment(
            start_s=0.0,
            end_s=1.0,
            frames=make_frames(1),
            frame_times_s=make_frame_times(1),
            dog_boxes=[np.array([1.0, 2.0, 0.0, 4.0])],
        )


def test_box_with_negative_height_raises() -> None:
    with pytest.raises(ValueError, match="h=-4.0; both must be > 0"):
        BarkSegment(
            start_s=0.0,
            end_s=1.0,
            frames=make_frames(1),
            frame_times_s=make_frame_times(1),
            dog_boxes=[np.array([1.0, 2.0, 3.0, -4.0])],
        )


def test_unknown_sed_class_raises() -> None:
    with pytest.raises(ValueError, match="sed_class 'meow' is not one of"):
        BarkSegment(start_s=0.0, end_s=1.0, sed_class="meow")


def test_confidence_below_zero_raises() -> None:
    with pytest.raises(ValueError, match="confidence -0.1 is not between 0 and 1"):
        BarkSegment(start_s=0.0, end_s=1.0, confidence=-0.1)


def test_confidence_above_one_raises() -> None:
    with pytest.raises(ValueError, match="confidence 1.5 is not between 0 and 1"):
        BarkSegment(start_s=0.0, end_s=1.0, confidence=1.5)
