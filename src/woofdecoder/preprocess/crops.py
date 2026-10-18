"""Crop an image to a padded box, moving its keypoints along."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

RESIZE_FILTER = Image.Resampling.BILINEAR


def padded_bounds(box: np.ndarray | list[float], pad: float) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) in whole pixels of box (x, y, w, h) grown by pad
    around its centre. The bounds may lie outside the image.

    Raises:
        ValueError: If the box has no area.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError(f"box {list(box)} has no area")
    centre_x = x + w / 2
    centre_y = y + h / 2
    left = math.floor(centre_x - w * pad / 2)
    top = math.floor(centre_y - h * pad / 2)
    right = math.ceil(centre_x + w * pad / 2)
    bottom = math.ceil(centre_y + h * pad / 2)
    return left, top, right, bottom


def cut_region(image: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    """Return image[top:bottom, left:right], with zeros where the region leaves the image."""
    height, width = image.shape[:2]
    region = np.zeros((bottom - top, right - left) + image.shape[2:], dtype=image.dtype)
    inside_left = max(left, 0)
    inside_top = max(top, 0)
    inside_right = min(right, width)
    inside_bottom = min(bottom, height)
    if inside_right > inside_left and inside_bottom > inside_top:
        region[inside_top - top : inside_bottom - top, inside_left - left : inside_right - left] = (
            image[inside_top:inside_bottom, inside_left:inside_right]
        )
    return region


def crop_to_box(
    image: np.ndarray,
    box: np.ndarray | list[float],
    pad: float = 1.25,
    out_size: tuple[int, int] | None = None,
    keypoints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Crop image to box (x, y, w, h) grown by pad around its centre (zeros outside).

    Args:
        image: shape (height, width, 3).
        box: x, y, w, h in pixels.
        pad: 1.25 makes the crop 25% wider and taller than the box.
        out_size: (width, height) to resize the crop to; None keeps its size.
        keypoints: shape (num_keypoints, 3) as x, y, visibility; only points with
            visibility > 0 are moved (the others are left as they are).

    Returns:
        (crop, keypoints in crop coordinates, or None if none were given).
    """
    left, top, right, bottom = padded_bounds(box, pad)
    crop = cut_region(image, left, top, right, bottom)  # shape (crop_h, crop_w, 3)
    scale_x = 1.0
    scale_y = 1.0
    if out_size is not None:
        scale_x = out_size[0] / crop.shape[1]
        scale_y = out_size[1] / crop.shape[0]
        crop = np.asarray(Image.fromarray(crop).resize(out_size, RESIZE_FILTER))

    if keypoints is None:
        return crop, None
    moved = np.array(keypoints, dtype=np.float32)  # a copy, shape (num_keypoints, 3)
    labelled = moved[:, 2] > 0  # shape (num_keypoints,)
    moved[labelled, 0] = (moved[labelled, 0] - left) * scale_x
    moved[labelled, 1] = (moved[labelled, 1] - top) * scale_y
    return crop, moved
