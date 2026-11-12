"""SegmentResult -> plain JSON for the web page (numpy values, frame thumbnails)."""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
from PIL import Image

from woofdecoder.base_branch import BranchOutput
from woofdecoder.constants import BRANCH_NAMES
from woofdecoder.pipeline import SegmentResult
from woofdecoder.taxonomy import CATEGORIES

THUMBNAIL_MAX_WIDTH: int = 320  # pixels
THUMBNAIL_JPEG_QUALITY: int = 70


def to_json(value: Any) -> Any:
    """Turn numpy arrays/scalars, tuples and nested containers into plain JSON values."""
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def thumbnail(frame: np.ndarray) -> str:
    """RGB frame -> base64 JPEG, at most THUMBNAIL_MAX_WIDTH wide."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if image.width > THUMBNAIL_MAX_WIDTH:
        height = max(1, round(image.height * THUMBNAIL_MAX_WIDTH / image.width))
        image = image.resize((THUMBNAIL_MAX_WIDTH, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def output_json(output: BranchOutput) -> dict[str, Any]:
    return {
        "probs": output.probs,
        "present": bool(output.present),
        "top_category": CATEGORIES[int(output.probs.argmax())],
        "extras": output.extras,
    }


def body_reason(result: SegmentResult) -> str | None:
    """Why the body branch was not used, in plain words (None if it was used)."""
    if result.branches["body"].present:
        return None
    if not result.segment.has_video:
        return "no video in this clip"
    return "no frame showed exactly one dog"


def segment_json(number: int, result: SegmentResult) -> dict[str, Any]:
    segment = result.segment
    frames_used = result.branches["body"].extras.get("frames_used") or []
    dogs_per_frame = result.branches["body"].extras.get("dogs_per_frame") or []
    frames = []
    for index, (frame, time_s) in enumerate(zip(segment.frames, segment.frame_times_s)):
        frames.append(
            {
                "time_s": time_s,
                "jpeg": thumbnail(frame),
                "body_used": bool(frames_used[index]) if index < len(frames_used) else False,
                "dogs": dogs_per_frame[index] if index < len(dogs_per_frame) else None,
            }
        )

    data = {
        "number": number,
        "start_s": segment.start_s,
        "end_s": segment.end_s,
        "confidence": segment.confidence,
        "sound_sections": [
            {"label": label, "start_s": start, "end_s": end, "score": score}
            for label, (start, end), score in segment.sound_sections
        ],
        "num_frames": len(segment.frames),
        "has_video": segment.has_video,
        "frames": frames,
        "branches": {name: output_json(result.branches[name]) for name in BRANCH_NAMES},
        "body_reason": body_reason(result),
        "scene": result.scene,
        "fused": output_json(result.fused),
        "fusion": result.fusion,
        "category": result.category,
        "probability": result.probability,
    }
    return to_json(data)
