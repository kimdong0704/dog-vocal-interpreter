"""Frozen object-based scene features for multimodal fusion."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from woofdecoder.config import REPO_ROOT
from woofdecoder.preprocess.segments import BarkSegment

DEFAULT_CHECKPOINT = REPO_ROOT / "models/scene/yoloe-11s-seg-pf.pt"


class ObjectFeaturePool:
    feature_names = (
        "frame_fraction",
        "mean_count",
        "max_confidence",
        "mean_center_x",
        "mean_center_y",
        "mean_width",
        "mean_height",
        "mean_area",
        "delta_center_x",
        "delta_center_y",
    )

    def __init__(self, names):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Use a nonempty, unique detector vocabulary")
        self.dim = len(self.names) * len(self.feature_names)

    def encode(self, detections, frame_count):
        if not isinstance(frame_count, int) or frame_count < 0:
            raise ValueError("frame_count must be a nonnegative integer")
        counts = np.zeros((frame_count, len(self.names)), dtype=np.float32)
        scores = np.zeros_like(counts)
        weight_sums = np.zeros_like(counts)
        geometry = np.zeros((frame_count, len(self.names), 5), dtype=np.float32)

        for obj in detections:
            frame, class_id = obj["frame"], obj["class_id"]
            if not isinstance(frame, int) or not 0 <= frame < frame_count:
                raise ValueError("Detection frame is outside this segment")
            if not isinstance(class_id, int) or not 0 <= class_id < len(self.names):
                raise ValueError("Detection class ID is outside the saved vocabulary")
            if obj["name"] != self.names[class_id]:
                raise ValueError("Detection name and class ID disagree with the vocabulary")
            x1, y1, x2, y2 = obj["box"]
            confidence = obj["confidence"]
            values = np.array([x1, y1, x2, y2, confidence])
            if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError("Boxes and confidence must be normalized between 0 and 1")
            if x2 <= x1 or y2 <= y1 or confidence <= 0:
                raise ValueError("Detections need positive box area and confidence")

            width, height = x2 - x1, y2 - y1
            box_features = np.array(
                [
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    width,
                    height,
                    width * height,
                ]
            )
            counts[frame, class_id] += 1
            scores[frame, class_id] = max(scores[frame, class_id], confidence)
            geometry[frame, class_id] += confidence * box_features
            weight_sums[frame, class_id] += confidence

        pooled = np.zeros((len(self.names), len(self.feature_names)), dtype=np.float32)
        if frame_count == 0:
            return pooled.reshape(-1)

        geometry /= np.maximum(weight_sums[:, :, None], np.finfo(np.float32).tiny)
        pooled[:, 0] = (counts > 0).mean(axis=0)
        pooled[:, 1] = counts.mean(axis=0)
        pooled[:, 2] = scores.max(axis=0)
        for class_id in np.flatnonzero(counts.sum(axis=0)):
            observed = np.flatnonzero(counts[:, class_id])
            positions = geometry[observed, class_id]
            pooled[class_id, 3:8] = positions.mean(axis=0)
            pooled[class_id, 8:10] = positions[-1, :2] - positions[0, :2]
        return pooled.reshape(-1)


class SceneEncoder:
    name = "yoloe_object_pool_v1"

    def __init__(
        self,
        checkpoint=DEFAULT_CHECKPOINT,
        device=None,
        confidence=0.25,
        image_size=640,
        max_objects=32,
    ):
        from ultralytics import YOLOE

        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not checkpoint.name.endswith("-seg-pf.pt"):
            raise ValueError("Use a prompt-free YOLOE checkpoint ending in -seg-pf.pt")
        if not 0 < confidence <= 1 or image_size < 1 or max_objects < 1:
            raise ValueError("Confidence, image size, and detection limit must be positive")
        self.device = device
        self.confidence = confidence
        self.image_size = image_size
        self.max_objects = max_objects
        self.model = YOLOE(str(checkpoint))
        self.model.model.eval()
        for parameter in self.model.model.parameters():
            parameter.requires_grad_(False)
        names = self.model.names
        self.pool = ObjectFeaturePool([names[i] for i in range(len(names))])
        self.names = self.pool.names
        self.dim = self.pool.dim
        self.schema = {
            "encoder": self.name,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "vocabulary": list(self.names),
            "feature_names": list(self.pool.feature_names),
            "layout": "class_major",
            "confidence": confidence,
            "image_size": image_size,
            "max_objects_per_frame": max_objects,
        }
        schema_text = json.dumps(self.schema, sort_keys=True)
        self.schema_id = hashlib.sha256(schema_text.encode()).hexdigest()

    @torch.inference_mode()
    def encode_frames(self, frames, frame_times_s=None):
        frames = list(frames)
        if frame_times_s is not None:
            times = np.asarray(frame_times_s, dtype=float)
            if times.shape != (len(frames),) or not np.isfinite(times).all():
                raise ValueError("Supply one finite timestamp per frame")
            if (np.diff(times) < 0).any():
                raise ValueError("Frame timestamps must be in chronological order")
        detections = []
        self.model.model.eval()
        for frame_index, frame in enumerate(frames):
            image = np.asarray(frame)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("Frames must be uint8 RGB images")
            result = self.model.predict(
                source=Image.fromarray(image),
                device=self.device,
                conf=self.confidence,
                imgsz=self.image_size,
                max_det=self.max_objects,
                verbose=False,
            )[0]
            names = tuple(result.names[i] for i in range(len(result.names)))
            if names != self.names:
                raise ValueError("Detector vocabulary changed")
            for box in result.boxes:
                class_id = int(box.cls.item())
                detection = {
                    "frame": frame_index,
                    "class_id": class_id,
                    "name": self.names[class_id],
                    "confidence": float(box.conf.item()),
                    "box": box.xyxyn[0].cpu().tolist(),
                }
                if frame_times_s is not None:
                    detection["time_s"] = float(times[frame_index])
                detections.append(detection)
        return {
            "embedding": self.pool.encode(detections, len(frames)),
            "detected_objects": detections,
            "present": len(frames) > 0,
            "has_detections": bool(detections),
            "frame_count": len(frames),
            "schema_id": self.schema_id,
        }

    def embed_frames(self, frames):
        return self.encode_frames(frames)["embedding"]

    def embed(self, segment: BarkSegment) -> np.ndarray | None:
        if not segment.has_video:
            return None
        return self.embed_frames(segment.frames)


def encoder_from_config(config: dict, device: str | None = None) -> SceneEncoder:
    """Load the scene encoder using branches.scene settings."""
    section = config["branches"]["scene"]
    if section.get("encoder") != SceneEncoder.name:
        raise ValueError(f"Unsupported scene encoder: {section.get('encoder')!r}")
    if not section.get("model"):
        raise ValueError("Set branches.scene.model in configs/pipeline.yaml")
    return SceneEncoder(
        checkpoint=section["model"],
        device=device,
        confidence=float(section.get("confidence", 0.25)),
        image_size=int(section.get("image_size", 640)),
        max_objects=int(section.get("max_objects", 32)),
    )
