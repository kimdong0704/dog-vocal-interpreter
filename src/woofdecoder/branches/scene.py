"""Frozen YOLOE objects and RF-DETR dogs for multimodal fusion."""

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from woofdecoder.config import REPO_ROOT
from woofdecoder.preprocess.dog_detector import detect_dogs, load_dog_detector
from woofdecoder.preprocess.segments import BarkSegment

DEFAULT_CHECKPOINT = REPO_ROOT / "models/scene/yoloe-11s-seg-pf.pt"


def box_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    first_area = max(first[2] - first[0], 0) * max(first[3] - first[1], 0)
    second_area = max(second[2] - second[0], 0) * max(second[3] - second[1], 0)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def merge_dog_detections(
    yoloe_objects, dogs, frame_index, frame_shape, dog_class_id, overlap_iou=0.5
):
    """Prefer RF-DETR's dog label when both detectors describe the same box."""
    if not 0 < overlap_iou <= 1:
        raise ValueError("overlap_iou must be between 0 and 1")
    if not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    if not isinstance(dog_class_id, int) or dog_class_id < 0:
        raise ValueError("dog_class_id must be a nonnegative integer")
    height, width = frame_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Frame dimensions must be positive")
    dog_objects = []
    for box, confidence in dogs:
        values = np.asarray(box, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Dog boxes must be finite pixel xywh coordinates")
        if not np.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("Dog confidence must be between 0 and 1")
        x, y, box_width, box_height = values
        x1, x2 = np.clip([x, x + box_width], 0, width) / width
        y1, y2 = np.clip([y, y + box_height], 0, height) / height
        if x2 <= x1 or y2 <= y1:
            continue
        dog_objects.append(
            {
                "frame": frame_index,
                "class_id": dog_class_id,
                "name": "dog",
                "confidence": float(confidence),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "source": "rfdetr",
            }
        )
    kept, suppressed = [], []
    for obj in yoloe_objects:
        overlap = max((box_iou(obj["box"], dog["box"]) for dog in dog_objects), default=0.0)
        if obj["frame"] == frame_index and overlap >= overlap_iou:
            suppressed.append({**obj, "matched_dog_iou": overlap})
        else:
            kept.append(obj)
    return kept + dog_objects, suppressed


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
    name = "yoloe_rfdetr_object_pool_v2"

    def __init__(
        self,
        checkpoint=DEFAULT_CHECKPOINT,
        device=None,
        confidence=0.25,
        image_size=640,
        max_objects=32,
        dog_detector=None,
        dog_overlap_iou=0.5,
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
        if not 0 < dog_overlap_iou <= 1:
            raise ValueError("dog_overlap_iou must be between 0 and 1")
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
        if "dog" not in self.names:
            raise ValueError("The YOLOE vocabulary must include the canonical dog class")
        self.dog_class_id = self.names.index("dog")
        self.dog_detector = (
            dog_detector
            if dog_detector is not None
            else load_dog_detector(verbose=False, device=device)
        )
        self.dog_overlap_iou = dog_overlap_iou
        try:
            dog_version = version("rfdetr")
        except PackageNotFoundError:
            dog_version = None
        dog_schema = {
            "model": type(self.dog_detector.model).__name__,
            "package_version": dog_version,
            "confidence": self.dog_detector.score_threshold,
            "overlap_iou": dog_overlap_iou,
            "merge_policy": "prefer_dog_for_high_iou_boxes_v1",
        }
        dog_config = getattr(self.dog_detector.model, "model_config", None)
        dog_checkpoint = getattr(dog_config, "pretrain_weights", None)
        if dog_checkpoint and Path(dog_checkpoint).is_file():
            dog_path = Path(dog_checkpoint)
            dog_schema["checkpoint"] = dog_path.name
            dog_schema["checkpoint_sha256"] = hashlib.sha256(dog_path.read_bytes()).hexdigest()
        self.schema = {
            "encoder": self.name,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "vocabulary": list(self.names),
            "feature_names": list(self.pool.feature_names),
            "layout": "class_major",
            "confidence": confidence,
            "image_size": image_size,
            "max_yoloe_objects_per_frame": max_objects,
            "dog_detector": dog_schema,
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
        detections, yoloe_detections, dog_detections, suppressed = [], [], [], []
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
            frame_objects = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                detection = {
                    "frame": frame_index,
                    "class_id": class_id,
                    "name": self.names[class_id],
                    "confidence": float(box.conf.item()),
                    "box": box.xyxyn[0].cpu().tolist(),
                    "source": "yoloe",
                }
                frame_objects.append(detection)
            # Run one frame at a time; keep every dog returned by preprocessing's detector.
            dogs = detect_dogs(self.dog_detector, [image.copy()])[0]
            combined, replaced = merge_dog_detections(
                frame_objects,
                dogs,
                frame_index,
                image.shape,
                self.dog_class_id,
                self.dog_overlap_iou,
            )
            if frame_times_s is not None:
                for detection in frame_objects + combined + replaced:
                    detection["time_s"] = float(times[frame_index])
            detections.extend(combined)
            yoloe_detections.extend(frame_objects)
            dog_detections.extend(obj for obj in combined if obj["source"] == "rfdetr")
            suppressed.extend(replaced)
        return {
            "embedding": self.pool.encode(detections, len(frames)),
            "detected_objects": detections,
            "yoloe_objects": yoloe_detections,
            "dog_objects": dog_detections,
            "suppressed_objects": suppressed,
            "present": len(frames) > 0,
            "has_detections": bool(detections),
            "frame_count": len(frames),
            "schema_id": self.schema_id,
        }

    def embed_frames(self, frames):
        return self.encode_frames(frames)["embedding"]

    def encode_segment(self, segment: BarkSegment):
        """Return scene features and the source interval used by preprocessing."""
        times = np.asarray(segment.frame_times_s, dtype=float)
        if not np.isfinite([segment.start_s, segment.end_s]).all():
            raise ValueError("Segment boundaries must be finite")
        if times.size and (
            (times < segment.start_s - 1e-3).any() or (times > segment.end_s + 1e-3).any()
        ):
            raise ValueError("Frame timestamps must fall inside the segment")
        result = self.encode_frames(segment.frames, segment.frame_times_s)
        result.update(
            source_id=segment.source_id,
            start_s=float(segment.start_s),
            end_s=float(segment.end_s),
            frame_times_s=times.tolist(),
        )
        return result

    def encode_segments(self, segments):
        return [self.encode_segment(segment) for segment in segments]

    def embed(self, segment: BarkSegment) -> np.ndarray | None:
        if not segment.has_video:
            return None
        return self.encode_segment(segment)["embedding"]


def encoder_from_config(config: dict, device: str | None = None, dog_detector=None) -> SceneEncoder:
    """Load the scene encoder using branches.scene settings."""
    section = config["branches"]["scene"]
    if section.get("encoder") != SceneEncoder.name:
        raise ValueError(f"Unsupported scene encoder: {section.get('encoder')!r}")
    if not section.get("model"):
        raise ValueError("Set branches.scene.model in configs/pipeline.yaml")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dog_detector is None:
        dog_section = config["dog_detector"]
        dog_detector = load_dog_detector(
            model_size=dog_section["model"],
            score_threshold=float(dog_section["score_threshold"]),
            verbose=False,
            device=device,
        )
    return SceneEncoder(
        checkpoint=section["model"],
        device=device,
        confidence=float(section.get("confidence", 0.25)),
        image_size=int(section.get("image_size", 640)),
        max_objects=int(section.get("max_objects", 32)),
        dog_detector=dog_detector,
        dog_overlap_iou=float(section.get("dog_overlap_iou", 0.5)),
    )
