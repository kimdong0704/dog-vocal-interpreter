"""Frozen CLIP: full RGB frames -> one five-category distribution per segment."""

import hashlib
import json

import numpy as np
import torch
from PIL import Image

from woofdecoder.base_branch import Branch, BranchOutput
from woofdecoder.taxonomy import CATEGORIES, NUM_CATEGORIES

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
# Exploratory descriptions in taxonomy order; not validated behavioral labels.
DEFAULT_PROMPTS = dict(
    zip(
        CATEGORIES,
        (
            "Context where a dog is showing alert or aggressive behavior.",
            "Context where a dog is showing playful or excited behavior.",
            "Context where a dog is showing distress or loneliness.",
            "Context where a dog is seeking attention from a person.",
            "Context where a dog is in another everyday situation.",
        ),
        strict=True,
    )
)


def average_frame_probs(frame_probs):
    """Average distributions, retaining uncertainty rather than voting on winners."""
    values = np.asarray(frame_probs, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != NUM_CATEGORIES:
        raise ValueError("Expected frame probabilities with shape (frames, 5)")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Frame probabilities must be finite and nonnegative")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Each frame distribution must sum to one")
    if len(values) == 0:
        return np.full(NUM_CATEGORIES, 1 / NUM_CATEGORIES, dtype=np.float32)
    return values.mean(axis=0)


class SceneEncoder(Branch):
    """predict(segment) follows audio/body's BranchOutput contract."""

    name = "clip_scene_v1"

    def __init__(
        self,
        model=DEFAULT_MODEL,
        device=None,
        batch_size=16,
        temperature=1.0,
        revision=None,
        prompts=None,
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        prompts = dict(DEFAULT_PROMPTS if prompts is None else prompts)
        if set(prompts) != set(CATEGORIES) or any(
            not isinstance(p, str) or not p.strip() for p in prompts.values()
        ):
            raise ValueError("Supply one nonempty prompt for every taxonomy category")
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.temperature = float(temperature)
        self.prompts = [prompts[category] for category in CATEGORIES]
        self.model = CLIPModel.from_pretrained(model, revision=revision).to(self.device).eval()
        self.model.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(model, revision=revision)
        self.dim = self.model.config.projection_dim
        self.text_inputs = self.processor(text=self.prompts, return_tensors="pt", padding=True).to(
            self.device
        )
        self.schema = {
            "encoder": self.name,
            "model": model,
            "revision": getattr(self.model.config, "_commit_hash", None) or revision,
            "embedding_dim": self.dim,
            "categories": list(CATEGORIES),
            "prompts": self.prompts,
            "temperature": self.temperature,
            "aggregation": "mean_frame_probabilities",
            "embedding_pooling": "normalized_mean_of_normalized_frame_embeddings",
            "image_processor": self.processor.image_processor.to_dict(),
        }
        self.schema_id = hashlib.sha256(
            json.dumps(self.schema, sort_keys=True).encode()
        ).hexdigest()

    @torch.inference_mode()
    def encode_frames(self, frames, frame_times_s=None):
        frames = list(frames)
        if frame_times_s is not None:
            times = np.asarray(frame_times_s, dtype=float)
            if (
                times.shape != (len(frames),)
                or not np.isfinite(times).all()
                or (np.diff(times) < 0).any()
            ):
                raise ValueError("Supply one chronological, finite timestamp per frame")
        images = []
        for frame in frames:
            frame = np.asarray(frame)
            if (
                frame.ndim != 3
                or frame.shape[-1] != 3
                or frame.dtype != np.uint8
                or min(frame.shape[:2]) == 0
            ):
                raise ValueError("Frames must be nonempty uint8 RGB images")
            images.append(Image.fromarray(frame))
        probabilities, embeddings = [], []
        self.model.eval()
        for start in range(0, len(images), self.batch_size):
            inputs = self.processor(
                images=images[start : start + self.batch_size], return_tensors="pt"
            ).to(self.device)
            output = self.model(**inputs, **self.text_inputs)
            probabilities.append(
                (output.logits_per_image / self.temperature).softmax(dim=-1).cpu().numpy()
            )
            embeddings.append(output.image_embeds.cpu().numpy())
        frame_probs = (
            np.concatenate(probabilities).astype(np.float32)
            if probabilities
            else np.empty((0, NUM_CATEGORIES), dtype=np.float32)
        )
        embedding = np.zeros(self.dim, dtype=np.float32)
        if embeddings:
            vectors = np.concatenate(embeddings).astype(np.float32)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            embedding = vectors.mean(axis=0)
            embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        return {
            "embedding": embedding,
            "probs": average_frame_probs(frame_probs),
            "frame_probs": frame_probs,
            "present": bool(frames),
            "frame_count": len(frames),
            "schema_id": self.schema_id,
        }

    def encode_segment(self, segment):
        times = np.asarray(segment.frame_times_s, dtype=float)
        if (
            not np.isfinite([segment.start_s, segment.end_s]).all()
            or segment.start_s < 0
            or segment.end_s < segment.start_s
        ):
            raise ValueError("Invalid segment boundaries")
        if (times < segment.start_s - 1e-3).any() or (times > segment.end_s + 1e-3).any():
            raise ValueError("Frame timestamps must fall inside the segment")
        result = self.encode_frames(segment.frames, segment.frame_times_s)
        result.update(
            source_id=segment.source_id,
            start_s=float(segment.start_s),
            end_s=float(segment.end_s),
            frame_times_s=times.tolist(),
        )
        return result

    def predict(self, segment):
        result = self.encode_segment(segment)
        return BranchOutput(
            probs=result["probs"],
            present=result["present"],
            extras={
                "frame_probs": result["frame_probs"].tolist(),
                "frame_times_s": result["frame_times_s"],
                "frame_count": result["frame_count"],
                "aggregation": "mean_frame_probabilities",
                "calibrated": False,
            },
        )

    def embed(self, segment):
        return self.encode_segment(segment)["embedding"] if segment.has_video else None

    def embed_frames(self, frames):
        return self.encode_frames(frames)["embedding"]

    def encode_segments(self, segments):
        return [self.encode_segment(segment) for segment in segments]

    @classmethod
    def from_config(cls, config=None, device=None):
        from woofdecoder.config import load_config

        return encoder_from_config(load_config() if config is None else config, device)


def encoder_from_config(config, device=None):
    section = config["branches"]["scene"]
    if section.get("encoder") != SceneEncoder.name:
        raise ValueError(f"Unsupported scene encoder: {section.get('encoder')!r}")
    return SceneEncoder(
        model=section["model"],
        device=device,
        batch_size=section.get("batch_size", 16),
        temperature=section.get("temperature", 1.0),
        revision=section.get("revision"),
        prompts=section.get("prompts"),
    )
