"""Save scene features and their vocabulary for later fusion training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_scene_cache(path, clip_ids, results, encoder):
    if not results or len(clip_ids) != len(results) or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Supply one unique segment ID per encoder result")
    if any(result["schema_id"] != encoder.schema_id for result in results):
        raise ValueError("Encoder result schema differs from the cache schema")
    embeddings = np.stack([result["embedding"] for result in results]).astype(np.float32)
    if embeddings.shape != (len(clip_ids), encoder.dim) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid scene embedding shape or values")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        clip_id=np.array(clip_ids, dtype=str),
        embedding=embeddings,
        present=np.array([result["present"] for result in results], dtype=bool),
        has_detections=np.array([result["has_detections"] for result in results], dtype=bool),
        frame_count=np.array([result["frame_count"] for result in results], dtype=np.int32),
        schema_json=np.array(json.dumps(encoder.schema, sort_keys=True)),
        detections_json=np.array([json.dumps(result["detected_objects"]) for result in results]),
    )


def load_scene_cache(path, encoder):
    with np.load(path, allow_pickle=False) as cached:
        if json.loads(cached["schema_json"].item()) != encoder.schema:
            raise ValueError("Cached vocabulary, feature order, or detector settings differ")
        embeddings = cached["embedding"].astype(np.float32)
        if embeddings.shape != (len(cached["clip_id"]), encoder.dim):
            raise ValueError("Cached scene embedding shape is invalid")
        if not np.isfinite(embeddings).all():
            raise ValueError("Cached scene features are not finite")
        table = pd.DataFrame(
            {
                "clip_id": cached["clip_id"],
                "present": cached["present"],
                "has_detections": cached["has_detections"],
                "frame_count": cached["frame_count"],
            }
        )
    return embeddings, table
