"""Export aligned scene features; reading caches does not load CLIP."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from woofdecoder.taxonomy import CATEGORIES

CACHE_FORMAT_VERSION = 3
SEGMENT_FIELDS = ("source_id", "start_s", "end_s", "frame_times_s")
RESERVED_FIELDS = {"clip_id", "present", "probs", "frame_count"}


def schema_identity(schema):
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def check_ids(clip_ids):
    clip_ids = list(clip_ids)
    if not clip_ids or any(not isinstance(value, str) or not value.strip() for value in clip_ids):
        raise ValueError("Segment IDs must be nonempty strings")
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Segment IDs must be unique")
    return clip_ids


def schema_dimension(schema):
    dimension = schema.get("embedding_dim")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Invalid scene embedding dimension")
    if schema.get("categories") != list(CATEGORIES):
        raise ValueError("Cached categories differ from the taxonomy")
    return dimension


def check_metadata(metadata, frame_count):
    if metadata.get("mapped_category") not in (None, "", *CATEGORIES):
        raise ValueError("mapped_category must use the shared taxonomy or be unannotated")
    if metadata.get("split") not in (None, "", "train", "val", "test", "train_only"):
        raise ValueError("Unknown dataset split")
    start, end = metadata.get("start_s"), metadata.get("end_s")
    if (start is None) != (end is None):
        raise ValueError("Supply both segment boundaries or neither")
    if start is not None and (not np.isfinite([start, end]).all() or end < start):
        raise ValueError("Invalid segment boundaries")
    times = metadata.get("frame_times_s")
    if times is not None:
        times = np.asarray(times, dtype=float)
        if (
            times.shape != (frame_count,)
            or not np.isfinite(times).all()
            or (np.diff(times) < 0).any()
        ):
            raise ValueError("Invalid frame timestamps")
        if start is not None and ((times < start - 1e-3).any() or (times > end + 1e-3).any()):
            raise ValueError("Frame timestamps lie outside their segment")


def check_result(result, embedding):
    count = result["frame_count"]
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("frame_count must be a nonnegative integer")
    if not isinstance(result["present"], (bool, np.bool_)):
        raise TypeError("present must be a boolean")
    if result["present"] != (count > 0):
        raise ValueError("Availability mask disagrees with frame_count")
    probs = np.asarray(result["probs"])
    if (
        probs.shape != (len(CATEGORIES),)
        or not np.isfinite(probs).all()
        or (probs < 0).any()
        or not np.isclose(probs.sum(), 1, atol=1e-5)
    ):
        raise ValueError("Invalid segment probabilities")
    if not result["present"] and (np.any(embedding) or not np.allclose(probs, 1 / len(CATEGORIES))):
        raise ValueError("Missing video needs zero features and uniform placeholder scores")


def save_scene_cache(path, clip_ids, results, encoder, metadata=None):
    """Save one row per shared segment ID, including optional genuine annotations."""
    clip_ids = check_ids(clip_ids)
    results = list(results)
    if len(clip_ids) != len(results):
        raise ValueError("Supply one segment ID per encoder result")
    dimension = schema_dimension(encoder.schema)
    if dimension != encoder.dim or schema_identity(encoder.schema) != encoder.schema_id:
        raise ValueError("Encoder schema identity or dimension is inconsistent")
    if any(result["schema_id"] != encoder.schema_id for result in results):
        raise ValueError("Encoder result schema differs from the cache schema")
    embeddings = np.stack([result["embedding"] for result in results]).astype(np.float32)
    if embeddings.shape != (len(clip_ids), dimension) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid scene embedding shape or values")
    metadata = [{} for _ in results] if metadata is None else list(metadata)
    if len(metadata) != len(results):
        raise ValueError("Supply one metadata record per segment")
    records = []
    for result, embedding, extra in zip(results, embeddings, metadata, strict=True):
        check_result(result, embedding)
        if not isinstance(extra, dict) or RESERVED_FIELDS.intersection(extra):
            raise ValueError("Metadata cannot override segment IDs or availability fields")
        record = {key: result.get(key) for key in SEGMENT_FIELDS}
        for key, value in extra.items():
            if key in SEGMENT_FIELDS and record[key] is not None and record[key] != value:
                raise ValueError(f"Metadata disagrees with encoded segment {key}")
            record[key] = value
        check_metadata(record, result["frame_count"])
        records.append(record)
    metadata_json = np.array([json.dumps(record, allow_nan=False) for record in records], dtype=str)
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Scene cache path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream,
                format_version=np.array(CACHE_FORMAT_VERSION),
                clip_id=np.array(clip_ids, dtype=str),
                embedding=embeddings,
                present=np.array([result["present"] for result in results], dtype=bool),
                probs=np.array([result["probs"] for result in results], dtype=np.float32),
                frame_count=np.array([result["frame_count"] for result in results], dtype=np.int32),
                schema_json=np.array(json.dumps(encoder.schema, sort_keys=True)),
                schema_id=np.array(encoder.schema_id),
                metadata_json=metadata_json,
            )
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def cache_scene_segments(path, clip_ids, segments, encoder, metadata=None):
    """Encode the preprocessing team's segments without decoding videos again."""
    clip_ids = check_ids(clip_ids)
    results = [
        encoder.encode_segment(segment) for _, segment in zip(clip_ids, segments, strict=True)
    ]
    return save_scene_cache(path, clip_ids, results, encoder, metadata)


def load_scene_cache(path, encoder=None, clip_ids=None, expected_schema_id=None):
    """Read features without model weights; optionally align them to another modality's IDs."""
    with np.load(path, allow_pickle=False) as cached:
        if (
            "format_version" not in cached
            or cached["format_version"].item() != CACHE_FORMAT_VERSION
        ):
            raise ValueError("Scene cache format is outdated; rebuild the cache")
        schema = json.loads(cached["schema_json"].item())
        if encoder is not None and schema != encoder.schema:
            raise ValueError("Cached CLIP model, prompts, or preprocessing settings differ")
        schema_id = schema_identity(schema)
        if cached["schema_id"].item() != schema_id:
            raise ValueError("Cached scene schema hash is invalid")
        if expected_schema_id is not None and expected_schema_id != schema_id:
            raise ValueError("Cached scene schema does not match the expected schema ID")
        ids = check_ids(cached["clip_id"].tolist())
        embeddings = cached["embedding"].astype(np.float32)
        if (
            embeddings.shape != (len(ids), schema_dimension(schema))
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("Cached scene embedding shape or values are invalid")
        columns = {key: cached[key] for key in ("present", "probs", "frame_count")}
        if any(
            values.shape != ((len(ids), len(CATEGORIES)) if key == "probs" else (len(ids),))
            for key, values in columns.items()
        ):
            raise ValueError("Cached availability columns have invalid shapes")
        if cached["metadata_json"].shape != (len(ids),):
            raise ValueError("Cached metadata has invalid shape")
        records = []
        for i, clip_id in enumerate(ids):
            metadata = json.loads(cached["metadata_json"][i])
            if not isinstance(metadata, dict) or RESERVED_FIELDS.intersection(metadata):
                raise ValueError("Cached metadata overrides segment identifiers or masks")
            result = {key: values[i] for key, values in columns.items()}
            check_result(result, embeddings[i])
            check_metadata(metadata, result["frame_count"])
            records.append(
                {
                    "clip_id": clip_id,
                    **metadata,
                    **{key: values[i] for key, values in columns.items()},
                }
            )
    table = pd.DataFrame(records)
    if clip_ids is not None:
        wanted = check_ids(clip_ids)
        index = pd.Index(ids).get_indexer(wanted)
        if (index < 0).any():
            raise ValueError("Requested segment IDs are missing from the scene cache")
        embeddings = embeddings[index]
        table = table.iloc[index].reset_index(drop=True)
    table.attrs["scene_schema"] = schema
    table.attrs["scene_schema_id"] = schema_id
    return embeddings, table
