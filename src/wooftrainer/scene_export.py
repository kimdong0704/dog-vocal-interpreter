"""Export scene features from a manifest of already-preprocessed RGB frames."""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from woofdecoder.preprocess.segments import TIME_TOLERANCE_S, BarkSegment
from woofdecoder.taxonomy import CATEGORIES

SEGMENT_COLUMNS = {
    "clip_id",
    "source_id",
    "start_s",
    "end_s",
    "frames",
    "frame_times_s",
}


def read_manifest(path):
    path = Path(path).expanduser().resolve()
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("The scene manifest must contain a nonempty JSON list")
    seen = set()
    prepared = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not SEGMENT_COLUMNS.issubset(record):
            raise ValueError(f"Record {index} needs {sorted(SEGMENT_COLUMNS)}")
        record = dict(record)
        for column in ("clip_id", "source_id"):
            if not isinstance(record[column], str) or not record[column].strip():
                raise ValueError(f"Record {index} needs a nonempty {column}")
        if record["clip_id"] in seen:
            raise ValueError(f"Duplicate clip_id: {record['clip_id']}")
        seen.add(record["clip_id"])
        for column in ("start_s", "end_s"):
            value = record[column]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{record['clip_id']}: {column} must be finite seconds")
            record[column] = float(value)
        if record["start_s"] < 0 or record["end_s"] < record["start_s"]:
            raise ValueError(f"{record['clip_id']}: invalid segment interval")
        if not isinstance(record["frames"], list) or not isinstance(record["frame_times_s"], list):
            raise TypeError(f"{record['clip_id']}: frames and frame_times_s must be lists")
        if len(record["frames"]) != len(record["frame_times_s"]):
            raise ValueError(f"{record['clip_id']}: supply one timestamp per frame")
        times = record["frame_times_s"]
        if any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
            for t in times
        ):
            raise ValueError(f"{record['clip_id']}: timestamps must be finite seconds")
        if any(first > second for first, second in pairwise(times)):
            raise ValueError(f"{record['clip_id']}: frames must be chronological")
        if any(
            t < record["start_s"] - TIME_TOLERANCE_S or t > record["end_s"] + TIME_TOLERANCE_S
            for t in times
        ):
            raise ValueError(f"{record['clip_id']}: frame timestamp is outside its segment")
        frame_paths = []
        for name in record["frames"]:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{record['clip_id']}: frame paths must be nonempty strings")
            frame_path = (path.parent / name).expanduser().resolve()
            if not frame_path.is_file():
                raise FileNotFoundError(f"{record['clip_id']}: missing frame {frame_path}")
            with Image.open(frame_path) as image:
                image.verify()
            frame_paths.append(frame_path)
        category = record.get("mapped_category")
        if category not in (None, "") and category not in CATEGORIES:
            raise ValueError(f"{record['clip_id']}: mapped_category must use the shared taxonomy")
        split = record.get("split")
        if split not in (None, "", "train", "val", "test", "train_only"):
            raise ValueError(f"{record['clip_id']}: unknown split {split!r}")
        record["frames"] = frame_paths
        prepared.append(record)
    return prepared


def load_segments(records):
    for record in records:
        frames = []
        for frame_path in record["frames"]:
            with Image.open(frame_path) as image:
                frames.append(np.array(image.convert("RGB"), copy=True))
        yield BarkSegment(
            start_s=record["start_s"],
            end_s=record["end_s"],
            source_id=record["source_id"],
            frames=frames,
            frame_times_s=record["frame_times_s"],
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="JSON segment manifest")
    parser.add_argument("--output", required=True, type=Path, help="Destination scene .npz cache")
    parser.add_argument("--device", help="Inference device, for example cpu or cuda")
    parser.add_argument("--config", type=Path, help="Pipeline YAML; defaults to the project config")
    args = parser.parse_args(argv)
    try:
        if args.output.suffix.lower() != ".npz":
            raise ValueError("Scene cache output must use the .npz suffix")
        records = read_manifest(args.manifest)
        from woofdecoder.branches.scene import encoder_from_config
        from woofdecoder.config import load_config
        from wooftrainer.trainers.scene_embeddings import cache_scene_segments

        encoder = encoder_from_config(load_config(args.config), device=args.device)
        metadata = [
            {key: value for key, value in record.items() if key not in SEGMENT_COLUMNS}
            for record in records
        ]
        output = cache_scene_segments(
            args.output.expanduser(),
            [record["clip_id"] for record in records],
            load_segments(records),
            encoder,
            metadata=metadata,
        )
    except (OSError, ValueError, TypeError, RuntimeError, ModuleNotFoundError) as error:
        parser.exit(1, f"Scene export failed: {error}\n")
    print(f"Saved {len(records)} scene vectors with {encoder.dim} features each to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
