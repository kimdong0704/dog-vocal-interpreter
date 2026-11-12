from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd

from woofdata.path import data_path, load_data_dir_from_env_file
from woofdata.prepare.manifest import processed_dir
from woofdata.tasks import read_manifest
from woofdata.utils_fileio import read_csv
from woofdecoder.audio import AudioBranch
from woofdecoder.base_branch import Branch, BranchOutput
from woofdecoder.body.branch import BodyBranch
from woofdecoder.config import load_config
from woofdecoder.preprocess.bark_detector import load_bark_detector, settings_from_config
from woofdecoder.preprocess.bark_segments import build_segments
from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.scene import SceneBranch
from woofdecoder.taxonomy import CATEGORIES

FUSION_PREFIX: str = "self_labeled/fusion/"
VIDEOS_DIR: str = "raw/self_labeled/fusion/videos"
CACHE_DIR: str = "cache/fusion"
SEGMENTS_FILENAME: str = "branch_outputs.csv"
VIDEOS_FILENAME: str = "videos.csv"
INFO_FILENAME: str = "info.json"
BRANCH_NAMES: tuple[str, ...] = ("audio", "body", "scene")


def fusion_videos() -> pd.DataFrame:
    load_data_dir_from_env_file()
    manifest = read_manifest()
    rows = manifest[manifest["clip_id"].str.startswith(FUSION_PREFIX)]
    labels = read_csv(processed_dir() / "self_labeled" / "labels.csv")
    videos = rows[["clip_id", "dog_id", "split", "mapped_category"]].merge(
        labels[["clip_id", "video_id", "video_start_s", "video_end_s"]],
        on="clip_id",
        validate="one_to_one",
    )
    videos = videos.rename(columns={"mapped_category": "category"})
    videos["video_start_s"] = pd.to_numeric(videos["video_start_s"])
    videos["video_end_s"] = pd.to_numeric(videos["video_end_s"])

    paths: list[Path] = []
    missing: list[str] = []
    for video_id in videos["video_id"]:
        path = data_path(f"{VIDEOS_DIR}/{video_id}.mp4")
        paths.append(path)
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("fusion videos not found:\n  " + "\n  ".join(missing))
    videos["video_path"] = pd.Series(paths, index=videos.index, dtype=object)
    return videos.sort_values("video_id").reset_index(drop=True)


def segments_in_span(
    segments: list[BarkSegment], start_s: float, end_s: float
) -> list[BarkSegment]:
    kept: list[BarkSegment] = []
    for segment in segments:
        middle_s = (segment.start_s + segment.end_s) / 2
        if start_s <= middle_s <= end_s:
            kept.append(segment)
    return kept


def load_branches(config: dict, device: str | None) -> dict[str, Branch]:
    return {
        "audio": AudioBranch.from_config(config, device),
        "body": BodyBranch(),
        "scene": SceneBranch.from_config(config, device),
    }


def version_info(config: dict, branches: dict[str, Branch]) -> dict[str, Any]:
    scene = branches["scene"]
    if not isinstance(scene, SceneBranch):
        raise TypeError(f"branches['scene'] must be a SceneBranch, got {type(scene).__name__}")
    return {
        "audio_heads_dir": config["branches"]["audio"]["heads_dir"],
        "audio_layer": str(config["branches"]["audio"]["layer"]),
        "body_head": config["branches"]["body"]["head"],
        "scene_schema_id": scene.encoder.schema_id,
        "bark_detector": config["bark_detector"],
        "categories": CATEGORIES,
    }


def check_or_write_info(path: Path, info: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(info)):
            raise ValueError(
                f"{path.parent} was made with other model versions; delete that folder "
                f"to rebuild.\n  cached: {existing}\n  now:    {info}"
            )
        return
    
    path.write_text(json.dumps(info, indent=1, sort_keys=True), encoding="utf-8")


def segment_row(video: Any, segment: BarkSegment, outputs: dict[str, BranchOutput]) -> dict:
    row: dict[str, Any] = {
        "clip_id": f"{FUSION_PREFIX}{video.video_id}@{segment.start_s:.3f}-{segment.end_s:.3f}",
        "video_id": video.video_id,
        "dog_id": video.dog_id,
        "split": video.split,
        "category": video.category,
        "start_s": round(segment.start_s, 3),
        "end_s": round(segment.end_s, 3),
        "confidence": segment.confidence,
        "num_frames": len(segment.frames),
    }

    for name in BRANCH_NAMES:
        output = outputs[name]
        row[f"{name}_present"] = bool(output.present)
        for category, probability in zip(CATEGORIES, output.probs, strict=True):
            row[f"{name}_{category}"] = float(probability)
    return row


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.is_file(), index=False, lineterminator="\n")


def cache_branch_outputs(device: str | None = None, verbose: bool = True) -> Path:
    config = load_config()
    videos = fusion_videos()
    cache_dir = data_path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    segments_path = cache_dir / SEGMENTS_FILENAME
    videos_path = cache_dir / VIDEOS_FILENAME

    done: set[str] = set()
    if videos_path.is_file():
        done = set(read_csv(videos_path)["video_id"])
    if verbose:
        print(f"{len(videos)} fusion videos, {len(done)} already cached")
    if len(done) == len(videos):
        return segments_path

    detector = load_bark_detector(config["bark_detector"]["model"], device, verbose=False)
    settings = settings_from_config(config)
    branches = load_branches(config, device)
    check_or_write_info(cache_dir / INFO_FILENAME, version_info(config, branches))

    for video in videos.itertuples():
        # itertuples types every field as Scalar; fusion_videos() guarantees these types.
        video_id = cast(str, video.video_id)
        video_path = cast(Path, video.video_path)
        start_s = cast(float, video.video_start_s)
        end_s = cast(float, video.video_end_s)
        if video_id in done:
            continue

        started = time.perf_counter()
        segments = build_segments(
            video_path, detector, settings, source_id=video_id, verbose=False
        )
        kept = segments_in_span(segments, start_s, end_s)
        rows: list[dict] = []

        for segment in kept:
            outputs: dict[str, BranchOutput] = {}
            for name in BRANCH_NAMES:
                outputs[name] = branches[name].predict(segment)
            rows.append(segment_row(video, segment, outputs))
        append_rows(segments_path, rows)
        seconds = round(time.perf_counter() - started, 1)
        append_rows(
            videos_path,
            [
                {
                    "video_id": video.video_id,
                    "split": video.split,
                    "category": video.category,
                    "segments_found": len(segments),
                    "segments_kept": len(kept),
                    "seconds": seconds,
                }
            ],
        )
        if verbose:
            print(f"{video.video_id} {video.category:<18} {len(kept):>3} segments  {seconds:>6.1f} s")
    return segments_path


def load_branch_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    load_data_dir_from_env_file()
    cache_dir = data_path(CACHE_DIR)
    segments = pd.read_csv(cache_dir / SEGMENTS_FILENAME)
    videos = pd.read_csv(cache_dir / VIDEOS_FILENAME)
    return segments, videos