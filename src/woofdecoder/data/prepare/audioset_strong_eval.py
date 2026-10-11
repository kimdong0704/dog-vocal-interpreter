"""AudioSet strong-label eval converter: dog clips plus hard negatives, one 10 s window each.

Raw files: raw/audioset_strong_eval/data/eval-*.parquet with columns
    video_id: str, audio: {bytes, path}, labels / human_labels: list[str],
    events: list[{start, end, event_name}]  (event_name is the human-readable class name)
Every clip with at least one event in AUDIOSET_TO_SED is kept, plus NUM_NEGATIVES clips
without one (other animals and everyday sounds), so the bark detector's false alarms show.
No dog ids exist, so each clip is its own dog; all clips share one leakage group.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from woofdecoder.data.fileio import relative_data_path
from woofdecoder.data.manifest import ManifestRow, make_clip_id
from woofdecoder.data.prepare.common import (
    EVENTS_FILENAME,
    convert_audio,
    dataset_config,
    events_to_frame_targets,
    make_windows,
    save_frame_targets,
    write_events,
    write_sed_classes,
)
from woofdecoder.sed_classes import AUDIOSET_TO_SED, SED_CLASSES

SOURCE: str = "audioset_strong_eval"
PARQUET_PATTERN: str = "data/eval-*.parquet"
NUM_NEGATIVES: int = 300
NEGATIVE_SEED: int = 0
BATCH_ROWS: int = 32  # audio bytes per batch stay around 40 MB
LEAKAGE_GROUP: str = "audioset"
AUDIO_DIRNAME: str = "audio"
TARGETS_DIRNAME: str = "targets"
SOURCE_LICENSE: str = "CC-BY-4.0"
NOTES: str = (
    "Google raters; optimistic: bark detector was trained on AudioSet; "
    "labels CC BY 4.0; audio is YouTube content, do not redistribute"
)
# AudioSet "Pant" is human breathing, so dog panting is never labelled here.
UNLABELLED_CLASSES: tuple[str, ...] = ("pant",)


def has_dog_event(events: list[dict[str, Any]]) -> bool:
    """Return True if any event's name is in AUDIOSET_TO_SED."""
    for event in events:
        if event["event_name"] in AUDIOSET_TO_SED:
            return True
    return False


def select_clips(parquet_paths: list[Path]) -> tuple[list[str], list[str]]:
    """Return (video ids with a dog event, NUM_NEGATIVES sampled video ids without one).

    Only the video_id and events columns are read. Negatives are sampled with
    NEGATIVE_SEED from the sorted remaining ids, so every machine picks the same ones.
    """
    dog_ids: list[str] = []
    other_ids: list[str] = []
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=["video_id", "events"])
        for row in table.to_pylist():
            if has_dog_event(row["events"]):
                dog_ids.append(row["video_id"])
            else:
                other_ids.append(row["video_id"])
    negative_ids = random.Random(NEGATIVE_SEED).sample(sorted(other_ids), NUM_NEGATIVES)
    return sorted(dog_ids), sorted(negative_ids)


def clip_event_rows(clip_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return events.csv rows for every event of one clip (sed_class "" if unmapped)."""
    event_rows: list[dict[str, Any]] = []
    for event in events:
        event_rows.append(
            {
                "clip_id": clip_id,
                "start_s": event["start"],
                "end_s": event["end"],
                "sed_class": AUDIOSET_TO_SED.get(event["event_name"], ""),
                "original_label": event["event_name"],
            }
        )
    return event_rows


def dog_event_names(events: list[dict[str, Any]]) -> str:
    """Return the clip's mapped event names, sorted and joined with ";" ("" for negatives)."""
    names: set[str] = set()
    for event in events:
        if event["event_name"] in AUDIOSET_TO_SED:
            names.add(event["event_name"])
    return ";".join(sorted(names))


def convert_clip(
    row: dict[str, Any], out_dir: Path, source_version: str
) -> tuple[ManifestRow, list[dict[str, Any]]]:
    """Write one clip's 16 kHz WAV and target .npy; return its row and events.csv rows."""
    video_id = row["video_id"]
    wav_path = out_dir / AUDIO_DIRNAME / f"{video_id}.wav"
    duration_s = convert_audio(row["audio"]["bytes"], wav_path)
    start_s, end_s = make_windows(duration_s)[0]  # clips are 10 s at most: one padded window
    clip_id = make_clip_id(SOURCE, video_id)

    event_rows = clip_event_rows(clip_id, row["events"])
    sed_events: list[tuple[float, float, str]] = []
    for event_row in event_rows:
        if event_row["sed_class"]:
            sed_events.append((event_row["start_s"], event_row["end_s"], event_row["sed_class"]))
    targets = events_to_frame_targets(sed_events, SED_CLASSES, start_s)  # shape (250, 6)
    target_path = out_dir / TARGETS_DIRNAME / f"{video_id}.npy"
    save_frame_targets(target_path, targets)

    manifest_row = ManifestRow(
        clip_id=clip_id,
        source=SOURCE,
        source_version=source_version,
        source_license=SOURCE_LICENSE,
        dog_id=video_id,
        leakage_group=LEAKAGE_GROUP,
        split="val",
        modality="audio",
        file_path=relative_data_path(wav_path),
        start_s=start_s,
        end_s=end_s,
        target_path=relative_data_path(target_path),
        annotation_path=relative_data_path(out_dir / EVENTS_FILENAME),
        original_label=dog_event_names(row["events"]),
        label_type="call_type",
        label_provenance="expert",
        notes=NOTES,
    )
    return manifest_row, event_rows


def print_report(num_dog_clips: int, num_negatives: int, event_rows: list[dict[str, Any]]) -> None:
    """Print the clips kept and the number of events per SED class."""
    sed_counts: dict[str, int] = {}
    for class_name in SED_CLASSES:
        sed_counts[class_name] = 0
    for event_row in event_rows:
        if event_row["sed_class"]:
            sed_counts[event_row["sed_class"]] += 1
    print(f"  kept {num_dog_clips} clips with dog events, {num_negatives} negatives")
    print(f"  events per SED class: {sed_counts}")


def prepare(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Convert the kept AudioSet eval clips into WAVs, targets, events.csv and rows."""
    parquet_paths = sorted(raw_dir.glob(PARQUET_PATTERN))
    if not parquet_paths:
        raise ValueError(f"no {PARQUET_PATTERN} files in {raw_dir}")
    dog_ids, negative_ids = select_clips(parquet_paths)
    kept_ids = set(dog_ids) | set(negative_ids)
    source_version = dataset_config(SOURCE)["revision"]

    rows: list[ManifestRow] = []
    event_rows: list[dict[str, Any]] = []
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        columns = ["video_id", "audio", "events"]
        for batch in parquet_file.iter_batches(batch_size=BATCH_ROWS, columns=columns):
            for row in batch.to_pylist():
                if row["video_id"] not in kept_ids:
                    continue
                manifest_row, clip_events = convert_clip(row, out_dir, source_version)
                rows.append(manifest_row)
                event_rows.extend(clip_events)

    write_events(out_dir, event_rows)
    labelled_classes: list[str] = []
    for class_name in SED_CLASSES:
        if class_name not in UNLABELLED_CLASSES:
            labelled_classes.append(class_name)
    write_sed_classes(out_dir, labelled_classes)
    print_report(len(dog_ids), len(negative_ids), event_rows)
    return rows
