"""PlayPanT converter: dog-human play sessions -> session WAVs, 10 s windows, frame targets.

Raw files in raw/playpant/ (names taken from the zip listings, which differ from the readme):
    03 Play session recordings.zip  "03 Play session recordings/P-##/<session>.wav", 30 files
    04 Annotations.zip              "04 Annotations/FirstLayer/<session>.Table.1.selections.txt"
                                    "04 Annotations/SecondLayer/<session>.Table" (no .txt)
    05 Metadata.csv                 ";"-separated, one row per session (Audio_file, ID_dog)
A session id looks like "P-05-G-P_2": participant 05, microphone G, session 2. The macOS
"__MACOSX/" copies inside the zips are ignored.

Layer 2 (expert-reviewed Praat export) is used where a session has it, else Layer 1 (Raven
Lite). Layer 2 writes full category names, which are turned back into the readme's letter
codes so both layers give the same original_label. Bouts are kept as annotated.
Events stay in session time (like the rows' start_s/end_s); an event that spans two windows
is listed in events.csv under both.
"""

from __future__ import annotations

import random
import tarfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from woofdata.prepare.manifest import ManifestRow, make_clip_id
from woofdata.prepare.utils.archives import list_archive_members, open_archive, open_member
from woofdata.prepare.utils.audio import convert_audio
from woofdata.prepare.utils.label_files import (
    EVENTS_FILENAME,
    write_events,
    write_labels,
    write_sed_classes,
)
from woofdata.prepare.utils.windows import events_to_frame_targets, make_windows, save_frame_targets
from woofdata.utils_fileio import relative_data_path
from woofdecoder.sed_classes import PLAYPANT_TO_SED, SED_CLASSES

SOURCE: str = "playpant"
RECORDINGS_ARCHIVE: str = "03 Play session recordings.zip"
ANNOTATIONS_ARCHIVE: str = "04 Annotations.zip"
METADATA_FILENAME: str = "05 Metadata.csv"
METADATA_SEPARATOR: str = ";"
MACOS_FOLDER: str = "__MACOSX/"
RECORDING_SUFFIX: str = ".wav"
LAYER1_SUFFIX: str = ".Table.1.selections.txt"
LAYER2_SUFFIX: str = ".Table"
LAYER_1: str = "layer 1"
LAYER_2: str = "layer 2"
AUDIO_DIRNAME: str = "audio"
TARGETS_DIRNAME: str = "targets"

# Layer 2 text -> readme letter code. Layer 2 spells some names differently from the
# readme ("Grunt", "Human"); Howl (G) is listed though no session contains one.
LAYER2_TEXT_TO_CODE: dict[str, str] = {
    "Growl": "A",
    "Whine": "B",
    "Bark/yelp": "C",
    "Moan": "D",
    "Pant": "E",
    "Cough/woof": "F",
    "Howl": "G",
    "Grunt": "H",
    "Other": "I",
    "Sneeze": "J",
    "Shake": "K",
    "Human": "L",
}

SPLIT_SEED: int = 0
NUM_VAL_DOGS: int = 9  # the other 8 dogs go to test
MAPPED_CATEGORY: str = "playful_excited"
ORIGINAL_LABEL: str = "dog-human play"
# datasets.yaml pins no revision for the Zenodo record, so the version is named here.
SOURCE_VERSION: str = "zenodo18972388_v1"
SOURCE_LICENSE: str = "CC-BY-NC-4.0"
LICENSE_NOTE: str = "Zenodo says CC BY 4.0, readme says CC BY-NC 4.0; stricter adopted"

Event = tuple[float, float, str]  # (start_s, end_s, letter code), in session time


# --- Inputs ---


def session_members(member_names: list[str], suffix: str) -> dict[str, str]:
    """Return {session id: member name} for archive members whose file name ends in suffix."""
    members: dict[str, str] = {}
    for member_name in member_names:
        if member_name.startswith(MACOS_FOLDER):
            continue
        file_name = PurePosixPath(member_name).name
        if file_name.endswith(suffix):
            members[file_name.removesuffix(suffix)] = member_name
    return members


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Return {session id: its 05 Metadata.csv row, every column as text}."""
    metadata = pd.read_csv(path, sep=METADATA_SEPARATOR, dtype=str, keep_default_na=False)
    metadata_by_session: dict[str, dict[str, str]] = {}
    for record in metadata.to_dict("records"):
        metadata_by_session[record["Audio_file"].removesuffix(RECORDING_SUFFIX)] = record
    return metadata_by_session


def window_label_rows(
    rows: list[ManifestRow], metadata_by_session: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    """Return one labels.csv row per window: clip_id plus its session's metadata columns."""
    label_rows: list[dict[str, str]] = []
    for row in rows:
        session = PurePosixPath(row.file_path).stem
        label_rows.append({"clip_id": row.clip_id} | metadata_by_session[session])
    return label_rows


def assign_dog_splits(dog_ids: list[str]) -> dict[str, str]:
    """Shuffle the sorted, unique dog ids with SPLIT_SEED; the first NUM_VAL_DOGS go to val.

    random.Random gives the same order on every machine and Python 3 version.
    """
    shuffled = sorted(set(dog_ids))
    random.Random(SPLIT_SEED).shuffle(shuffled)
    split_by_dog: dict[str, str] = {}
    for position, dog_id in enumerate(shuffled):
        if position < NUM_VAL_DOGS:
            split_by_dog[dog_id] = "val"
        else:
            split_by_dog[dog_id] = "test"
    return split_by_dog


def matched_sessions(
    recordings: dict[str, str],
    annotations: set[str],
    metadata_by_session: dict[str, dict[str, str]],
) -> list[str]:
    """Return the sorted sessions that have a recording, annotations and a metadata row.

    Every session missing one of the three is printed, so nothing is dropped silently.
    """
    for session in sorted(set(recordings) - annotations):
        print(f"  recording without annotations: {session}")
    for session in sorted(annotations - set(recordings)):
        print(f"  annotations without a recording: {session}")
    for session in sorted(set(recordings) - set(metadata_by_session)):
        print(f"  recording without a metadata row: {session}")

    sessions: list[str] = []
    for session in sorted(recordings):
        if session in annotations and session in metadata_by_session:
            sessions.append(session)
    return sessions


# --- Annotations ---


def read_annotation_table(
    archive: zipfile.ZipFile | tarfile.TarFile, member_name: str
) -> pd.DataFrame:
    """Read one tab-delimited annotation file from the open annotations zip, all as text."""
    with open_member(archive, member_name) as member_file:
        return pd.read_csv(member_file, sep="\t", dtype=str, keep_default_na=False)


def layer1_events(table: pd.DataFrame) -> list[Event]:
    """Return (start_s, end_s, code) events from a Raven Lite (Layer 1) table.

    Codes carry stray spaces ("A "), which are stripped. A row with a blank code has no
    label to keep, so it is dropped.

    Raises:
        ValueError: If a code is not one of the readme's letters A-L.
    """
    events: list[Event] = []
    columns = zip(table["Begin Time (s)"], table["End Time (s)"], table["Annotation"], strict=True)
    for start_s, end_s, raw_code in columns:
        code = raw_code.strip()
        if not code:
            continue
        if code not in LAYER2_TEXT_TO_CODE.values():
            raise ValueError(f"unknown Layer 1 code {raw_code!r}")
        events.append((float(start_s), float(end_s), code))
    return events


def layer2_events(table: pd.DataFrame) -> list[Event]:
    """Return (start_s, end_s, code) events from a Praat (Layer 2) table, both tiers.

    Raises:
        ValueError: If a text label is not in LAYER2_TEXT_TO_CODE.
    """
    events: list[Event] = []
    for start_s, end_s, raw_text in zip(table["tmin"], table["tmax"], table["text"], strict=True):
        text = raw_text.strip()  # one label ends in a no-break space
        if text not in LAYER2_TEXT_TO_CODE:
            raise ValueError(f"unknown Layer 2 label {raw_text!r}")
        events.append((float(start_s), float(end_s), LAYER2_TEXT_TO_CODE[text]))
    return events


def read_session_events(
    archive: zipfile.ZipFile | tarfile.TarFile,
    session: str,
    layer1_members: dict[str, str],
    layer2_members: dict[str, str],
) -> tuple[list[Event], str]:
    """Return a session's events and the layer they came from (Layer 2 when it exists)."""
    if session in layer2_members:
        table = read_annotation_table(archive, layer2_members[session])
        return layer2_events(table), LAYER_2
    table = read_annotation_table(archive, layer1_members[session])
    return layer1_events(table), LAYER_1


# --- Windows ---


def window_events(
    clip_id: str, events: list[Event], start_s: float, end_s: float
) -> tuple[list[tuple[float, float, str]], list[dict[str, Any]]]:
    """Return the SED events overlapping one window, and its events.csv rows (all codes)."""
    sed_events: list[tuple[float, float, str]] = []
    event_rows: list[dict[str, Any]] = []
    for event_start_s, event_end_s, code in events:
        if event_start_s >= end_s or event_end_s <= start_s:
            continue
        sed_class = PLAYPANT_TO_SED.get(code, "")
        event_rows.append(
            {
                "clip_id": clip_id,
                "start_s": event_start_s,
                "end_s": event_end_s,
                "sed_class": sed_class,
                "original_label": code,
            }
        )
        if sed_class:
            sed_events.append((event_start_s, event_end_s, sed_class))
    return sed_events, event_rows


def session_windows(
    session: str,
    dog_id: str,
    split: str,
    layer: str,
    duration_s: float,
    events: list[Event],
    out_dir: Path,
) -> tuple[list[ManifestRow], list[dict[str, Any]]]:
    """Cut one session WAV into windows: one manifest row and one target .npy per window."""
    wav_path = out_dir / AUDIO_DIRNAME / f"{session}{RECORDING_SUFFIX}"
    rows: list[ManifestRow] = []
    event_rows: list[dict[str, Any]] = []
    for start_s, end_s in make_windows(duration_s):
        clip_id = make_clip_id(SOURCE, f"{AUDIO_DIRNAME}/{session}{RECORDING_SUFFIX}", start_s)
        sed_events, window_event_rows = window_events(clip_id, events, start_s, end_s)
        event_rows.extend(window_event_rows)
        targets = events_to_frame_targets(sed_events, SED_CLASSES, start_s)  # shape (250, 6)
        target_path = out_dir / TARGETS_DIRNAME / f"{session}@{start_s:.3f}.npy"
        save_frame_targets(target_path, targets)
        rows.append(
            ManifestRow(
                clip_id=clip_id,
                source=SOURCE,
                source_version=SOURCE_VERSION,
                source_license=SOURCE_LICENSE,
                dog_id=dog_id,
                split=split,
                modality="audio",
                file_path=relative_data_path(wav_path),
                start_s=start_s,
                end_s=end_s,
                target_path=relative_data_path(target_path),
                annotation_path=relative_data_path(out_dir / EVENTS_FILENAME),
                original_label=ORIGINAL_LABEL,
                label_type="context",
                label_provenance="protocol",
                mapped_category=MAPPED_CATEGORY,
                notes=f"{LICENSE_NOTE}; events from {layer}",
            )
        )
    return rows, event_rows


# --- Converter ---


def print_report(
    rows: list[ManifestRow], session_events: dict[str, list[Event]], layers: dict[str, str]
) -> None:
    """Print sessions, windows, layers used and event counts per letter code and SED class."""
    code_counts: Counter[str] = Counter()
    sed_counts: Counter[str] = Counter()
    for events in session_events.values():
        for _, _, code in events:
            code_counts[code] += 1
            sed_counts[PLAYPANT_TO_SED.get(code, "(none)")] += 1
    print(f"  {len(session_events)} sessions, {len(rows)} windows")
    print(f"  layers used: {dict(sorted(Counter(layers.values()).items()))}")
    print(f"  events per code: {dict(sorted(code_counts.items()))}")
    print(f"  events per SED class: {dict(sorted(sed_counts.items()))}")


def print_dog_splits(split_by_dog: dict[str, str]) -> None:
    """Print which dogs went to val and which to test."""
    for split in ("val", "test"):
        split_dogs: list[str] = []
        for dog_id, dog_split in sorted(split_by_dog.items()):
            if dog_split == split:
                split_dogs.append(dog_id)
        print(f"  {split} dogs ({len(split_dogs)}): {', '.join(split_dogs)}")


def prepare(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Convert raw/playpant/ into session WAVs, window targets, events.csv and rows."""
    metadata_by_session = read_metadata(raw_dir / METADATA_FILENAME)
    dog_by_session: dict[str, str] = {}
    for session, record in metadata_by_session.items():
        dog_by_session[session] = record["ID_dog"]
    dog_ids = sorted(set(dog_by_session.values()))
    print(f"  metadata: {len(dog_by_session)} rows, dog ids {', '.join(dog_ids)}")
    split_by_dog = assign_dog_splits(dog_ids)
    print_dog_splits(split_by_dog)

    recordings_path = raw_dir / RECORDINGS_ARCHIVE
    annotations_path = raw_dir / ANNOTATIONS_ARCHIVE
    recordings = session_members(sorted(list_archive_members(recordings_path)), RECORDING_SUFFIX)
    annotation_names = sorted(list_archive_members(annotations_path))
    layer1_members = session_members(annotation_names, LAYER1_SUFFIX)
    layer2_members = session_members(annotation_names, LAYER2_SUFFIX)
    annotated = set(layer1_members) | set(layer2_members)
    sessions = matched_sessions(recordings, annotated, metadata_by_session)

    rows: list[ManifestRow] = []
    event_rows: list[dict[str, Any]] = []
    session_events: dict[str, list[Event]] = {}
    layers: dict[str, str] = {}
    with (
        open_archive(recordings_path) as recording_archive,
        open_archive(annotations_path) as annotation_archive,
    ):
        for session in sessions:
            events, layer = read_session_events(
                annotation_archive, session, layer1_members, layer2_members
            )
            with open_member(recording_archive, recordings[session]) as recording_file:
                wav_path = out_dir / AUDIO_DIRNAME / f"{session}{RECORDING_SUFFIX}"
                duration_s = convert_audio(recording_file.read(), wav_path)
            dog_id = dog_by_session[session]
            session_rows, session_event_rows = session_windows(
                session, dog_id, split_by_dog[dog_id], layer, duration_s, events, out_dir
            )
            rows.extend(session_rows)
            event_rows.extend(session_event_rows)
            session_events[session] = events
            layers[session] = layer

    write_events(out_dir, event_rows)
    metadata_columns = list(next(iter(metadata_by_session.values())))
    write_labels(
        out_dir, window_label_rows(rows, metadata_by_session), ["clip_id"] + metadata_columns
    )
    write_sed_classes(out_dir, SED_CLASSES)  # the annotators could use every class
    print_report(rows, session_events, layers)
    return rows
