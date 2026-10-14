"""ESC-50 converter: 2,000 five-second environmental clips, one padded 10 s window each.

Raw file: raw/esc50/ESC-50.zip, a GitHub archive whose members sit under one top folder
"ESC-50-<commit>/". Only meta/esc50.csv and audio/*.wav are extracted (into a temporary
folder, since raw/ is never written to).
The 40 "dog" clips have only a clip-level label, so they get no target file. The other
1,960 clips are negatives with all-zero frame targets; folds 1-3 double as noise for
augmentation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path, PurePosixPath

from woofdata.download.datasets import dataset_config
from woofdata.prepare.manifest import ManifestRow, make_clip_id
from woofdata.prepare.utils.archives import extract_members
from woofdata.prepare.utils.audio import convert_audio
from woofdata.prepare.utils.label_files import write_events, write_labels, write_sed_classes
from woofdata.prepare.utils.windows import events_to_frame_targets, make_windows, save_frame_targets
from woofdata.utils_fileio import read_csv, relative_data_path
from woofdecoder.sed_classes import SED_CLASSES

SOURCE: str = "esc50"
ARCHIVE_FILENAME: str = "ESC-50.zip"
METADATA_PATTERN: str = "*/meta/esc50.csv"
AUDIO_PATTERN: str = "*/audio/*.wav"
AUDIO_DIRNAME: str = "audio"
TARGETS_DIRNAME: str = "targets"
DOG_CATEGORY: str = "dog"
DOG_NOTE: str = "clip-level dog label, no frame targets"

# The dataset's own folds keep most clips of one source recording together; the few that
# do not follow the lowest fold of their source (see lowest_fold_by_source).
FOLD_TO_SPLIT: dict[str, str] = {"1": "train", "2": "train", "3": "train", "4": "val", "5": "test"}
# The esc10 subset is CC BY; the rest of ESC-50 is CC BY-NC.
ESC10_LICENSE: str = "CC-BY-3.0"
OTHER_LICENSE: str = "CC-BY-NC-3.0"


def clip_license(esc10: str) -> str:
    """Return the license of one clip from its esc10 column ("True" or "False")."""
    if esc10 == "True":
        return ESC10_LICENSE
    return OTHER_LICENSE


def clip_split(fold: str) -> str:
    """Return the split of one clip from its fold ("1" to "5").

    Raises:
        ValueError: If fold is not in FOLD_TO_SPLIT.
    """
    if fold not in FOLD_TO_SPLIT:
        raise ValueError(f"unknown ESC-50 fold {fold!r}")
    return FOLD_TO_SPLIT[fold]


def lowest_fold_by_source(clips: list[dict[str, str]]) -> dict[str, str]:
    """Return {src_file: lowest fold of its clips}, so one recording's clips share a split.

    Two sources (209698, 234879) have clips in both fold 4 and fold 5, which would put one
    recording in val and test.
    """
    lowest_fold: dict[str, str] = {}
    for clip in clips:
        source_file = clip["src_file"]
        if source_file not in lowest_fold or int(clip["fold"]) < int(lowest_fold[source_file]):
            lowest_fold[source_file] = clip["fold"]
    return lowest_fold


def archive_commit(source: str) -> str:
    """Return the commit sha in the source's GitHub archive url (".../archive/<sha>.zip")."""
    return PurePosixPath(dataset_config(source)["args"]["url"]).stem


def convert_clip(
    clip: dict[str, str], audio_path: Path, out_dir: Path, split: str, source_version: str
) -> ManifestRow:
    """Write one clip's 16 kHz WAV (and target .npy unless it is a dog clip); return its row."""
    file_name = clip["filename"]
    wav_path = out_dir / AUDIO_DIRNAME / file_name
    duration_s = convert_audio(audio_path, wav_path)
    start_s, end_s = make_windows(duration_s)[0]  # 5 s clip: one padded 10 s window

    is_dog = clip["category"] == DOG_CATEGORY
    target_path = ""
    if not is_dog:
        npy_path = out_dir / TARGETS_DIRNAME / f"{Path(file_name).stem}.npy"
        targets = events_to_frame_targets([], SED_CLASSES, start_s)  # all zero, shape (250, 6)
        save_frame_targets(npy_path, targets)
        target_path = relative_data_path(npy_path)

    label_type = "object"
    notes = ""
    if is_dog:
        label_type = "call_type"
        notes = DOG_NOTE
    return ManifestRow(
        clip_id=make_clip_id(SOURCE, f"{AUDIO_DIRNAME}/{file_name}"),
        source=SOURCE,
        source_version=source_version,
        source_license=clip_license(clip["esc10"]),
        dog_id=clip["src_file"],
        split=split,
        modality="audio",
        file_path=relative_data_path(wav_path),
        start_s=start_s,
        end_s=end_s,
        target_path=target_path,
        original_label=clip["category"],
        label_type=label_type,
        label_provenance="expert",
        notes=notes,
    )


def print_report(rows: list[ManifestRow]) -> None:
    """Print the clips and dog clips per split."""
    for split in ("train", "val", "test"):
        clip_count = 0
        dog_count = 0
        for row in rows:
            if row.split == split:
                clip_count += 1
                if row.original_label == DOG_CATEGORY:
                    dog_count += 1
        print(f"  {split}: {clip_count} clips, {dog_count} dog clips")


def prepare(raw_dir: Path, out_dir: Path) -> list[ManifestRow]:
    """Convert ESC-50.zip into WAVs, negative targets, events.csv (empty) and rows."""
    archive_path = raw_dir / ARCHIVE_FILENAME
    rows: list[ManifestRow] = []
    label_rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as extract_dir:
        metadata_path = extract_members(archive_path, METADATA_PATTERN, Path(extract_dir))[0]
        audio_paths = extract_members(archive_path, AUDIO_PATTERN, Path(extract_dir))
        audio_by_name: dict[str, Path] = {}
        for audio_path in audio_paths:
            audio_by_name[audio_path.name] = audio_path

        metadata = read_csv(metadata_path)
        clips = metadata.to_dict("records")
        lowest_fold = lowest_fold_by_source(clips)
        source_version = archive_commit(SOURCE)
        moved_count = 0
        for clip in clips:
            if clip["filename"] not in audio_by_name:
                raise ValueError(f"esc50.csv lists {clip['filename']}, which is not in the zip")
            split = clip_split(lowest_fold[clip["src_file"]])
            if split != clip_split(clip["fold"]):
                moved_count += 1
            audio_path = audio_by_name[clip["filename"]]
            row = convert_clip(clip, audio_path, out_dir, split, source_version)
            rows.append(row)
            label_rows.append({"clip_id": row.clip_id} | clip)
    print(f"  {moved_count} clips follow their source's lowest fold into another split")

    write_events(out_dir, [])  # no clip has time-stamped events
    write_labels(out_dir, label_rows, ["clip_id"] + list(metadata.columns))
    write_sed_classes(out_dir, SED_CLASSES)  # negatives: every class is known to be absent
    print_report(rows)
    return rows
