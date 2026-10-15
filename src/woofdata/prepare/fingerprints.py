"""Fingerprints of each prepared source, committed so teammates can confirm identical data.

A fingerprint is the number of rows, the number of files in processed/<source>/, a sha1
of the sorted (clip_id, split, mapped_category) list, and a sha1 of the sorted text rows of
events.csv and labels.csv where the source has them (so changed events, which drive the
frame targets, and changed labels show). It does not hash audio, image or .npy bytes, so
tiny resampling differences between machines do not count as a mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from woofdata.download.datasets import load_datasets
from woofdata.download.reporter import folder_stats
from woofdata.prepare import splits
from woofdata.prepare.manifest import ROWS_FILENAME, processed_dir
from woofdata.prepare.sources.stanford_shared import BOXES_FILENAME, INDEX_FILENAME
from woofdata.prepare.utils.label_files import EVENTS_FILENAME, LABELS_FILENAME
from woofdata.utils_fileio import read_csv, write_json

FINGERPRINTS_FILENAME: str = "fingerprints.json"
FINGERPRINT_COLUMNS: list[str] = ["clip_id", "split", "mapped_category"]
# Optional per-source text files hashed into the fingerprint (key -> file name).
HASHED_FILES: dict[str, str] = {
    "events_sha1": EVENTS_FILENAME,
    "labels_sha1": LABELS_FILENAME,
    # stanford_dogs has no rows: its shared split and boxes live in these two files.
    "index_sha1": INDEX_FILENAME,
    "boxes_sha1": BOXES_FILENAME,
}


def fingerprints_path(splits_dir: Path | None = None) -> Path:
    """Return <splits_dir>/fingerprints.json; splits_dir defaults to SPLITS_DIR."""
    if splits_dir is None:
        splits_dir = splits.SPLITS_DIR
    return splits_dir / FINGERPRINTS_FILENAME


def sorted_lines_sha1(lines: list[str]) -> str:
    """Return the sha1 of lines after sorting them (so row order does not matter)."""
    return hashlib.sha1("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def fingerprint_source(source: str, data_dir: Path | None = None) -> dict[str, Any]:
    """Return {"rows", "files", "sha1"} (+ "events_sha1", "labels_sha1" when those files
    exist) for processed/<source>/, read back from its files."""
    source_dir = processed_dir(data_dir) / source
    rows = read_csv(source_dir / ROWS_FILENAME)

    lines: list[str] = []
    for clip_id, split, mapped_category in rows[FINGERPRINT_COLUMNS].itertuples(index=False):
        lines.append(f"{clip_id}\t{split}\t{mapped_category}")

    fingerprint: dict[str, Any] = {
        "rows": len(rows),
        "files": folder_stats(source_dir).file_count,
        "sha1": sorted_lines_sha1(lines),
    }
    for key, filename in HASHED_FILES.items():
        if (source_dir / filename).is_file():
            file_lines = (source_dir / filename).read_text(encoding="utf-8").splitlines()
            fingerprint[key] = sorted_lines_sha1(file_lines)
    return fingerprint


def load_fingerprints(splits_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the recorded fingerprints by source ({} if the file does not exist yet)."""
    path = fingerprints_path(splits_dir)
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fingerprints_file:
        return json.load(fingerprints_file)


def record_fingerprint(
    source: str,
    fingerprint: dict[str, Any],
    update: bool = False,
    splits_dir: Path | None = None,
) -> str:
    """Compare a source's fingerprint with fingerprints.json and record it if allowed.

    A new source is recorded. A differing entry is only overwritten when update is True,
    because every teammate prepares on their own machine and the committed entry is the
    reference everyone compares against.

    Returns:
        "fingerprint recorded", "fingerprint match", "fingerprint updated (...)" or a
        "WARNING: ..." message naming the old and new counts.
    """
    fingerprints = load_fingerprints(splits_dir)
    previous = fingerprints.get(source)
    if previous == fingerprint:
        return "fingerprint match"
    if previous is not None and not update:
        difference = describe_difference(previous, fingerprint)
        return (
            f"WARNING: fingerprint differs from {FINGERPRINTS_FILENAME} ({difference}); "
            "entry not overwritten, rerun with --update-fingerprints if the change is intended"
        )

    fingerprints[source] = fingerprint
    write_json(fingerprints_path(splits_dir), fingerprints)
    if previous is None:
        return "fingerprint recorded"
    return f"fingerprint updated ({describe_difference(previous, fingerprint)})"


def describe_difference(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    """Return e.g. "rows 120 -> 118, files 10 -> 10, sha1 differs (...)" for two fingerprints."""
    differences: list[str] = []
    for key in ("rows", "files"):
        differences.append(f"{key} {expected[key]} -> {actual[key]}")
    meanings = {"sha1": "clip ids, splits or labels"} | HASHED_FILES
    any_sha1_differs = False
    for key, meaning in meanings.items():
        if expected.get(key) != actual.get(key):
            differences.append(f"{key} differs ({meaning})")
            any_sha1_differs = True
    if not any_sha1_differs:
        differences.append("sha1 same")
    return ", ".join(differences)


def prepared_sources(data_dir: Path | None = None) -> list[str]:
    """Return the datasets.yaml ids that have a processed/<id>/rows.csv on this machine."""
    sources: list[str] = []
    for dataset in load_datasets():
        if (processed_dir(data_dir) / dataset["id"] / ROWS_FILENAME).is_file():
            sources.append(dataset["id"])
    return sources


def check_fingerprints(data_dir: Path | None = None, splits_dir: Path | None = None) -> list[str]:
    """Re-compute each recorded or prepared source's fingerprint and compare.

    Returns:
        One message per source that differs, is not prepared here, or has nothing
        recorded; ["nothing recorded"] if fingerprints.json is empty; [] means match.
    """
    recorded = load_fingerprints(splits_dir)
    sources = set(recorded) | set(prepared_sources(data_dir))
    if not recorded and not sources:
        return ["nothing recorded"]

    differences: list[str] = []
    for source in sorted(sources):
        rows_path = processed_dir(data_dir) / source / ROWS_FILENAME
        if source not in recorded:
            differences.append(f"{source}: nothing recorded")
        elif not rows_path.is_file():
            differences.append(f"{source}: not prepared on this machine")
        else:
            actual = fingerprint_source(source, data_dir)
            if actual != recorded[source]:
                differences.append(f"{source}: {describe_difference(recorded[source], actual)}")
    return differences
