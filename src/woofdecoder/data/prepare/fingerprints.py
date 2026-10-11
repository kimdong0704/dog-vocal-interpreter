"""Fingerprints of each prepared source, committed so teammates can confirm identical data.

A fingerprint is the number of rows, the number of files in processed/<source>/, and a
sha1 of the sorted (clip_id, split, mapped_subtype) list. It does not hash audio or image
bytes, so tiny resampling differences between machines do not count as a mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from woofdecoder.data import splits
from woofdecoder.data.download import load_datasets
from woofdecoder.data.fileio import read_csv, write_json
from woofdecoder.data.manifest import ROWS_FILENAME, processed_dir
from woofdecoder.data.status import folder_stats

FINGERPRINTS_FILENAME: str = "fingerprints.json"
FINGERPRINT_COLUMNS: list[str] = ["clip_id", "split", "mapped_subtype"]


def fingerprints_path(splits_dir: Path | None = None) -> Path:
    """Return <splits_dir>/fingerprints.json; splits_dir defaults to SPLITS_DIR."""
    if splits_dir is None:
        splits_dir = splits.SPLITS_DIR
    return splits_dir / FINGERPRINTS_FILENAME


def fingerprint_source(source: str, data_dir: Path | None = None) -> dict[str, Any]:
    """Return {"rows", "files", "sha1"} for processed/<source>/, read back from rows.csv."""
    source_dir = processed_dir(data_dir) / source
    rows = read_csv(source_dir / ROWS_FILENAME)

    lines: list[str] = []
    for clip_id, split, mapped_subtype in rows[FINGERPRINT_COLUMNS].itertuples(index=False):
        lines.append(f"{clip_id}\t{split}\t{mapped_subtype}")
    lines.sort()
    digest = hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()

    return {"rows": len(rows), "files": folder_stats(source_dir).file_count, "sha1": digest}


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
    """Return e.g. "rows 120 -> 118, files 10 -> 10, sha1 differs" for two fingerprints."""
    differences: list[str] = []
    for key in ("rows", "files"):
        differences.append(f"{key} {expected[key]} -> {actual[key]}")
    if expected["sha1"] != actual["sha1"]:
        differences.append("sha1 differs (clip ids, splits or labels)")
    else:
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
