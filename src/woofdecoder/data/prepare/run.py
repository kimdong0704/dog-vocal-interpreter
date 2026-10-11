"""Prepare downloaded datasets: raw/<id>/ -> processed/<id>/ + manifest + split files.

Run with:
    python -m woofdecoder.data.prepare             # every source with a converter, in order
    python -m woofdecoder.data.prepare <id>        # only that source
    python -m woofdecoder.data.prepare --dry-run   # list sources, inputs found, outputs planned
    python -m woofdecoder.data.prepare --check     # compare with data/splits/fingerprints.json
    python -m woofdecoder.data.prepare <id> --update-fingerprints   # accept a changed source

or from Python: prepare_one("<id>") or prepare_all().

A source whose fingerprint differs from fingerprints.json gets a WARNING and its entry is
kept; only --update-fingerprints overwrites it.

Exit codes: 0 done (or "match"), 1 fingerprints differ or nothing recorded, 2 bad usage.
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from woofdecoder.config import get_data_dir, load_data_dir_from_env_file
from woofdecoder.data.console import use_utf8_output
from woofdecoder.data.download import load_datasets, select_dataset
from woofdecoder.data.manifest import build_manifest, processed_dir, write_source_rows
from woofdecoder.data.prepare.fingerprints import (
    check_fingerprints,
    fingerprint_source,
    record_fingerprint,
)
from woofdecoder.data.prepare.registry import CONVERTERS
from woofdecoder.data.splits import check_splits, save_splits
from woofdecoder.data.status import folder_stats

STATUS_DONE: str = "done"
STATUS_SKIPPED: str = "skipped"
STATUS_NO_CONVERTER: str = "no converter yet"

WARNING_PREFIX: str = "WARNING"

EXIT_OK: int = 0
EXIT_MISMATCH: int = 1


@dataclass
class PrepareResult:
    """Outcome of preparing one source."""

    source: str
    status: str
    rows_written: int = 0
    files_written: int = 0
    seconds: float = 0.0
    message: str = ""


def prepare_source(dataset_id: str, update_fingerprints: bool = False) -> PrepareResult:
    """Rebuild processed/<id>/ from raw/<id>/ with the source's converter.

    Only processed/<id>/ is deleted and rebuilt; raw/ is never written to. A missing raw
    folder or converter is reported in the result, not raised. The fingerprint message
    (match, recorded, updated or WARNING) goes into the result.
    """
    started = time.perf_counter()
    converter = CONVERTERS.get(dataset_id)
    if converter is None:
        return PrepareResult(dataset_id, STATUS_NO_CONVERTER)
    data_dir = get_data_dir()
    raw_dir = data_dir / "raw" / dataset_id
    if not raw_dir.is_dir():
        message = f"raw folder missing ({raw_dir}); download it first"
        return PrepareResult(dataset_id, STATUS_SKIPPED, message=message)

    out_dir = processed_dir(data_dir) / dataset_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rows = converter(raw_dir, out_dir)
    write_source_rows(dataset_id, rows, data_dir)

    fingerprint = fingerprint_source(dataset_id, data_dir)
    fingerprint_message = record_fingerprint(dataset_id, fingerprint, update_fingerprints)
    seconds = round(time.perf_counter() - started, 1)
    return PrepareResult(
        dataset_id,
        STATUS_DONE,
        fingerprint["rows"],
        fingerprint["files"],
        seconds,
        fingerprint_message,
    )


def prepare_sources(
    dataset_ids: list[str], update_fingerprints: bool = False
) -> list[PrepareResult]:
    """Prepare each source in order; then rebuild the manifest and split files if any ran.

    Raises:
        ValueError: If the rebuilt manifest leaks a dog or leakage group across splits.
    """
    load_data_dir_from_env_file()
    results: list[PrepareResult] = []
    any_done = False
    for dataset_id in dataset_ids:
        result = prepare_source(dataset_id, update_fingerprints)
        results.append(result)
        if result.status == STATUS_DONE:
            any_done = True

    if any_done:
        manifest = build_manifest()
        check_splits(manifest)
        save_splits(manifest)
    return results


def prepare_one(dataset_id: str, update_fingerprints: bool = False) -> list[PrepareResult]:
    """Prepare one source from configs/datasets.yaml.

    Raises:
        ValueError: If dataset_id is not in configs/datasets.yaml.
    """
    selected = select_dataset(load_datasets(), dataset_id)
    return prepare_sources([selected[0]["id"]], update_fingerprints)


def prepare_all(update_fingerprints: bool = False) -> list[PrepareResult]:
    """Prepare every source in configs/datasets.yaml order (those without a converter are listed)."""
    dataset_ids: list[str] = []
    for dataset in load_datasets():
        dataset_ids.append(dataset["id"])
    return prepare_sources(dataset_ids, update_fingerprints)


def dry_run() -> pd.DataFrame:
    """Return one row per source: converter, inputs found in raw/, outputs planned. Writes nothing."""
    load_data_dir_from_env_file()
    data_dir = get_data_dir()
    rows: list[dict[str, Any]] = []
    for dataset in load_datasets():
        dataset_id = dataset["id"]
        raw_dir = data_dir / "raw" / dataset_id
        if raw_dir.is_dir():
            inputs_found = f"{folder_stats(raw_dir).file_count} files"
        else:
            inputs_found = "raw folder missing"
        if dataset_id in CONVERTERS:
            outputs_planned = f"processed/{dataset_id}/"
        else:
            outputs_planned = STATUS_NO_CONVERTER
        rows.append({"id": dataset_id, "inputs_found": inputs_found, "outputs": outputs_planned})
    return pd.DataFrame(rows)


def print_results(results: list[PrepareResult]) -> None:
    """Print one line per source: status, rows, files, seconds and any message."""
    for result in results:
        line = f"{result.source:<22} {result.status}"
        if result.status == STATUS_DONE:
            line += f": {result.rows_written} rows, {result.files_written} files, "
            line += f"{result.seconds} s"
        if result.message.startswith(WARNING_PREFIX):
            line += f"\n  {result.message}"  # own line, so a changed source is hard to miss
        elif result.message:
            line += f"  ({result.message})"
        print(line)


def run_check() -> int:
    """Print "match" or each differing (or unrecorded) source; return the exit code."""
    load_data_dir_from_env_file()
    differences = check_fingerprints()
    if not differences:
        print("match")
        return EXIT_OK
    for difference in differences:
        print(difference)
    return EXIT_MISMATCH


def main(argv: list[str] | None = None) -> int:
    """Prepare every source, one source, or only list (--dry-run) or verify (--check)."""
    use_utf8_output()
    parser = argparse.ArgumentParser(
        prog="python -m woofdecoder.data.prepare",
        description="Prepare raw datasets into <data_dir>/processed/<id>/.",
    )
    parser.add_argument("dataset_id", nargs="?", help="source to prepare (default all)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="list what would be prepared")
    mode.add_argument("--check", action="store_true", help="compare with fingerprints.json")
    parser.add_argument(
        "--update-fingerprints",
        action="store_true",
        help="overwrite fingerprints.json entries that differ (only when the change is intended)",
    )
    args = parser.parse_args(argv)
    try:
        select_dataset(load_datasets(), args.dataset_id)
    except ValueError as error:
        parser.error(str(error))  # exits with code 2

    if args.check:
        return run_check()
    if args.dry_run:
        print(dry_run().to_string(index=False))
        return EXIT_OK
    if args.dataset_id is None:
        print_results(prepare_all(args.update_fingerprints))
    else:
        print_results(prepare_one(args.dataset_id, args.update_fingerprints))
    return EXIT_OK
