"""Download the raw datasets into <data_dir>/raw/<dataset_name>/

Run with:
    python -m woofdecoder.data.download                 # all batch-1 datasets
    python -m woofdecoder.data.download --list          # show the registry
    python -m woofdecoder.data.download --only act_env  # just some datasets

Resumable and safe to re-run
Extraction added later

download_reg.py : the registry of all datasets available for download
download_config.py : configuration for downloads (workers, log filename, etc.)
download_url.py : handles URL-based downloads
download_hf.py : handles Hugging Face dataset downloads
download_report.py : progress bars, download_log.json and the --list table
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from woofdecoder.config import get_data_dir, load_env_file
from woofdecoder.data.download_config import DEFAULT_WORKERS
from woofdecoder.data.download_hf import download_hf_dataset, show_hf_warnings_above_bars
from woofdecoder.data.download_reg import Dataset, select_datasets
from woofdecoder.data.download_report import (
    DatasetBar,
    DownloadResult,
    close_progress_bars,
    finish_progress,
    new_progress_bar,
    print_failures,
    print_registry,
    write_log,
)
from woofdecoder.data.download_url import download_url_dataset


def list_files_on_disk(target_dir: Path) -> list[dict]:
    """List finished files in a dataset folder, sorted by path.

    Skips the Hugging Face .cache folder and our own .part and .bad files.

    Returns:
        list of dict
        dict per file with "path" (relative, forward slashes), "size_bytes" and "mtime_ns".
    """
    file_entries: list[dict] = []
    if not target_dir.exists():
        return file_entries

    for path in sorted(target_dir.rglob("*")):
        relative_path = path.relative_to(target_dir)
        if not path.is_file() or ".cache" in relative_path.parts:
            continue
        if path.suffix in (".part", ".bad"):
            continue

        file_stat = path.stat()
        entry = {
            "path": relative_path.as_posix(),
            "size_bytes": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
        }
        file_entries.append(entry)
    return file_entries


# one dataset
def download_dataset(dataset: Dataset, raw_dir: Path, progress: DatasetBar) -> DownloadResult:
    """Download one dataset and report what happened. Never raises.

    Returns:
        A DownloadResult with status "ok", "skipped" (nothing changed on disk) or "failed".
    """
    start_time = time.perf_counter()
    target_dir = raw_dir / dataset.name
    result = DownloadResult(dataset=dataset)

    files_before = list_files_on_disk(target_dir)
    md5_by_filename: dict[str, str] = {}
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if dataset.kind == "hf":
            result.sha = download_hf_dataset(dataset, target_dir, progress)
        else:
            md5_by_filename = download_url_dataset(dataset, target_dir, progress)
    except Exception as error:  # noqa: BLE001  one dataset failing must not stop the others
        result.status = "failed"
        result.reason = f"{type(error).__name__}: {error}"

    files_after = list_files_on_disk(target_dir)
    if result.status == "ok" and files_after and files_after == files_before:
        result.status = "skipped"
        result.reason = "already downloaded"

    for entry in files_after:
        log_entry = {"path": entry["path"], "size_bytes": entry["size_bytes"]}
        if entry["path"] in md5_by_filename:
            log_entry["md5"] = md5_by_filename[entry["path"]]
        result.files.append(log_entry)
        result.size_bytes += entry["size_bytes"]

    result.seconds = time.perf_counter() - start_time
    finish_progress(progress, result.status)
    return result


# all datasets
def download_all(datasets: list[Dataset], raw_dir: Path, workers: int) -> list[DownloadResult]:
    """Download several datasets in parallel threads.

    Every dataset gets its console line before any download starts, so waiting
    datasets are visible at 0% too.

    Returns:
        One DownloadResult per dataset, in the same order as datasets
    """
    bars: list[DatasetBar] = []
    for position, dataset in enumerate(datasets):
        bars.append(new_progress_bar(dataset, position))

    with show_hf_warnings_above_bars(), ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for dataset, progress in zip(datasets, bars, strict=True):
            futures.append(executor.submit(download_dataset, dataset, raw_dir, progress))

        results: list[DownloadResult] = []
        for future in futures:
            results.append(future.result())

    close_progress_bars(bars)
    return results


# ---------------------------------------------------------------- command line


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(
        prog="python -m woofdecoder.data.download",
        description="Download raw datasets into <data_dir>/raw/<dataset_name>/.",
    )
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="download just these datasets")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel datasets")
    parser.add_argument("--data-dir", type=Path, help="overrides WOOFDECODER_DATA_DIR")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the downloader from the command line.

    Returns:
        Exit code: 0 if every dataset is ok or skipped, 1 if any failed.
    """
    args = parse_args(argv)
    if args.list:
        print_registry()
        return 0
    if args.workers < 1:
        raise ValueError(f"--workers must be at least 1, got {args.workers}")

    # Load .env first: it can hold HF_TOKEN and WOOFDECODER_DATA_DIR.
    load_env_file()

    datasets = select_datasets(args.only)
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = get_data_dir()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = download_all(datasets, raw_dir, args.workers)
    write_log(results, raw_dir)
    print_failures(results)

    for result in results:
        if result.status == "failed":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
