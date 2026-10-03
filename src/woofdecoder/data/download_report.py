"""Report downloads: the progress bars, the download_log.json file and the --list table."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from woofdecoder.data.download_config import LOG_FILENAME, PROGRESS_BAR_FORMAT
from woofdecoder.data.download_reg import REGISTRY, Dataset


@dataclass
class DownloadResult:
    """What happened to one dataset. status is "ok", "skipped" or "failed"."""

    dataset: Dataset
    status: str = "ok"
    reason: str = ""
    sha: str | None = None
    files: list[dict] = field(default_factory=list)
    size_bytes: int = 0
    seconds: float = 0.0


def write_log(results: list[DownloadResult], raw_dir: Path) -> Path:
    """Add these results to <raw_dir>/download_log.json, keeping other datasets' entries.

    Args:
        results: One result per downloaded dataset.
        raw_dir: The <data_dir>/raw folder.

    Returns:
        The path of the log file.
    """
    log_path = raw_dir / LOG_FILENAME
    log: dict[str, dict] = {}
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))

    download_date = datetime.now(UTC).isoformat(timespec="seconds")
    for result in results:
        dataset = result.dataset
        log[dataset.name] = {
            "license": dataset.license,
            "license_url": dataset.license_url,
            "revision": dataset.revision,
            "sha": result.sha,
            "files": result.files,
            "download_date": download_date,
            "status": result.status,
            "reason": result.reason,
        }

    # Write a temp file first so an interrupted run never leaves a half-written log.
    temp_path = log_path.with_name(LOG_FILENAME + ".tmp")
    temp_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    temp_path.replace(log_path)
    return log_path


class DatasetBar(tqdm):
    """A tqdm bar whose {speed} shows "0.00B/s" before any bytes arrive, instead of "?B/s"."""

    @property
    def format_dict(self) -> dict:
        """tqdm's display values, plus "speed" used by PROGRESS_BAR_FORMAT."""
        values = super().format_dict
        # "rate" is tqdm's recent speed. It is None before any bytes arrive, and again
        # when the bar is closed, so fall back to the average speed (0 if nothing came).
        bytes_per_s = values["rate"]
        if bytes_per_s is None:
            downloaded_bytes = values["n"] - values["initial"]
            bytes_per_s = 0.0
            if values["elapsed"] > 0 and downloaded_bytes > 0:
                bytes_per_s = downloaded_bytes / values["elapsed"]
        values["speed"] = tqdm.format_sizeof(bytes_per_s) + "B/s"
        return values


def new_progress_bar(dataset: Dataset, position: int) -> DatasetBar:
    """Create the console line for one dataset, shown from the start of the run.

    Until the dataset starts, the total is the registry's approximate size;
    start_progress() replaces it with the exact size.

    Args:
        dataset: Registry entry; its name starts the line.
        position: Line number, so parallel datasets do not overwrite each other.

    Returns:
        A bar counting bytes.
    """
    approx_bytes = int(dataset.approx_size_gb * 1e9)
    return DatasetBar(
        desc=dataset.name,
        total=approx_bytes,
        unit="B",
        unit_scale=True,
        position=position,
        leave=True,
        bar_format=PROGRESS_BAR_FORMAT,
    )


def start_progress(progress: tqdm, total_bytes: int | None, done_bytes: int) -> None:
    """Set a bar's exact total, and count bytes already on disk as done.

    Args:
        progress: The dataset's bar.
        total_bytes: Size of the whole dataset, or None if unknown.
        done_bytes: Bytes already downloaded by an earlier run.
    """
    progress.total = total_bytes
    # Set as the starting point, not via update(), so old bytes do not count as speed.
    progress.initial = done_bytes
    progress.n = done_bytes
    progress.last_print_n = done_bytes
    # Measure speed from now, not from when the bar appeared (it may have waited in line).
    now = time.time()
    progress.start_t = now
    progress.last_print_t = now
    progress.refresh()


def finish_progress(progress: tqdm, status: str) -> None:
    """Add the final status ("ok", "skipped", "failed") to a bar.

    The bar is closed later by close_progress_bars(), so every line stays in place.
    """
    progress.set_postfix_str(status)


def close_progress_bars(bars: list[DatasetBar]) -> None:
    """Close all bars top to bottom, leaving them on screen, so later output prints below."""
    for progress in bars:
        progress.close()


def print_failures(results: list[DownloadResult]) -> None:
    """Print why each failed dataset failed. Prints nothing if all succeeded."""
    for result in results:
        if result.status == "failed":
            print(f"{result.dataset.name} failed: {result.reason}")


def print_registry() -> None:
    """Print every registry entry: name, approximate size and license."""
    print(f"{'name':<22} {'size':>8}  license")
    for dataset in REGISTRY:
        size_text = f"{dataset.approx_size_gb:.2f} GB"
        print(f"{dataset.name:<22} {size_text:>8}  {dataset.license}")
