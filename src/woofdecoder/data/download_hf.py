"""Download "hf" datasets: a snapshot of a Hugging Face dataset repo at a fixed commit.

huggingface_hub's own progress bars are switched off (its warnings still show). Instead,
a small watcher thread measures the bytes on disk and moves the dataset's single progress bar.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from woofdecoder.data.download_config import DISK_POLL_INTERVAL_S
from woofdecoder.data.download_reg import Dataset
from woofdecoder.data.download_report import start_progress

if TYPE_CHECKING:
    # Only for type hints; the real import happens inside the functions (see below).
    from huggingface_hub import DatasetInfo


def bytes_on_disk(target_dir: Path) -> int:
    """Count downloaded bytes: finished files plus files still being written.

    huggingface_hub writes unfinished files as .incomplete inside target_dir/.cache,
    then moves them into place, so each byte is counted exactly once.
    """
    total_bytes = 0
    for path in target_dir.rglob("*"):
        relative_path = path.relative_to(target_dir)
        in_cache = ".cache" in relative_path.parts
        if not path.is_file():
            continue
        if in_cache and path.suffix != ".incomplete":
            continue
        total_bytes += path.stat().st_size
    return total_bytes


def watch_bytes(target_dir: Path, progress: tqdm, stop_event: threading.Event) -> None:
    """Move the progress bar to the bytes on disk until stop_event is set."""
    while not stop_event.wait(DISK_POLL_INTERVAL_S):
        progress.update(bytes_on_disk(target_dir) - progress.n)


def measure_hf_dataset(repo_info: DatasetInfo, allow_patterns: list[str] | None) -> int:
    """Add up the size of the repo files that allow_patterns selects.

    Args:
        repo_info: Dataset info fetched with files_metadata=True.
        allow_patterns: Glob patterns of files to download, or None for all files.

    Returns:
        Total bytes to download.
    """
    from huggingface_hub.utils import filter_repo_objects

    if repo_info.siblings is None:
        raise ValueError(f"{repo_info.id}: Hugging Face did not return the file list")

    size_by_filename: dict[str, int] = {}
    for repo_file in repo_info.siblings:
        size_by_filename[repo_file.rfilename] = repo_file.size or 0

    total_bytes = 0
    for filename in filter_repo_objects(size_by_filename, allow_patterns=allow_patterns):
        total_bytes += size_by_filename[filename]
    return total_bytes


def show_hf_warnings_above_bars() -> AbstractContextManager:
    """Print huggingface_hub warnings (e.g. no HF_TOKEN) on their own line above the bars.

    Without this, a warning is written into the middle of a progress bar line.

    Returns:
        A context manager; use it around the downloads.
    """
    from huggingface_hub.utils import logging as hf_logging
    from tqdm.contrib.logging import logging_redirect_tqdm

    # get_logger also sets up huggingface_hub's console handler, which is what gets redirected.
    hf_root_logger = hf_logging.get_logger("huggingface_hub")
    return logging_redirect_tqdm(loggers=[hf_root_logger])


def download_hf_dataset(dataset: Dataset, target_dir: Path, progress: tqdm) -> str:
    """Download a Hugging Face dataset snapshot. Already-downloaded files are not fetched again.

    Args:
        dataset: Registry entry with kind "hf".
        target_dir: Folder to save into.
        progress: The dataset's progress bar.

    Returns:
        The commit sha that was downloaded.
    """
    # Imported here so --list and the tests work without the "data" extra.
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    # Keep the console to our one line per dataset. HF warnings (e.g. no HF_TOKEN) still show.
    disable_progress_bars()

    repo_id = dataset.repo_id
    if repo_id is None:
        raise ValueError(f"{dataset.name}: an 'hf' dataset needs a repo_id")

    # Resolve "main" to a fixed sha first, so the log records exactly what we got.
    repo_info = HfApi().dataset_info(repo_id, revision=dataset.revision, files_metadata=True)
    resolved_sha = repo_info.sha
    if resolved_sha is None:
        raise ValueError(f"{dataset.name}: Hugging Face did not return a commit sha")

    total_bytes = measure_hf_dataset(repo_info, dataset.allow_patterns)
    start_progress(progress, total_bytes, bytes_on_disk(target_dir))

    stop_event = threading.Event()
    watcher = threading.Thread(target=watch_bytes, args=(target_dir, progress, stop_event))
    watcher.start()
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=resolved_sha,
            allow_patterns=dataset.allow_patterns,
            local_dir=target_dir,
        )
    finally:
        stop_event.set()
        watcher.join()

    progress.update(bytes_on_disk(target_dir) - progress.n)
    return resolved_sha
