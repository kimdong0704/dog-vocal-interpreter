"""Download the raw datasets in configs/datasets.yaml into <data_dir>/raw/<id>/

Run with:
    python -m woofdecoder.data.download          # every dataset
    python -m woofdecoder.data.download <id>     # only that dataset

or from Python (also inside Jupyter): download_one("<id>") or download_all().

Downloads go through dataset-downloader: interrupted files resume on the next run, and
every run appends one row per dataset to <data_dir>/raw/download_log.csv.
Tokens are read only from the repo's .env file (see .env.example).

Exit codes: 0 all done, 1 a dataset failed or needs action, 2 bad usage, 130 interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from dataset_downloader import Downloader
from dataset_downloader.auth import EnvFileError
from dataset_downloader.models import DatasetResult, FileTask

from woofdecoder.config import ENV_FILE, REPO_ROOT, get_data_dir, load_data_dir_from_env_file
from woofdecoder.data.console import use_utf8_output

DATASETS_CONFIG: Path = REPO_ROOT / "configs" / "datasets.yaml"
LOG_FILENAME: str = "download_log.csv"

# dataset-downloader statuses that mean nothing is left to do.
DONE_STATUSES: tuple[str, ...] = ("ok", "skipped")

EXIT_OK: int = 0
EXIT_FAILED: int = 1
EXIT_USAGE: int = 2
EXIT_INTERRUPTED: int = 130


def load_datasets(path: Path = DATASETS_CONFIG) -> list[dict[str, Any]]:
    """Read the dataset entries from configs/datasets.yaml.

    Returns:
        One dict per dataset, with "id", "name" and either "link" or "provider" + "args".
    """
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config["datasets"]


def select_dataset(datasets: list[dict[str, Any]], dataset_id: str | None) -> list[dict[str, Any]]:
    """Pick the datasets to download: all of them for None, else the one with dataset_id.

    Raises:
        ValueError: If dataset_id is not in the list; the message lists the valid ids.
    """
    if dataset_id is None:
        return datasets

    valid_ids: list[str] = []
    for dataset in datasets:
        if dataset["id"] == dataset_id:
            return [dataset]
        valid_ids.append(dataset["id"])
    raise ValueError(f"unknown dataset {dataset_id!r}. Valid ids: {', '.join(valid_ids)}")


def queue_dataset(downloader: Downloader, dataset: dict[str, Any]) -> None:
    """Queue one dataset into <raw>/<id>/ with Downloader.add or the named provider method."""
    dataset_id = dataset["id"]
    if "link" in dataset:
        downloader.add(
            dataset["link"],
            dest=dataset_id,
            patterns=dataset.get("patterns"),
            revision=dataset.get("revision"),
        )
    elif "provider" in dataset:
        provider_method = getattr(downloader, dataset["provider"])
        provider_method(dest=dataset_id, **dataset["args"])
    else:
        raise ValueError(f"{dataset_id}: needs either 'link' or 'provider' + 'args'")


def is_event_loop_running() -> bool:
    """Return True if this thread already runs an asyncio event loop (as Jupyter does)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def print_file_event(name: str, task: FileTask, details: dict[str, Any]) -> None:
    """Print one line per file event, e.g. "finished  esc50/ESC-50.zip".

    Used instead of the live progress bar inside Jupyter, where that bar needs ipywidgets
    and fails to render in VS Code. "reason" (failed, retry, skipped) is already redacted.
    """
    line = f"{name:>9}  {task.dataset}/{task.path}"
    reason = details.get("reason")
    if reason:
        line += f"  ({reason})"
    print(line, flush=True)


def run_downloads(selected: list[dict[str, Any]]) -> list[DatasetResult]:
    """Download the selected datasets into <data_dir>/raw/<id>/.

    Downloader.run() calls asyncio.run(), which fails inside a running event loop. In that
    case (Jupyter) the run moves to a worker thread, which has no loop of its own, and prints
    one text line per file instead of the live progress bar. From a terminal it runs in this
    thread with the progress bar, so Ctrl+C still stops it.

    Returns:
        One DatasetResult per dataset (the rows of the console summary).

    Raises:
        EnvFileError: If the .env file is missing or lacks an entry.
        KeyboardInterrupt: If the download was interrupted (partial files are kept).
    """
    load_data_dir_from_env_file()
    raw_dir = get_data_dir() / "raw"
    downloader = Downloader(raw_dir, log_path=raw_dir / LOG_FILENAME, env_file=ENV_FILE)
    for dataset in selected:
        queue_dataset(downloader, dataset)

    if not is_event_loop_running():
        return downloader.run()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(downloader.run, progress=False, on_event=print_file_event)
        return future.result()


def download_one(dataset_id: str) -> list[DatasetResult]:
    """Download one dataset from configs/datasets.yaml; finished files are skipped.

    Returns:
        The DatasetResult of that dataset (a list, like download_all).

    Raises:
        ValueError: If dataset_id is not in configs/datasets.yaml.
    """
    selected = select_dataset(load_datasets(), dataset_id)
    return run_downloads(selected)


def download_all() -> list[DatasetResult]:
    """Download every dataset in configs/datasets.yaml; finished datasets are skipped.

    Returns:
        One DatasetResult per dataset.
    """
    return run_downloads(load_datasets())


def main(argv: list[str] | None = None) -> int:
    """Download every dataset, or only the one named on the command line.

    Returns:
        Exit code: 0 all done, 1 a dataset failed or needs action, 2 bad .env, 130 interrupted.
    """
    use_utf8_output()
    parser = argparse.ArgumentParser(
        prog="python -m woofdecoder.data.download",
        description="Download raw datasets into <data_dir>/raw/<id>/.",
    )
    parser.add_argument("dataset_id", nargs="?", help="dataset to download (default all)")
    args = parser.parse_args(argv)
    try:
        select_dataset(load_datasets(), args.dataset_id)
    except ValueError as error:
        parser.error(str(error))  # exits with code 2

    try:
        if args.dataset_id is None:
            results = download_all()
        else:
            results = download_one(args.dataset_id)
    except EnvFileError as error:
        print(error)
        return EXIT_USAGE
    except KeyboardInterrupt:
        # The library has already logged the datasets as interrupted.
        return EXIT_INTERRUPTED

    for result in results:
        if result.status not in DONE_STATUSES:
            return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
