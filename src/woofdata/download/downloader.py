"""Download the raw datasets in configs/datasets.yaml into <data_dir>/raw/<id>/

Run with:
    python -m woofdata.download          # every dataset
    python -m woofdata.download <id>     # only that dataset

Downloads go through dataset-downloader: interrupted files resume on the next run, and
every run appends one row per dataset to <data_dir>/raw/download_log.csv.
Tokens are read only from the repo's .env file (see .env.example).

Exit codes: 0 all done, 1 a dataset failed or needs action, 2 bad usage, 130 interrupted.
"""

from __future__ import annotations

import argparse
from typing import Any

from dataset_downloader import Downloader
from dataset_downloader.auth import EnvFileError
from dataset_downloader.models import DatasetResult

from woofdata.download.datasets import load_datasets, select_dataset
from woofdata.path import ENV_FILE, get_data_dir, load_data_dir_from_env_file
from woofdata.utils_console import use_utf8_output

LOG_FILENAME: str = "download_log.csv"

# dataset-downloader statuses that mean nothing is left to do.
DONE_STATUSES: tuple[str, ...] = ("ok", "skipped")

EXIT_OK: int = 0
EXIT_FAILED: int = 1
EXIT_USAGE: int = 2
EXIT_INTERRUPTED: int = 130


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


def run_downloads(selected: list[dict[str, Any]]) -> list[DatasetResult]:
    load_data_dir_from_env_file()

    raw_dir = get_data_dir() / "raw"
    downloader = Downloader(raw_dir, log_path=raw_dir / LOG_FILENAME, env_file=ENV_FILE)

    for dataset in selected:
        queue_dataset(downloader, dataset)

    return downloader.run()


def download_one(dataset_id: str) -> list[DatasetResult]:
    return run_downloads(select_dataset(load_datasets(), dataset_id))


def download_all() -> list[DatasetResult]:
    return run_downloads(load_datasets())


def main(argv: list[str] | None = None) -> int:
    """Download every dataset, or only the one named on the command line.

    Returns:
        Exit code: 0 all done, 1 a dataset failed or needs action, 2 bad .env, 130 interrupted.
    """
    use_utf8_output()
    parser = argparse.ArgumentParser(
        prog="python -m woofdata.download",
        description="Download raw datasets into <data_dir>/raw/<id>/.",
    )
    parser.add_argument("dataset_id", nargs="?", help="dataset to download (default all)")
    args = parser.parse_args(argv)
    try:
        selected = select_dataset(load_datasets(), args.dataset_id)
    except ValueError as error:
        parser.error(str(error))  # exits with code 2

    try:
        results = run_downloads(selected)
    except EnvFileError as error:
        print(error)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED

    if all(result.status in DONE_STATUSES for result in results):
        return EXIT_OK
    return EXIT_FAILED
