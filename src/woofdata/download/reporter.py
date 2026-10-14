"""Report what the datasets in configs/datasets.yaml need and what is already on disk.

Used by the download section of the data notebook. Nothing here downloads anything.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from dataset_downloader.auth import ENV_VARS, EnvFileError, read_env_file
from dataset_downloader.models import DatasetResult
from dataset_downloader.providers.detect import detect

from woofdata.download.datasets import load_datasets
from woofdata.path import ENV_FILE, get_data_dir

# Credentials the datasets in configs/datasets.yaml use.
CREDENTIAL_NAMES: tuple[str, ...] = (
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "ROBOFLOW_API_KEY",
    "HF_TOKEN",
)

EMPTY_CREDENTIAL_VALUES: tuple[str, ...] = ("", "None")
TOKEN_REQUIRED_PROVIDERS: tuple[str, ...] = ("kaggle", "roboflow")


# Not dataset files: download caches and the downloader's resume state.
IGNORED_DIR_NAMES: tuple[str, ...] = (".cache",)
RESUME_STATE_SUFFIX: str = ".part"

# A folder this much smaller than approx_size_gb is flagged.
SIZE_SHORTFALL_TOLERANCE: float = 0.10
BYTES_PER_GB: float = 1e9


class FolderStats(NamedTuple):
    """Dataset files in one folder, without caches and resume state."""

    file_count: int
    total_bytes: int
    partial_count: int  # unfinished downloads (resume-state files)


def credential_status(env_file: Path = ENV_FILE) -> dict[str, str]:
    """Return "set" or "missing" for each credential in the .env file, never the value.

    Empty values and "None" count as missing. A missing or incomplete .env file means
    every credential is missing.
    """
    try:
        env_values = read_env_file(env_file)
    except EnvFileError:
        env_values = {}

    status_by_name: dict[str, str] = {}

    for name in CREDENTIAL_NAMES:
        value = env_values.get(name, "").strip()

        if value in EMPTY_CREDENTIAL_VALUES:
            status_by_name[name] = "missing"
        else:
            status_by_name[name] = "set"

    return status_by_name


def needed_credentials(dataset: dict[str, Any]) -> str:
    """Return the .env entries a dataset uses, e.g. "KAGGLE_USERNAME, KAGGLE_KEY".

    Tokens the provider only uses when given are marked "(optional)"; "" means none.
    """
    if "provider" in dataset:
        provider = dataset["provider"]
    else:
        provider, _source, _options = detect(dataset["link"])

    env_var_names = list(ENV_VARS.get(provider, {}).values())
    if not env_var_names:
        return ""

    credentials_text = ", ".join(env_var_names)
    if provider not in TOKEN_REQUIRED_PROVIDERS:
        credentials_text += " (optional)"

    return credentials_text


def folder_stats(folder: Path) -> FolderStats:
    """Count the dataset files in folder and their size, skipping caches and resume state."""
    file_count = 0
    total_bytes = 0
    partial_count = 0

    for current_dir, dir_names, file_names in os.walk(folder):
        kept_dir_names: list[str] = []

        for dir_name in dir_names:
            if dir_name not in IGNORED_DIR_NAMES:
                kept_dir_names.append(dir_name)

        dir_names[:] = kept_dir_names  # os.walk then skips the ignored folders

        for file_name in file_names:
            if file_name.endswith(RESUME_STATE_SUFFIX):
                partial_count += 1
                continue

            file_count += 1
            total_bytes += os.path.getsize(os.path.join(current_dir, file_name))

    return FolderStats(file_count, total_bytes, partial_count)


def yes_no(flag: bool) -> str:
    """Return "yes" for True and "no" for False."""
    if flag:
        return "yes"
    return "no"


def dataset_table() -> pd.DataFrame:
    """Return one row per dataset: size, downloaded and credentials needed."""
    raw_dir = get_data_dir() / "raw"
    rows: list[dict[str, Any]] = []

    for dataset in load_datasets():
        dataset_id = dataset["id"]
        stats = folder_stats(raw_dir / dataset_id)
        is_downloaded = stats.file_count > 0 and stats.partial_count == 0

        rows.append(
            {
                "id": dataset_id,
                "name": dataset["name"],
                "approx_size_gb": dataset["approx_size_gb"],
                "downloaded": yes_no(is_downloaded),
                "credentials": needed_credentials(dataset),
            }
        )

    return pd.DataFrame(rows)


def results_table(results: list[DatasetResult]) -> pd.DataFrame:
    """Return one summary row per download result: status, files and size on disk."""
    raw_dir = get_data_dir() / "raw"
    rows: list[dict[str, Any]] = []

    for result in results:
        dataset_id = result.request.dest
        stats = folder_stats(raw_dir / dataset_id)

        rows.append(
            {
                "id": dataset_id,
                "status": str(result.status),
                "reason": result.reason,
                "files": stats.file_count,
                "size_gb": round(stats.total_bytes / BYTES_PER_GB, 3),
            }
        )

    return pd.DataFrame(rows)


def size_flag(stats: FolderStats, expected_gb: float | None) -> str:
    """Return why a folder looks wrong ("missing", "too small", "not in datasets.yaml"), or ""."""
    if stats.file_count == 0:
        return "missing"

    if expected_gb is None:
        return "not in datasets.yaml"

    size_gb = stats.total_bytes / BYTES_PER_GB
    smallest_ok_gb = expected_gb * (1 - SIZE_SHORTFALL_TOLERANCE)

    if size_gb < smallest_ok_gb:
        return "too small"

    return ""


def disk_check() -> pd.DataFrame:
    """Return file count and size of each folder in <data_dir>/raw/ next to the expected size.

    Datasets in configs/datasets.yaml without a folder get a row flagged "missing".
    """
    raw_dir = get_data_dir() / "raw"
    expected_gb_by_id: dict[str, float | None] = {}

    for dataset in load_datasets():
        expected_gb_by_id[dataset["id"]] = dataset["approx_size_gb"]

    if raw_dir.is_dir():
        for folder in sorted(raw_dir.iterdir()):
            if folder.is_dir() and folder.name not in IGNORED_DIR_NAMES:
                expected_gb_by_id.setdefault(folder.name, None)

    rows: list[dict[str, Any]] = []

    for folder_name, expected_gb in expected_gb_by_id.items():
        stats = folder_stats(raw_dir / folder_name)

        rows.append(
            {
                "folder": folder_name,
                "files": stats.file_count,
                "size_gb": round(stats.total_bytes / BYTES_PER_GB, 3),
                "expected_gb": expected_gb,
                "flag": size_flag(stats, expected_gb),
            }
        )

    return pd.DataFrame(rows)
