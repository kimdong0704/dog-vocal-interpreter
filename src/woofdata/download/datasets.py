from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from woofdata.path import REPO_ROOT

DATASETS_CONFIG: Path = REPO_ROOT / "configs" / "datasets.yaml"


def load_datasets(path: Path = DATASETS_CONFIG) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError(f"{path} must hold a mapping with a 'datasets' list")
    return config["datasets"]


def select_dataset(datasets: list[dict[str, Any]], dataset_id: str | None) -> list[dict[str, Any]]:
    if dataset_id is None:
        return datasets

    valid_ids: list[str] = []

    for dataset in datasets:
        if dataset["id"] == dataset_id:
            return [dataset]

        valid_ids.append(dataset["id"])

    raise ValueError(f"unknown dataset {dataset_id!r}. Valid ids: {', '.join(valid_ids)}")


def dataset_config(source: str) -> dict[str, Any]:
    """Return the configs/datasets.yaml entry of one source (for its pinned revision)."""
    return select_dataset(load_datasets(), source)[0]
