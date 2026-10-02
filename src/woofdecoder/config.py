"""Load the pipeline settings from configs/pipeline.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Path = REPO_ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the pipeline config YAML file.

    Args:
        path: Path to a YAML config file. Defaults to DEFAULT_CONFIG.

    Returns:
        The config as a nested dict.
    """
    if path is None:
        path = DEFAULT_CONFIG

    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config
