"""Load the pipeline settings from configs/pipeline.yaml and locate the data folder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Path = REPO_ROOT / "configs" / "pipeline.yaml"
DATA_DIR_ENV_VAR: str = "WOOFDECODER_DATA_DIR"
ENV_FILE: Path = REPO_ROOT / ".env"


def load_data_dir_from_env_file(path: Path | None = None) -> bool:
    """Set WOOFDECODER_DATA_DIR from the .env file, unless the terminal already set it.

    Only this one entry is read. The tokens in .env are read by dataset-downloader itself
    and never put into the environment, so other programs started from here cannot see them.

    Args:
        path: The .env file to read. Defaults to ENV_FILE (<repo root>/.env).

    Returns:
        True if WOOFDECODER_DATA_DIR was set from the file, False otherwise.
    """
    if path is None:
        path = ENV_FILE
    if DATA_DIR_ENV_VAR in os.environ or not path.exists():
        return False

    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() != DATA_DIR_ENV_VAR:
            continue
        data_dir = value.strip().strip("\"'")
        if not data_dir:
            return False
        os.environ[DATA_DIR_ENV_VAR] = data_dir
        return True
    return False


def get_data_dir() -> Path:
    """Find the folder that holds all datasets

    Save data on another drive by setting WOOFDECODER_DATA_DIR

    Returns:
        The path from WOOFDECODER_DATA_DIR if set, otherwise <repo root>/data.
    """
    env_value = os.environ.get(DATA_DIR_ENV_VAR)
    if env_value:
        return Path(env_value)

    return REPO_ROOT / "data"


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
