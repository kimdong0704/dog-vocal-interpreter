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


def load_env_file(path: Path | None = None) -> bool:
    """Load tokens and settings from a .env file into the environment.

    The .env file holds personal values (HF_TOKEN, WOOFDECODER_DATA_DIR, ...) and is
    never committed; see .env.example. Variables already set in the environment win,
    so a value set in the terminal always overrides the file.

    Args:
        path: The .env file to read. Defaults to ENV_FILE (<repo root>/.env).

    Returns:
        True if the file existed and was loaded, False if there is no file.
    """
    # Imported here because python-dotenv is in the "data" extra, not the base install.
    from dotenv import load_dotenv

    if path is None:
        path = ENV_FILE
    if not path.exists():
        return False

    return load_dotenv(path, override=False)


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
