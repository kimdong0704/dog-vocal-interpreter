from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR_ENV_VAR: str = "WOOFDECODER_DATA_DIR"
ENV_FILE: Path = REPO_ROOT / ".env"


def load_data_dir_from_env_file(path: Path | None = None) -> bool:
    if path is None:
        path = ENV_FILE

    if DATA_DIR_ENV_VAR in os.environ or not path.exists():
        return False

    match = re.search(
        rf"^[ \t]*{DATA_DIR_ENV_VAR}[ \t]*=(.*)$", path.read_text(encoding="utf-8"), re.MULTILINE
    )
    data_dir = match.group(1).strip().strip("\"'") if match else ""

    if not data_dir:
        return False

    os.environ[DATA_DIR_ENV_VAR] = data_dir

    return True


def get_data_dir() -> Path:
    env_value = os.environ.get(DATA_DIR_ENV_VAR)

    if env_value:
        return Path(env_value)

    return REPO_ROOT / "data"


def data_path(relative_path: str) -> Path:
    return get_data_dir() / relative_path
