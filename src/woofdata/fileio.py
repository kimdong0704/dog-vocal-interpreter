"""Small file helpers shared by prepare, the manifest and the split files.

Paths written to disk are relative to the data folder with forward slashes, and text
files always use "\n" line endings, so the same data gives the same bytes on any OS.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

import pandas as pd

from woofdata.path import get_data_dir


def to_posix_path(path: str | PurePath) -> str:
    """Return path with forward slashes, whether it was written Windows- or POSIX-style."""
    return PureWindowsPath(str(path)).as_posix()


def relative_data_path(path: str | PurePath, data_dir: str | PurePath | None = None) -> str:
    """Return path relative to the data folder, with forward slashes (for the manifest).

    Raises:
        ValueError: If path is not inside the data folder.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    posix_path = PurePosixPath(to_posix_path(path))
    posix_data_dir = PurePosixPath(to_posix_path(data_dir))
    try:
        relative_path = posix_path.relative_to(posix_data_dir)
    except ValueError:
        raise ValueError(f"{path} is not inside the data folder {data_dir}") from None
    return relative_path.as_posix()


def write_json(path: Path, content: Any) -> None:
    """Write content as sorted, indented JSON with "\\n" line endings (same bytes on any OS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as json_file:
        json.dump(content, json_file, indent=1, sort_keys=True, ensure_ascii=False)
        json_file.write("\n")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write frame as CSV with "\\n" line endings, so the bytes match on every OS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a rows or manifest CSV with every column as text ("" for empty cells)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def text_records(table: pd.DataFrame) -> list[dict[str, str]]:
    """Return table's rows as {column: cell text} dicts, for tables read as text.

    pandas types to_dict("records") as dict[Hashable, Any]; str() states that the columns
    and cells are already text (read_csv above), so it does not change any value.
    """
    records: list[dict[str, str]] = []
    for record in table.to_dict("records"):
        text_record: dict[str, str] = {}
        for column, value in record.items():
            text_record[str(column)] = str(value)
        records.append(text_record)
    return records
