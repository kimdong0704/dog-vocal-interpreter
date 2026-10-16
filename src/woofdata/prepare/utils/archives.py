"""Extract only the wanted members of a zip or tar, safely and the same on every OS."""

from __future__ import annotations

import fnmatch
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO

from woofdata.utils_fileio import to_posix_path


def list_archive_members(archive_path: Path) -> dict[str, int]:
    """Return {member name: size in bytes} for every file (not folder) in a zip or tar."""
    member_sizes: dict[str, int] = {}
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zip_archive:
            for info in zip_archive.infolist():
                if not info.is_dir():
                    member_sizes[info.filename] = info.file_size
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tar_archive:
            for member in tar_archive.getmembers():
                if member.isfile():
                    member_sizes[member.name] = member.size
    else:
        raise ValueError(f"{archive_path} is neither a zip nor a tar archive")
    return member_sizes


def select_members(member_names: list[str], wanted: list[str] | str) -> list[str]:
    """Pick members by exact names (list) or one glob pattern (str), sorted.

    Patterns match case-sensitively on every OS (fnmatchcase), so results are the same
    on Windows and macOS.

    Raises:
        ValueError: If a named member is missing, or the pattern matches nothing.
    """
    selected: list[str] = []
    if isinstance(wanted, str):
        for member_name in member_names:
            if fnmatch.fnmatchcase(member_name, wanted):
                selected.append(member_name)
        if not selected:
            raise ValueError(f"no archive member matches {wanted!r}")
    else:
        for member_name in wanted:
            if member_name not in member_names:
                raise ValueError(f"archive has no member {member_name!r}")
            selected.append(member_name)
    return sorted(selected)


def safe_member_path(out_dir: Path, member_name: str) -> Path:
    """Return where member_name goes inside out_dir; refuse names that would escape it."""
    member_path = PurePosixPath(to_posix_path(member_name))
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"unsafe archive member path: {member_name}")
    return out_dir.joinpath(*member_path.parts)


def open_archive(archive_path: Path) -> zipfile.ZipFile | tarfile.TarFile:
    """Open a zip or tar (use it in a with statement)."""
    if zipfile.is_zipfile(archive_path):
        return zipfile.ZipFile(archive_path)
    return tarfile.open(archive_path)


def open_member(archive: zipfile.ZipFile | tarfile.TarFile, member_name: str) -> IO[bytes]:
    """Open one member of an already opened zip or tar for reading."""
    if isinstance(archive, zipfile.ZipFile):
        return archive.open(member_name)
    member_file = archive.extractfile(member_name)
    if member_file is None:
        raise ValueError(f"{member_name} is not a regular file")
    return member_file


def extract_members(archive_path: Path, wanted: list[str] | str, out_dir: Path) -> list[Path]:
    """Extract only the wanted members of a zip or tar into out_dir, keeping their folders.

    Members already extracted with the right size are skipped, so reruns are fast.

    Args:
        archive_path: The .zip, .tar or .tar.gz file.
        wanted: Exact member names, or one glob pattern such as "Images/*.jpg".
        out_dir: Folder to extract into.

    Returns:
        The extracted file paths, sorted by member name.
    """
    member_sizes = list_archive_members(archive_path)
    selected = select_members(sorted(member_sizes), wanted)
    extracted_paths: list[Path] = []
    with open_archive(archive_path) as archive:
        for member_name in selected:
            target_path = safe_member_path(out_dir, member_name)
            extracted_paths.append(target_path)
            if target_path.is_file() and target_path.stat().st_size == member_sizes[member_name]:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                open_member(archive, member_name) as source_file,
                open(target_path, "wb") as target_file,
            ):
                shutil.copyfileobj(source_file, target_file)
    return extracted_paths
