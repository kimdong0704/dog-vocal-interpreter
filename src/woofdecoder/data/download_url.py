"""Download "url" datasets: plain files over HTTP, resumable, with size and md5 checks.

Each file is streamed to <file>.part, renamed to <file> when complete, then verified.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tqdm import tqdm

from woofdecoder.data.download_config import CHUNK_BYTES, CONNECT_TIMEOUT_S, READ_TIMEOUT_S
from woofdecoder.data.download_reg import Dataset, UrlFile
from woofdecoder.data.download_report import start_progress


def compute_md5(path: Path) -> str:
    """Return the md5 hex digest of a file, read in chunks to keep memory low."""
    digest = hashlib.md5()
    with open(path, "rb") as file:
        chunk = file.read(CHUNK_BYTES)
        while chunk:
            digest.update(chunk)
            chunk = file.read(CHUNK_BYTES)
    return digest.hexdigest()


def verify_file(path: Path, url_file: UrlFile) -> str | None:
    """Check a downloaded file's size and md5 when the registry knows them.

    A file that fails is renamed to <file>.bad so it is never used by mistake.

    Args:
        path: The completed file.
        url_file: Registry entry with the expected size and md5.

    Returns:
        The md5 hex digest if it was computed, else None.
    """
    actual_bytes = path.stat().st_size
    actual_md5 = None
    problem = ""
    if url_file.expected_bytes is not None and actual_bytes != url_file.expected_bytes:
        problem = f"size {actual_bytes} bytes, expected {url_file.expected_bytes}"
    elif url_file.md5 is not None:
        actual_md5 = compute_md5(path)
        if actual_md5 != url_file.md5:
            problem = f"md5 {actual_md5}, expected {url_file.md5}"

    if problem:
        bad_path = path.with_name(path.name + ".bad")
        path.replace(bad_path)
        raise ValueError(f"{path.name}: {problem}. Renamed to {bad_path.name}")
    return actual_md5


def file_is_complete(final_path: Path, url_file: UrlFile) -> bool:
    """True if the file is already downloaded (and has the expected size, when known)."""
    if not final_path.exists():
        return False
    if url_file.expected_bytes is None:
        return True
    return final_path.stat().st_size == url_file.expected_bytes


def remote_size(url: str) -> int | None:
    """Ask the server how big a file is, without downloading it.

    Returns:
        The size in bytes, or None if the server does not say.
    """
    import requests  # imported here so --list and the tests work without the "data" extra

    timeout = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
    response = requests.head(url, allow_redirects=True, timeout=timeout)
    content_length = response.headers.get("Content-Length")
    if not response.ok or content_length is None:
        return None
    return int(content_length)


def measure_url_dataset(dataset: Dataset, target_dir: Path) -> tuple[int | None, int]:
    """Work out a dataset's total size and how much of it is already on disk.

    Files whose size the server does not report count as 0. On Zenodo that only happens
    for small text files, so the total stays useful for the big downloads.

    Returns:
        (total_bytes, done_bytes). total_bytes is None if no size is known at all.
    """
    total_bytes = 0
    done_bytes = 0
    for url_file in dataset.files:
        final_path = target_dir / url_file.filename
        if file_is_complete(final_path, url_file):
            file_bytes = final_path.stat().st_size
            done_bytes += file_bytes
        else:
            file_bytes = url_file.expected_bytes
            if file_bytes is None:
                file_bytes = remote_size(url_file.url)
            part_path = target_dir / (url_file.filename + ".part")
            if part_path.exists():
                done_bytes += part_path.stat().st_size

        if file_bytes is not None:
            total_bytes += file_bytes

    if total_bytes == 0:
        return None, done_bytes
    return total_bytes, done_bytes


def stream_to_part(url: str, part_path: Path, progress: tqdm) -> None:
    """Download a URL into a .part file, resuming from what is already there.

    Args:
        url: Source URL.
        part_path: Partial file to create or append to.
        progress: The dataset's progress bar; already counts the existing .part bytes.
    """
    import requests  # imported here so --list and the tests work without the "data" extra

    start_byte = 0
    headers: dict[str, str] = {}
    if part_path.exists():
        start_byte = part_path.stat().st_size
        headers["Range"] = f"bytes={start_byte}-"

    timeout = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        # 416 means the range starts at the end: the .part file is already complete.
        if response.status_code == 416:
            return
        response.raise_for_status()

        # A server that ignores Range answers 200 with the whole file, so start over
        # and take the old .part bytes back off the bar.
        if response.status_code != 206:
            progress.update(-start_byte)
            start_byte = 0

        write_mode = "ab" if start_byte > 0 else "wb"
        with open(part_path, write_mode) as part_file:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                part_file.write(chunk)
                progress.update(len(chunk))


def download_url_file(url_file: UrlFile, target_dir: Path, progress: tqdm) -> str | None:
    """Download and verify one file, skipping it if it is already complete.

    Args:
        url_file: Registry entry for the file.
        target_dir: Folder to save into.
        progress: The dataset's progress bar.

    Returns:
        The md5 hex digest if it was computed, else None.
    """
    final_path = target_dir / url_file.filename
    if file_is_complete(final_path, url_file):
        return None

    part_path = target_dir / (url_file.filename + ".part")
    stream_to_part(url_file.url, part_path, progress)
    part_path.replace(final_path)
    return verify_file(final_path, url_file)


def download_url_dataset(dataset: Dataset, target_dir: Path, progress: tqdm) -> dict[str, str]:
    """Download every file of a "url" dataset.

    Args:
        dataset: Registry entry with kind "url".
        target_dir: Folder to save into.
        progress: The dataset's progress bar.

    Returns:
        A dict mapping filename to md5 for the files whose md5 was computed.
    """
    total_bytes, done_bytes = measure_url_dataset(dataset, target_dir)
    start_progress(progress, total_bytes, done_bytes)

    md5_by_filename: dict[str, str] = {}
    for url_file in dataset.files:
        md5 = download_url_file(url_file, target_dir, progress)
        if md5 is not None:
            md5_by_filename[url_file.filename] = md5

    # Files of unknown size were counted as 0, so end the bar at the real total.
    if progress.total is None or progress.n > progress.total:
        progress.total = progress.n
        progress.refresh()
    return md5_by_filename
