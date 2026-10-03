"""Settings for the dataset downloader (woofdecoder.data.download)."""

from __future__ import annotations

# Read and write downloads in 1 MB pieces, so big files never sit fully in memory.
CHUNK_BYTES: int = 1024 * 1024

# How many datasets download at the same time (the --workers default).
DEFAULT_WORKERS: int = 3

# Seconds to wait for a server to connect, and for the next piece of data to arrive.
CONNECT_TIMEOUT_S: float = 15.0
READ_TIMEOUT_S: float = 120.0

# Written to <data_dir>/raw/, one entry per dataset.
LOG_FILENAME: str = "download_log.json"

# "hf" = Hugging Face dataset repo, "url" = plain files over HTTP.
VALID_KINDS: tuple[str, ...] = ("hf", "url")

# One console line per dataset: name, percent, bar, downloaded/total bytes, speed, status.
PROGRESS_BAR_FORMAT: str = (
    "{desc:<22} {percentage:3.0f}%|{bar:25}| {n_fmt}/{total_fmt} {speed}{postfix}"
)

# How often (seconds) to measure bytes on disk for Hugging Face downloads.
DISK_POLL_INTERVAL_S: float = 0.5
