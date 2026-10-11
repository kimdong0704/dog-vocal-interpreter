"""Tests for the dataset status tables. None of these touch the network."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from woofdecoder.data.download import load_datasets
from woofdecoder.data.status import (
    credential_status,
    dataset_table,
    disk_check,
    folder_stats,
    needed_credentials,
    results_table,
)

ENV_FILE_WITH_VALUES: str = (
    "HF_TOKEN=None\nGITHUB_TOKEN=\nZENODO_TOKEN=\nKAGGLE_USERNAME=someone\n"
    "KAGGLE_KEY=\nROBOFLOW_API_KEY=abc123\n"
)


def write_file(path: Path, size_bytes: int) -> None:
    """Create path (and its folders) holding size_bytes zero bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(size_bytes))


def dataset_by_id(dataset_id: str) -> dict:
    """Return the configs/datasets.yaml entry with this id."""
    for dataset in load_datasets():
        if dataset["id"] == dataset_id:
            return dataset
    raise KeyError(dataset_id)


def test_every_dataset_has_size_and_license_url() -> None:
    """Test: Every dataset lists an expected size and a license URL."""
    for dataset in load_datasets():
        assert dataset["approx_size_gb"] > 0
        assert dataset["license_url"].startswith("http")


def test_credential_status_never_shows_values(tmp_path: Path) -> None:
    """Test: Credentials show only set/missing; empty and "None" count as missing."""
    env_path = tmp_path / ".env"
    env_path.write_text(ENV_FILE_WITH_VALUES)

    status_by_name = credential_status(env_path)

    assert status_by_name == {
        "KAGGLE_USERNAME": "set",
        "KAGGLE_KEY": "missing",
        "ROBOFLOW_API_KEY": "set",
        "HF_TOKEN": "missing",
    }


def test_credential_status_without_env_file(tmp_path: Path) -> None:
    """Test: Without a .env file every credential is missing."""
    status_by_name = credential_status(tmp_path / ".env")

    for status in status_by_name.values():
        assert status == "missing"


def test_needed_credentials() -> None:
    """Test: Kaggle and Roboflow need their keys; Hugging Face's token is optional; ESC-50 none."""
    assert needed_credentials(dataset_by_id("stanfordextra")) == "KAGGLE_USERNAME, KAGGLE_KEY"
    assert needed_credentials(dataset_by_id("rf_leash")) == "ROBOFLOW_API_KEY"
    assert needed_credentials(dataset_by_id("act_env")) == "HF_TOKEN (optional)"
    assert needed_credentials(dataset_by_id("esc50")) == ""


def test_folder_stats_skips_cache_and_resume_state(tmp_path: Path) -> None:
    """Test: .cache folders and .part files are not counted as dataset files."""
    write_file(tmp_path / "a.zip", 100)
    write_file(tmp_path / "sub" / "b.csv", 50)
    write_file(tmp_path / ".cache" / "c.lock", 1000)
    write_file(tmp_path / "d.zip.part", 1000)

    stats = folder_stats(tmp_path)

    assert stats.file_count == 2
    assert stats.total_bytes == 150
    assert stats.partial_count == 1


def test_dataset_table_marks_downloads_and_licenses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test: A finished folder is downloaded; a folder with a .part file is not."""
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(tmp_path))
    write_file(tmp_path / "raw" / "esc50" / "ESC-50.zip", 10)
    write_file(tmp_path / "raw" / "act_env" / "split1.zip.part", 10)
    write_file(tmp_path / "licenses" / "esc50_license.md", 10)

    table = dataset_table().set_index("id")

    assert table.loc["esc50", "downloaded"] == "yes"
    assert table.loc["esc50", "license_file"] == "yes"
    assert table.loc["act_env", "downloaded"] == "no"
    assert table.loc["act_env", "license_file"] == "no"
    assert len(table) == len(load_datasets())


def test_results_table_shows_missing_license_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test: A result without a saved license shows ✗ missing and the license URL."""
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(tmp_path))
    write_file(tmp_path / "raw" / "esc50" / "ESC-50.zip", 10)
    result = SimpleNamespace(request=SimpleNamespace(dest="esc50"), status="skipped", reason="")

    row = results_table([result]).iloc[0]

    assert row["status"] == "skipped"
    assert row["files"] == 1
    assert row["license"] == f"✗ missing ({dataset_by_id('esc50')['license_url']})"


def test_disk_check_flags_small_missing_and_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test: Folders under 90% of the expected size, absent or not in the yaml are flagged."""
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(tmp_path))
    write_file(tmp_path / "raw" / "esc50" / "ESC-50.zip", 1)  # far below the expected 0.6 GB
    write_file(tmp_path / "raw" / "extra_folder" / "x.bin", 1)
    write_file(tmp_path / "raw" / ".cache" / "x.bin", 1)

    table = disk_check().set_index("folder")

    assert table.loc["esc50", "flag"] == "too small"
    assert table.loc["act_env", "flag"] == "missing"
    assert table.loc["extra_folder", "flag"] == "not in datasets.yaml"
    assert ".cache" not in table.index
