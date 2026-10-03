"""Tests for the dataset downloader

None of these touch the network
"""

from __future__ import annotations

import io
import os
import socket
from pathlib import Path

import pytest

from woofdecoder.config import get_data_dir, load_env_file
from woofdecoder.data.download import main
from woofdecoder.data.download_config import VALID_KINDS
from woofdecoder.data.download_reg import REGISTRY
from woofdecoder.data.download_report import new_progress_bar


def test_registry_entries_are_complete() -> None:
    """Test: Every registry entry has a name, license, license URL and a valid kind."""
    for dataset in REGISTRY:
        assert dataset.name
        assert dataset.license
        assert dataset.license_url.startswith("http")
        assert dataset.kind in VALID_KINDS


def test_registry_names_are_unique() -> None:
    """Test: No two registry entries share a name."""
    names: list[str] = []
    for dataset in REGISTRY:
        names.append(dataset.name)

    assert len(names) == len(set(names))


def test_get_data_dir_honors_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: WOOFDECODER_DATA_DIR overrides the default data folder."""
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(tmp_path))
    assert get_data_dir() == tmp_path

    monkeypatch.delenv("WOOFDECODER_DATA_DIR")
    assert get_data_dir().name == "data"


def test_load_env_file_does_not_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: .env values are loaded, but a value already in the environment wins."""
    pytest.importorskip("dotenv")
    env_path = tmp_path / ".env"
    env_path.write_text("WOOF_TEST_FROM_FILE=file\nWOOF_TEST_ALREADY_SET=file\n")
    monkeypatch.delenv("WOOF_TEST_FROM_FILE", raising=False)
    monkeypatch.setenv("WOOF_TEST_ALREADY_SET", "terminal")

    loaded = load_env_file(env_path)

    assert loaded
    assert os.environ["WOOF_TEST_FROM_FILE"] == "file"
    assert os.environ["WOOF_TEST_ALREADY_SET"] == "terminal"
    monkeypatch.delenv("WOOF_TEST_FROM_FILE")  # do not leak into other tests


def test_load_env_file_without_file(tmp_path: Path) -> None:
    """Test: A missing .env file is fine and reports False."""
    pytest.importorskip("dotenv")
    assert load_env_file(tmp_path / "missing.env") is False


def test_waiting_bar_shows_zero_progress() -> None:
    """Test: A dataset that has not started shows 0%, 0/approx size and 0 speed, no status."""
    dataset = REGISTRY[0]
    progress = new_progress_bar(dataset, position=0)
    progress.fp = io.StringIO()  # keep the test output clean

    line = str(progress)
    progress.close()

    assert line.startswith(dataset.name)
    assert " 0%|" in line
    assert "| 0.00/380M " in line
    assert line.endswith("0.00B/s")


def test_list_runs_without_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test: --list prints every dataset and never opens a connection."""

    def refuse_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("--list tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse_connection)

    exit_code = main(["--list"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    for dataset in REGISTRY:
        assert dataset.name in printed
