"""Tests for locating the data folder and reading it from .env."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from woofdata.path import get_data_dir, load_data_dir_from_env_file


def test_get_data_dir_honors_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: WOOFDECODER_DATA_DIR overrides the default data folder."""
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(tmp_path))
    assert get_data_dir() == tmp_path

    monkeypatch.delenv("WOOFDECODER_DATA_DIR")
    assert get_data_dir().name == "data"


def test_env_file_sets_only_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: The .env loader sets WOOFDECODER_DATA_DIR but never a token."""
    env_path = tmp_path / ".env"
    env_path.write_text("HF_TOKEN=secret\nWOOFDECODER_DATA_DIR=D:/woof-data\n")
    monkeypatch.delenv("WOOFDECODER_DATA_DIR", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    loaded = load_data_dir_from_env_file(env_path)

    assert loaded
    assert os.environ["WOOFDECODER_DATA_DIR"] == "D:/woof-data"
    assert "HF_TOKEN" not in os.environ


def test_env_file_does_not_override_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test: A WOOFDECODER_DATA_DIR already set in the terminal wins over .env."""
    env_path = tmp_path / ".env"
    env_path.write_text("WOOFDECODER_DATA_DIR=D:/woof-data\n")
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", "terminal")

    assert load_data_dir_from_env_file(env_path) is False
    assert os.environ["WOOFDECODER_DATA_DIR"] == "terminal"
