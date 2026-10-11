"""Tests for the console set-up of the download and prepare command lines."""

from __future__ import annotations

import io
import sys

import pytest
from dataset_downloader.logs import TABLE_COLUMNS, print_table

from woofdecoder.data.console import use_utf8_output


def cp949_stream() -> io.TextIOWrapper:
    """Return a text stream like stdout on a Korean Windows console (code page cp949)."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp949")


def summary_rows() -> list[dict[str, str]]:
    """Return one summary-table row whose cells hold "—", which cp949 cannot encode."""
    row: dict[str, str] = {}
    for column in TABLE_COLUMNS:
        row[column] = "—"
    return [row]


def test_summary_table_crashes_on_cp949_without_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test: The problem is real: the downloader's table cannot be printed to cp949 as is."""
    monkeypatch.setattr(sys, "stdout", cp949_stream())

    with pytest.raises(UnicodeEncodeError):
        print_table(summary_rows())


def test_summary_table_prints_on_cp949_after_use_utf8_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test: After use_utf8_output, the same table prints to a cp949 stream without error."""
    stream = cp949_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", cp949_stream())

    use_utf8_output()
    print_table(summary_rows())
    print("prepare line with — and 🐶")

    stream.flush()
    written = stream.buffer.getvalue().decode("utf-8")
    assert "—" in written
    assert stream.encoding == "utf-8"
