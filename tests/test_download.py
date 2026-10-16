"""Tests for the dataset downloader

None of these touch the network
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from dataset_downloader import Downloader
from dataset_downloader.models import DatasetResult, Status

from woofdata.download import downloader
from woofdata.download.datasets import load_datasets
from woofdata.download.downloader import (
    download_all,
    download_one,
    main,
)

COMPLETE_ENV_FILE: str = (
    "HF_TOKEN=\nGITHUB_TOKEN=\nZENODO_TOKEN=\nKAGGLE_USERNAME=\nKAGGLE_KEY=\nROBOFLOW_API_KEY=\n"
)


def all_dataset_ids() -> list[str]:
    """Return every id in configs/datasets.yaml, in file order."""
    ids: list[str] = []
    for dataset in load_datasets():
        ids.append(dataset["id"])
    return ids


def use_fake_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: object) -> None:
    """Point the downloader at a temporary complete .env and data folder, and replace run()."""
    env_path = tmp_path / ".env"
    env_path.write_text(COMPLETE_ENV_FILE)
    monkeypatch.setattr(downloader, "ENV_FILE", env_path)
    # Set in the "terminal", so the repo's real .env is not read.
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Downloader, "run", run)


def queued_dataset_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]
) -> list[str]:
    """Run main(argv) with a fake run() and return the dest of every queued request."""
    queued_ids: list[str] = []

    def recording_run(self: Downloader) -> list[object]:
        for request in self.requests:
            queued_ids.append(request.dest)
        return []

    use_fake_run(monkeypatch, tmp_path, recording_run)
    assert main(argv) == 0
    return queued_ids


def test_dataset_ids_are_unique() -> None:
    """Test: No two datasets share an id."""
    ids = all_dataset_ids()
    assert len(ids) == len(set(ids))


def test_datasets_use_link_or_provider() -> None:
    """Test: Every dataset has a name and exactly one of link or provider (+ args)."""
    for dataset in load_datasets():
        assert dataset["name"]
        assert ("link" in dataset) != ("provider" in dataset)
        if "provider" in dataset:
            assert "args" in dataset


def test_providers_are_downloader_methods() -> None:
    """Test: Every provider names a public Downloader method."""
    for dataset in load_datasets():
        if "provider" in dataset:
            assert not dataset["provider"].startswith("_")
            assert callable(getattr(Downloader, dataset["provider"]))


def test_no_id_queues_every_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: Without an id, every dataset is queued, each into <raw>/<id>."""
    assert queued_dataset_ids(monkeypatch, tmp_path, []) == all_dataset_ids()


def test_one_id_queues_only_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: With an id, only that dataset is queued."""
    assert queued_dataset_ids(monkeypatch, tmp_path, ["act_env"]) == ["act_env"]


def test_unknown_id_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    """Test: An unknown id is bad usage (exit code 2) and the message lists the valid ids."""
    with pytest.raises(SystemExit) as exit_info:
        main(["no_such_dataset"])

    message = capsys.readouterr().err
    assert exit_info.value.code == 2
    for dataset_id in all_dataset_ids():
        assert dataset_id in message


@pytest.mark.parametrize("removed_option", ["--list", "--workers=4", "--data-dir=D:/data"])
def test_removed_options_are_rejected(removed_option: str) -> None:
    """Test: The old --list, --workers and --data-dir options are bad usage (exit code 2)."""
    with pytest.raises(SystemExit) as exit_info:
        main([removed_option])

    assert exit_info.value.code == 2


def test_incomplete_env_file_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: A .env file missing token entries (EnvFileError) exits with code 2."""
    env_path = tmp_path / ".env"
    env_path.write_text("HF_TOKEN=\n")
    monkeypatch.setattr(downloader, "ENV_FILE", env_path)
    monkeypatch.setenv("WOOFDECODER_DATA_DIR", str(tmp_path))

    assert main([]) == 2


def test_interrupt_exits_130(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test: Ctrl+C during the download exits with code 130."""

    def interrupted_run(self: Downloader) -> list[object]:
        raise KeyboardInterrupt

    use_fake_run(monkeypatch, tmp_path, interrupted_run)

    assert main(["act_env"]) == 130


@pytest.mark.parametrize(("status", "expected_code"), [("skipped", 0), ("needs_credentials", 1)])
def test_exit_code_follows_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str, expected_code: int
) -> None:
    """Test: Exit code 0 when every dataset is done, 1 when one needs action."""

    def finished_run(self: Downloader) -> list[object]:
        return [SimpleNamespace(status=status)]

    use_fake_run(monkeypatch, tmp_path, finished_run)

    assert main(["act_env"]) == expected_code


def one_result_per_request_run(self: Downloader) -> list[DatasetResult]:
    """Fake run() that returns one "ok" result per queued request."""
    results: list[DatasetResult] = []
    for request in self.requests:
        results.append(DatasetResult(request, Status.OK))
    return results


def test_download_all_returns_one_result_per_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test: download_all queues every dataset and returns one result for each."""
    use_fake_run(monkeypatch, tmp_path, one_result_per_request_run)

    results = download_all()

    result_ids: list[str] = []
    for result in results:
        result_ids.append(result.request.dest)
    assert result_ids == all_dataset_ids()


def test_download_one_rejects_unknown_id() -> None:
    """Test: download_one raises ValueError for an id not in configs/datasets.yaml."""
    with pytest.raises(ValueError, match="Valid ids"):
        download_one("no_such_dataset")
