"""Package boundary test: the app package woofdecoder must not import woofdata or wooftrainer."""

from __future__ import annotations

from pathlib import Path

WOOFDECODER_DIR: Path = Path(__file__).resolve().parents[1] / "src" / "woofdecoder"
FORBIDDEN_PACKAGES: list[str] = ["woofdata", "wooftrainer"]


def is_forbidden_import(line: str) -> bool:
    """Return True if line imports one of FORBIDDEN_PACKAGES (or a submodule of one)."""
    stripped_line = line.strip()
    for package_name in FORBIDDEN_PACKAGES:
        for prefix in ["from ", "import "]:
            for ending in [" ", "."]:
                if stripped_line.startswith(prefix + package_name + ending):
                    return True
            if stripped_line == prefix + package_name:
                return True
    return False


def test_is_forbidden_import_catches_both_forms() -> None:
    assert is_forbidden_import("from woofdata.path import REPO_ROOT")
    assert is_forbidden_import("    import wooftrainer")
    assert is_forbidden_import("import woofdata.loader as loader")
    assert not is_forbidden_import("from woofdecoder.config import REPO_ROOT")
    assert not is_forbidden_import("import woofdatasets")


def test_woofdecoder_does_not_import_woofdata_or_wooftrainer() -> None:
    python_files = sorted(WOOFDECODER_DIR.rglob("*.py"))
    assert len(python_files) > 0, f"no .py files found under {WOOFDECODER_DIR}"

    problems: list[str] = []
    for python_file in python_files:
        lines = python_file.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if is_forbidden_import(line):
                relative_file = python_file.relative_to(WOOFDECODER_DIR.parents[1])
                problems.append(f"{relative_file}:{line_number}: {line.strip()}")

    assert not problems, "woofdecoder must not import woofdata or wooftrainer:\n" + "\n".join(
        problems
    )
