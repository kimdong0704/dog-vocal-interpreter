"""Check that no dog or leakage group crosses splits, and save the committed split files.

Splits are always by dog: every row of one dog_id is in the same split. Rows without a
dog id get their own group ("<source>_<clip_id>"), see manifest.ManifestRow.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from woofdecoder.config import REPO_ROOT
from woofdecoder.data.fileio import write_csv, write_json

# Committed to git (an exception in .gitignore), unlike everything else under data/.
SPLITS_DIR: Path = REPO_ROOT / "data" / "splits"
SPLIT_FILENAME: str = "split_v1.json"
DOG_GROUPS_FILENAME: str = "dog_groups.csv"

# Splits a leakage group must not cross; train_only rows never reach val or test.
EVALUATED_SPLITS: tuple[str, ...] = ("train", "val", "test")
# Leakage groups allowed in more than one of train / val / test. Each entry needs a comment
# explaining why the overlap cannot inflate the reported scores.
ALLOWED_CROSS_SPLIT_GROUPS: tuple[str, ...] = ()
MAX_LISTED_PROBLEMS: int = 10


def groups_in_several_splits(manifest: pd.DataFrame, group_column: str) -> dict[str, list[str]]:
    """Return {group: sorted splits} for every group value found in more than one split."""
    splits_by_group = manifest.groupby(group_column)["split"].unique()
    problems: dict[str, list[str]] = {}
    for group, group_splits in splits_by_group.items():
        if len(group_splits) > 1:
            problems[group] = sorted(group_splits)
    return problems


def describe_problems(title: str, problems: dict[str, list[str]]) -> str:
    """Return a readable block listing up to MAX_LISTED_PROBLEMS groups and their splits."""
    lines = [f"{title} ({len(problems)}):"]
    for group in sorted(problems)[:MAX_LISTED_PROBLEMS]:
        lines.append(f"  {group}: {', '.join(problems[group])}")
    if len(problems) > MAX_LISTED_PROBLEMS:
        lines.append(f"  ... and {len(problems) - MAX_LISTED_PROBLEMS} more")
    return "\n".join(lines)


def check_splits(manifest: pd.DataFrame) -> None:
    """Raise ValueError if a dog or a leakage group would leak between splits.

    - A dog_id may appear in only one split (train_only counts as its own split).
    - A leakage_group may appear in only one of train / val / test, unless it is listed
      in ALLOWED_CROSS_SPLIT_GROUPS.
    """
    dog_problems = groups_in_several_splits(manifest, "dog_id")

    evaluated_rows = manifest[manifest["split"].isin(EVALUATED_SPLITS)]
    group_problems: dict[str, list[str]] = {}
    for group, group_splits in groups_in_several_splits(evaluated_rows, "leakage_group").items():
        if group not in ALLOWED_CROSS_SPLIT_GROUPS:
            group_problems[group] = group_splits

    messages: list[str] = []
    if dog_problems:
        messages.append(describe_problems("Dogs in more than one split", dog_problems))
    if group_problems:
        title = "Leakage groups in more than one of train/val/test"
        messages.append(describe_problems(title, group_problems))
    if messages:
        raise ValueError("Split leakage found.\n" + "\n".join(messages))


def save_splits(manifest: pd.DataFrame, splits_dir: Path | None = None) -> None:
    """Write split_v1.json (clip_id -> split) and dog_groups.csv (dog_id, source, split).

    splits_dir defaults to SPLITS_DIR (data/splits/ in the repo).
    """
    if splits_dir is None:
        splits_dir = SPLITS_DIR
    split_by_clip: dict[str, str] = {}
    for clip_id, split in zip(manifest["clip_id"], manifest["split"], strict=True):
        split_by_clip[clip_id] = split
    write_json(splits_dir / SPLIT_FILENAME, split_by_clip)

    dog_groups = manifest[["dog_id", "source", "split"]].drop_duplicates()
    dog_groups = dog_groups.sort_values(["dog_id", "source"]).reset_index(drop=True)
    write_csv(dog_groups, splits_dir / DOG_GROUPS_FILENAME)
