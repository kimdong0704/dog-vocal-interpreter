"""PyTorch Dataset over one task and split, built on woofdata's task rows and loaders.

Works with DataLoader(num_workers=2) on Windows: the dataset holds only the task name,
the rows DataFrame and plain settings, so it pickles into the worker processes. Files
are opened per sample, and each worker fills its own label-file caches.
The default collate function does not know the sample dataclasses: pass a transform
that turns a sample into tensors, or a collate_fn.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch.utils.data import Dataset

from woofdata.loader import load_sample
from woofdata.tasks import task_rows


class TaskDataset(Dataset):
    """The samples of task_rows(task, split, eval_only), loaded with load_sample.

    Args:
        task: A TASKS key, e.g. "pose".
        split: "train", "val" or "test" (see task_rows).
        transform: Called on each sample dataclass; must be picklable for workers on
            Windows (a module-level function or class, not a lambda).
        eval_only: Keep only eval_allowed rows.
        max_seconds: Audio length cap; None uses the task's default.
    """

    def __init__(
        self,
        task: str,
        split: str,
        transform: Callable[[Any], Any] | None = None,
        eval_only: bool = False,
        max_seconds: float | None = None,
    ) -> None:
        self.task = task
        self.rows = task_rows(task, split, eval_only)
        self.transform = transform
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.rows)

    def __getitem__(self, index: int) -> Any:
        """Return sample number index, or transform(sample) if a transform was given."""
        sample = load_sample(self.task, self.rows.iloc[index], self.max_seconds)
        if self.transform is None:
            return sample
        return self.transform(sample)
