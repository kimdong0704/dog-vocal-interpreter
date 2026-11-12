"""Plots shared by the notebooks: one confusion-matrix style for every branch and fusion.

plot_confusion_matrix(true, predicted, CATEGORIES, "Audio, val")
plt.show()
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from woofdecoder.taxonomy import DISPLAY_NAMES

# One colour map for every confusion matrix in the project.
CONFUSION_CMAP: str = "Blues"
# Cells whose colour value is above this share of the colour range get white text.
DARK_CELL_SHARE: float = 0.5
X_TICK_ROTATION_DEGREES: float = 35.0


def confusion_counts(
    true: Sequence[str] | np.ndarray,
    predicted: Sequence[str] | np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    """Count (true, predicted) pairs.

    Args:
        true: The true label of each sample.
        predicted: The predicted label of each sample, same length as true.
        labels: Row and column order. Pairs with a label outside this list are skipped
            (as in sklearn.metrics.confusion_matrix).

    Returns:
        Integer counts, shape (len(labels), len(labels)); rows = true, columns = predicted.
    """
    if len(true) != len(predicted):
        raise ValueError(f"true has {len(true)} labels but predicted has {len(predicted)}")

    position_of_label: dict[str, int] = {}
    for position, label in enumerate(labels):
        position_of_label[label] = position

    counts = np.zeros((len(labels), len(labels)), dtype=np.int64)  # (num_labels, num_labels)
    for true_label, predicted_label in zip(true, predicted, strict=True):
        if true_label not in position_of_label or predicted_label not in position_of_label:
            continue
        counts[position_of_label[true_label], position_of_label[predicted_label]] += 1
    return counts


def row_shares(counts: np.ndarray) -> np.ndarray:
    """Divide each row by its total (recall per class); a row with no samples stays 0.

    Args:
        counts: Shape (num_labels, num_labels), rows = true.

    Returns:
        Float shares, same shape; each non-empty row sums to 1.
    """
    row_totals = counts.sum(axis=1, keepdims=True)  # (num_labels, 1)
    shares = np.zeros(counts.shape, dtype=np.float64)  # (num_labels, num_labels)
    for row in range(counts.shape[0]):
        if row_totals[row, 0] > 0:
            shares[row] = counts[row] / row_totals[row, 0]
    return shares


def tick_names(labels: Sequence[str]) -> list[str]:
    """Display names for category labels; any other label is shown as it is."""
    names: list[str] = []
    for label in labels:
        names.append(DISPLAY_NAMES.get(label, label))
    return names


def plot_confusion_matrix(
    true: Sequence[str] | np.ndarray,
    predicted: Sequence[str] | np.ndarray,
    labels: Sequence[str],
    title: str,
    normalize: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Draw a confusion matrix: rows = true, columns = predicted, both in labels order.

    Args:
        true: The true label of each sample.
        predicted: The predicted label of each sample.
        labels: Row and column order; pass CATEGORIES for categories (ticks then show
            DISPLAY_NAMES).
        title: Axes title, e.g. "Fusion, val".
        normalize: True colours each cell by its row share (recall per class) and writes
            "share\\n(count)"; False colours and writes the raw counts.
        ax: Axes to draw on; None makes a new figure.

    Returns:
        The Axes. The caller calls plt.show().
    """
    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    counts = confusion_counts(true, predicted, labels)  # (num_labels, num_labels)
    if normalize:
        colour_values = row_shares(counts)  # (num_labels, num_labels)
        colour_max = 1.0
        colour_bar_label = "share of true class"
    else:
        colour_values = counts.astype(np.float64)
        colour_max = max(float(counts.max()), 1.0)
        colour_bar_label = "count"

    image = ax.imshow(colour_values, cmap=CONFUSION_CMAP, vmin=0.0, vmax=colour_max)
    ax.figure.colorbar(image, ax=ax, label=colour_bar_label)

    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            if normalize:
                cell_text = f"{colour_values[row, column]:.2f}\n({counts[row, column]})"
            else:
                cell_text = str(counts[row, column])
            is_dark = colour_values[row, column] > DARK_CELL_SHARE * colour_max
            text_colour = "white" if is_dark else "black"
            ax.text(column, row, cell_text, ha="center", va="center", color=text_colour)

    names = tick_names(labels)
    ax.set_xticks(range(len(labels)), names, rotation=X_TICK_ROTATION_DEGREES, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    return ax
