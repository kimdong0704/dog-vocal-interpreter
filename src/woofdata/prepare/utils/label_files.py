"""Write a source's sed_classes.json, events.csv and labels.csv into processed/<id>/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from woofdata.fileio import write_csv, write_json
from woofdata.settings import EVENTS_FILENAME, LABELS_FILENAME, SED_CLASSES_FILENAME
from woofdecoder.taxonomy import SED_CLASSES

# One row per annotated event; sed_class is "" when the label maps to no SED class.
EVENT_COLUMNS: list[str] = ["clip_id", "start_s", "end_s", "sed_class", "original_label"]


def write_sed_classes(out_dir: Path, labelled_classes: list[str]) -> None:
    """Write out_dir/sed_classes.json: SED_CLASSES and whether this source labels each one.

    Evaluation masks the classes a source does not label, so a missing pant label in
    AudioSet is not counted as "no pant here".
    """
    labelled: dict[str, bool] = {}
    for class_name in SED_CLASSES:
        labelled[class_name] = class_name in labelled_classes
    write_json(out_dir / SED_CLASSES_FILENAME, {"classes": SED_CLASSES, "labelled": labelled})


def write_events(out_dir: Path, events: list[dict[str, Any]]) -> None:
    """Write out_dir/events.csv (EVENT_COLUMNS), sorted by every column so it is the same
    on every machine."""
    frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    frame = frame.sort_values(EVENT_COLUMNS, kind="stable")
    write_csv(frame, out_dir / EVENTS_FILENAME)


def write_labels(out_dir: Path, label_rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write out_dir/labels.csv with the given columns (clip_id first), sorted by clip_id."""
    frame = pd.DataFrame(label_rows, columns=columns)
    frame = frame.sort_values("clip_id", kind="stable")
    write_csv(frame, out_dir / LABELS_FILENAME)
