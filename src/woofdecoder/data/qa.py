"""Quality checks on the prepared data, run from the "Data QA" section of notebook 00.

Each function returns a DataFrame or saves a PNG under outputs/qa/ and returns its path.
Nothing here writes to the data folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from woofdecoder.config import REPO_ROOT
from woofdecoder.data.loaders import crop_to_box, data_path, load_audio_segment, load_sample
from woofdecoder.data.prepare.common import TARGET_SAMPLE_RATE, WAV_SUBTYPE
from woofdecoder.data.prepare.fingerprints import check_fingerprints
from woofdecoder.data.prepare.roboflow import UNIFIED_CATEGORIES
from woofdecoder.data.splits import check_splits
from woofdecoder.data.tasks import TASKS, read_manifest
from woofdecoder.sed_classes import SED_CLASSES

QA_DIR: Path = REPO_ROOT / "outputs" / "qa"
FIGURE_DPI: int = 100
# A window quieter than -60 dBFS RMS is flagged as silent.
SILENCE_RMS: float = 1e-3
SED_SOURCES: tuple[str, ...] = ("playpant", "audioset_strong_eval")
SPECTROGRAM_NFFT: int = 512  # 32 ms at 16 kHz
SPECTROGRAM_HOP: int = 160  # 10 ms
GRID_COLUMNS: int = 6
KEYPOINT_COLUMNS: int = 10
TILE_INCHES: float = 2.2
# Colors in the fixed order of the shared categorical palette. Boxes and points are also
# named (labels or a legend), so color is never the only cue.
PALETTE: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
KEYPOINT_GROUPS: list[str] = ["left legs", "right legs", "tail", "head"]
LABEL_TEXT_COLOR: str = "white"


# --- Shared helpers ---


def save_figure(figure: Figure, file_name: str) -> Path:
    """Save figure as outputs/qa/<file_name> and return the path."""
    QA_DIR.mkdir(parents=True, exist_ok=True)
    path = QA_DIR / file_name
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    return path


def sample_rows(sources: tuple[str, ...] | list[str], n: int, seed: int) -> list[dict[str, Any]]:
    """Return n random manifest rows (all splits) of the given sources, as dicts."""
    manifest = read_manifest()
    rows = manifest[manifest["source"].isin(sources)]
    picked = rows.sample(n=min(n, len(rows)), random_state=seed)
    return picked.to_dict("records")


def image_grid(num_images: int, columns: int) -> tuple[Figure, list[Axes]]:
    """Return a figure with one axes per image (columns per row), axes frames hidden."""
    num_rows = max(1, -(-num_images // columns))  # ceiling division
    figure = Figure(figsize=(columns * TILE_INCHES, num_rows * TILE_INCHES))
    axes_grid = figure.subplots(num_rows, columns, squeeze=False)
    axes_list: list[Axes] = []
    for axes in axes_grid.flat:
        axes.set_axis_off()
        axes_list.append(axes)
    return figure, axes_list[:num_images]


def audio_rows() -> pd.DataFrame:
    """Return the manifest rows of every audio source."""
    manifest = read_manifest()
    return manifest[manifest["modality"] == "audio"]


# --- Tables ---


def qa_summary() -> pd.DataFrame:
    """Print the split-leakage check and the fingerprint check; return rows per source
    per split (with totals)."""
    manifest = read_manifest()
    try:
        check_splits(manifest)
        print("check_splits: OK")
    except ValueError as error:
        print(f"check_splits: FAILED\n{error}")
    differences = check_fingerprints()
    if differences:
        print("fingerprints: DIFFER")
        for difference in differences:
            print(f"  {difference}")
    else:
        print("fingerprints: match")
    return pd.crosstab(manifest["source"], manifest["split"], margins=True, margins_name="total")


def file_format_problem(file_path: str) -> tuple[str, float]:
    """Return (problem text or "", duration in seconds) from one WAV's header."""
    try:
        info = soundfile.info(data_path(file_path))
    except soundfile.LibsndfileError as error:
        return f"cannot read: {error}", 0.0
    duration_s = info.frames / info.samplerate
    if info.samplerate != TARGET_SAMPLE_RATE or info.channels != 1 or info.subtype != WAV_SUBTYPE:
        return f"{info.samplerate} Hz, {info.channels} channels, {info.subtype}", duration_s
    return "", duration_s


def qa_audio_format() -> pd.DataFrame:
    """Check every audio file's header (16 kHz, mono, 16-bit) and that every manifest
    window starts inside its file. Prints one summary line per source.

    Returns:
        One row per problem (source, clip_id, file_path, problem); empty if all pass.
    """
    rows = audio_rows()
    problems: list[dict[str, str]] = []
    durations: dict[str, float] = {}
    files = rows[["source", "file_path"]].drop_duplicates()
    for source, file_path in files.itertuples(index=False):
        problem, durations[file_path] = file_format_problem(file_path)
        if problem:
            problems.append(
                {"source": source, "clip_id": "", "file_path": file_path, "problem": problem}
            )
    window_columns = ["source", "clip_id", "file_path", "start_s"]
    for source, clip_id, file_path, start_s in rows[window_columns].itertuples(index=False):
        if not 0 <= start_s < durations[file_path]:
            problem = f"window starts at {start_s} s, file is {durations[file_path]:.3f} s"
            problems.append(
                {"source": source, "clip_id": clip_id, "file_path": file_path, "problem": problem}
            )

    problem_frame = pd.DataFrame(problems, columns=["source", "clip_id", "file_path", "problem"])
    for source in sorted(rows["source"].unique()):
        num_files = int((files["source"] == source).sum())
        num_windows = int((rows["source"] == source).sum())
        num_problems = int((problem_frame["source"] == source).sum())
        print(f"{source}: {num_files} files, {num_windows} windows, {num_problems} problems")
    return problem_frame


def qa_audio_silence(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Load n random windows per audio source and flag those with RMS below SILENCE_RMS.

    Returns:
        One row per window checked (source, clip_id, seconds, rms, silent), quietest first.
    """
    records: list[dict[str, Any]] = []
    for source in sorted(audio_rows()["source"].unique()):
        for row in sample_rows([source], n, seed):
            audio = load_audio_segment(row)  # shape (num_samples,)
            rms = float(np.sqrt(np.mean(np.square(audio))))
            records.append(
                {
                    "source": source,
                    "clip_id": row["clip_id"],
                    "seconds": len(audio) / TARGET_SAMPLE_RATE,
                    "rms": rms,
                    "silent": rms < SILENCE_RMS,
                }
            )
    results = pd.DataFrame(records)
    for source, source_results in results.groupby("source"):
        num_silent = int(source_results["silent"].sum())
        quietest = source_results["rms"].min()
        summary = f"{len(source_results)} checked, {num_silent} below RMS {SILENCE_RMS}"
        print(f"{source}: {summary}, quietest RMS {quietest:.5f}")
    return results.sort_values("rms").reset_index(drop=True)


# --- Figures ---


def windows_with_events(source: str, n: int, seed: int) -> list[Any]:
    """Return up to n bark_detector samples of source whose targets contain an event,
    taken in a random order (seed)."""
    picked: list[Any] = []
    for row in sample_rows([source], len(read_manifest()), seed):
        sample = load_sample("bark_detector", row)
        if sample.targets.any():
            picked.append(sample)
        if len(picked) == n:
            break
    return picked


def draw_sed_example(spectrogram_axes: Axes, targets_axes: Axes, sample: Any) -> None:
    """Draw one window's spectrogram and, underneath, its frame targets per SED class."""
    window_s = len(sample.audio) / TARGET_SAMPLE_RATE
    with np.errstate(divide="ignore"):  # zero padding past the end of a clip has log10(0)
        spectrogram_axes.specgram(
            sample.audio,
            NFFT=SPECTROGRAM_NFFT,
            Fs=TARGET_SAMPLE_RATE,
            noverlap=SPECTROGRAM_NFFT - SPECTROGRAM_HOP,
            cmap="magma",
        )
    spectrogram_axes.set_title(sample.clip_id, fontsize=9)
    spectrogram_axes.set_ylabel("Hz")
    num_classes = len(SED_CLASSES)
    targets_axes.imshow(
        sample.targets.T,  # shape (num_classes, num_frames)
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(0, window_s, num_classes - 0.5, -0.5),
    )
    class_labels: list[str] = []
    for class_name, labelled in zip(SED_CLASSES, sample.mask, strict=True):
        if labelled:
            class_labels.append(class_name)
        else:
            class_labels.append(f"{class_name} (not labelled)")
    targets_axes.set_yticks(range(num_classes), class_labels, fontsize=8)
    targets_axes.set_xlabel("seconds")


def qa_sed_examples(n: int = 3, seed: int = 0) -> Path:
    """Save sed_examples.png: n windows with events from each SED_SOURCES source, each a
    spectrogram with its frame targets drawn underneath."""
    samples: list[Any] = []
    for source in SED_SOURCES:
        samples.extend(windows_with_events(source, n, seed))
    figure = Figure(figsize=(12, 4.2 * len(samples)), layout="constrained")
    height_ratios: list[float] = []
    for _ in samples:
        height_ratios.extend([3.0, 1.2])
    axes_grid = figure.subplots(len(height_ratios), 1, height_ratios=height_ratios)
    for position, sample in enumerate(samples):
        spectrogram_axes = axes_grid[2 * position]
        targets_axes = axes_grid[2 * position + 1]
        targets_axes.sharex(spectrogram_axes)
        draw_sed_example(spectrogram_axes, targets_axes, sample)
    return save_figure(figure, "sed_examples.png")


def keypoint_group(name: str) -> int:
    """Return the KEYPOINT_GROUPS position of a keypoint name (left legs ... head)."""
    is_leg = "front" in name or "rear" in name
    if is_leg and name.startswith("left_"):
        return 0
    if is_leg and name.startswith("right_"):
        return 1
    if name.startswith("tail"):
        return 2
    return 3


def qa_keypoints(n: int = 50, seed: int = 0) -> Path:
    """Save keypoints.png: n random pose crops with their labelled keypoints drawn."""
    rows = sample_rows(TASKS["pose"].sources, n, seed)
    figure, axes_list = image_grid(len(rows), KEYPOINT_COLUMNS)
    for axes, row in zip(axes_list, rows, strict=True):
        sample = load_sample("pose", row)
        crop, keypoints = crop_to_box(sample.image, sample.box, keypoints=sample.keypoints)
        axes.imshow(crop)
        for name, (x, y, visibility) in zip(sample.keypoint_names, keypoints, strict=True):
            if visibility > 0:
                color = PALETTE[keypoint_group(name)]
                axes.plot(x, y, "o", markersize=4, color=color, markeredgecolor="white")
        axes.set_title(row["clip_id"].split("/")[-1], fontsize=7)
    handles: list[Line2D] = []
    for position, group in enumerate(KEYPOINT_GROUPS):
        handles.append(Line2D([], [], marker="o", linestyle="", color=PALETTE[position]))
    figure.legend(handles, KEYPOINT_GROUPS, loc="upper center", ncols=len(KEYPOINT_GROUPS))
    return save_figure(figure, "keypoints.png")


def qa_posture(n: int = 12, seed: int = 0) -> Path:
    """Save posture.png: n random posture crops titled with their posture."""
    rows = sample_rows(TASKS["posture"].sources, n, seed)
    figure, axes_list = image_grid(len(rows), GRID_COLUMNS)
    for axes, row in zip(axes_list, rows, strict=True):
        sample = load_sample("posture", row)
        crop, _ = crop_to_box(sample.image, sample.box)
        axes.imshow(crop)
        axes.set_title(sample.posture, fontsize=10)
    return save_figure(figure, "posture.png")


def draw_boxes(axes: Axes, boxes: np.ndarray, category_ids: np.ndarray) -> None:
    """Draw each box (x, y, w, h) with its unified category name above it."""
    for (x, y, w, h), category_id in zip(boxes, category_ids, strict=True):
        color = PALETTE[category_id - 1]
        axes.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
        label_box = {"facecolor": color, "edgecolor": "none", "pad": 1}
        name = UNIFIED_CATEGORIES[category_id - 1]
        axes.text(x, y, name, color=LABEL_TEXT_COLOR, fontsize=7, va="bottom", bbox=label_box)


def qa_scene(n: int = 12, seed: int = 0) -> Path:
    """Save scene.png: n random images from each scene_objects source with boxes and
    class names."""
    rows: list[dict[str, Any]] = []
    for source in TASKS["scene_objects"].sources:
        rows.extend(sample_rows([source], n, seed))
    figure, axes_list = image_grid(len(rows), GRID_COLUMNS)
    for axes, row in zip(axes_list, rows, strict=True):
        sample = load_sample("scene_objects", row)
        axes.imshow(sample.image)
        draw_boxes(axes, sample.boxes, sample.category_ids)
        axes.set_title(row["source"], fontsize=8)
    return save_figure(figure, "scene.png")
