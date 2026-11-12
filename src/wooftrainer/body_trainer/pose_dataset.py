"""Body-language training data, as built in notebook 31.

Steps:
1. build_label_table: one row per image with its 5-category label and split group
   (Kaggle Dog Emotion, Roboflow Dog Emotion Videos, self-labelled attention_seeking).
2. add_pose_columns / load_or_build_pose_table: run the pose encoder (YOLO11m-pose) and keep
   only images with exactly 1 dog. The result is cached as <data dir>/cache/body/pose.pkl.
3. features_and_labels: 64 features per kept image (woofdecoder.body.branch.make_features) and
   integer labels in DECODER_CLASSES order.
4. split_by_group: 70 / 15 / 15 split where a video (group) never lands in two splits.

Column names match the notebook's pose.pkl, so an existing cache loads unchanged. Image paths
are stored with forward slashes on every OS; older caches written on Windows are normalised
on load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from woofdata.path import branch_cache_dir, get_data_dir
from woofdecoder.body.branch import (
    BODY_CONFIG,
    DECODER_CLASSES,
    make_features,
    parse_result,
    pick_device,
)
from woofdecoder.config import REPO_ROOT

PATH_COLUMN = "path to image"  # relative to raw_root()
TABLE_COLUMNS: list[str] = [PATH_COLUMN, "source", "original_label", "category", "group"]
POSE_COLUMNS: list[str] = ["n_dogs", "kept", "drop_reason", "box", "keypoints"]

KAGGLE_DIR = Path("dog_emotion_kaggle") / "Dog Emotion"
KAGGLE_MAP: dict[str, str] = {
    "angry": "alert_aggressive",
    "happy": "playful_excited",
    "sad": "distressed_lonely",
    "relaxed": "other",
}

ROBOFLOW_DIR = Path("rf_dog_emotion")
ROBOFLOW_SPLITS: tuple[str, ...] = ("train", "valid", "test")
# Roboflow "Happy" is left out on purpose (notebook 31 only maps these three).
ROBOFLOW_MAP: dict[str, str] = {
    "Aggressive": "alert_aggressive",
    "Fear": "alert_aggressive",
    "Anxiety": "distressed_lonely",
}

SELF_LABELED_DIR = Path("self_labeled_attention_seeking")

LABEL_INDEX: dict[str, int] = {name: index for index, name in enumerate(DECODER_CLASSES)}
BATCH_SIZE = 32  # images per encoder call (larger batches overflowed the buffer)
SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15  # test gets the rest


def raw_root() -> Path:
    return get_data_dir() / "raw"


def default_pose_cache() -> Path:
    return branch_cache_dir("body") / "pose.pkl"


def posix_path(path: str | Path) -> str:
    """Cache key for an image path: forward slashes on every OS (Windows caches used '\\')."""
    return str(path).replace("\\", "/")


# --- 1. Label table ---


def kaggle_rows(root: Path) -> list[list[str]]:
    rows = []
    for folder in sorted((root / KAGGLE_DIR).iterdir()):
        category = KAGGLE_MAP.get(folder.name)
        if category is None or not folder.is_dir():
            continue
        for file in sorted(folder.iterdir()):
            rows.append(
                [
                    file.relative_to(root).as_posix(),
                    "kaggle",
                    folder.name,
                    category,
                    f"kaggle_{file.name}",
                ]
            )
    return rows


def roboflow_video_id(file_name: str) -> str:
    """Frames of one video share this id, e.g. 'YT_ID_220_mp4-12.jpg' -> 'YT_ID_220'."""
    return file_name.split("_mp4")[0].split("_jpg")[0]


def roboflow_rows(root: Path) -> list[list[str]]:
    rows = []
    for split in ROBOFLOW_SPLITS:
        split_dir = root / ROBOFLOW_DIR / split
        with open(split_dir / "_annotations.coco.json", encoding="utf-8") as file:
            coco = json.load(file)
        category_names = {c["id"]: c["name"] for c in coco["categories"]}
        image_labels = {
            a["image_id"]: category_names[a["category_id"]] for a in coco["annotations"]
        }
        for image in coco["images"]:
            label = image_labels.get(image["id"])
            if label not in ROBOFLOW_MAP:
                continue
            path = split_dir / image["file_name"]
            rows.append(
                [
                    path.relative_to(root).as_posix(),
                    "roboflow",
                    label,
                    ROBOFLOW_MAP[label],
                    f"roboflow_{roboflow_video_id(image['file_name'])}",
                ]
            )
    return rows


def self_labeled_rows(root: Path) -> list[list[str]]:
    """labels.csv has columns filename, label, group.

    Groups get a 'self_labeled_' prefix (like 'kaggle_' / 'roboflow_') so they never collide
    with another source's group in split_by_group.
    """
    labels_path = root / SELF_LABELED_DIR / "labels.csv"
    if not labels_path.is_file():
        return []
    table = pd.read_csv(labels_path)
    rows = []
    for record in table.itertuples(index=False):
        path = (SELF_LABELED_DIR / str(record.filename)).as_posix()
        rows.append(
            [path, "self_labeled", record.label, record.label, f"self_labeled_{record.group}"]
        )
    return rows


def build_label_table(root: Path | None = None, include_self_labeled: bool = True) -> pd.DataFrame:
    """One row per image: path (relative to root), source, original_label, category, group."""
    root = root or raw_root()
    rows = kaggle_rows(root) + roboflow_rows(root)
    if include_self_labeled:
        rows += self_labeled_rows(root)
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    unknown = set(table["category"]) - set(DECODER_CLASSES)
    if unknown:
        raise ValueError(f"Unknown categories in label table: {sorted(unknown)}")
    return table


# --- 2. Pose encoder ---


def load_pose_encoder(encoder_path: str | Path | None = None):
    from ultralytics import YOLO  # imported here so the rest of the module works without it

    path = REPO_ROOT / (encoder_path or BODY_CONFIG["pose"])
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found (see configs/pipeline.yaml, branches.body.pose)")
    return YOLO(str(path))


def drop_reason(n_dogs: int) -> str:
    if n_dogs == 0:
        return "no_dog"
    if n_dogs > 1:
        return "multi_dog"
    return ""


def run_pose_encoder(
    encoder, paths: list[Path], device: str | None = None, batch_size: int = BATCH_SIZE
) -> pd.DataFrame:
    """Columns n_dogs, kept, drop_reason, box (4,), keypoints (20, 3); one row per path."""
    device = device or str(pick_device())
    rows = []
    for start in range(0, len(paths), batch_size):
        chunk = [str(p) for p in paths[start : start + batch_size]]
        for result in encoder.predict(source=chunk, device=device, verbose=False):
            parsed = parse_result(result)
            parsed["drop_reason"] = drop_reason(parsed["n_dogs"])
            rows.append(parsed)
    return pd.DataFrame(rows, columns=POSE_COLUMNS)


def add_pose_columns(
    table: pd.DataFrame, encoder, root: Path | None = None, device: str | None = None
) -> pd.DataFrame:
    root = root or raw_root()
    paths = [root / p for p in table[PATH_COLUMN]]
    pose = run_pose_encoder(encoder, paths, device=device)
    return pd.concat([table.reset_index(drop=True), pose], axis=1)


def load_or_build_pose_table(
    cache_path: Path | None = None,
    encoder=None,
    root: Path | None = None,
    device: str | None = None,
    include_self_labeled: bool = True,
) -> pd.DataFrame:
    """Current label table joined with pose results from pose.pkl.

    Only label-table images missing from the cache are encoded; they are appended and the
    cache is rewritten. The encoder is loaded only when something needs encoding.

    The returned rows are exactly the current label table (same order, current source /
    category / group), so images removed or relabelled since the cache was built, and
    self-labelled images when include_self_labeled=False, never leak in from the cache.
    The cache itself keeps every image it has encoded, so toggling sources needs no re-encode.
    """
    root = root or raw_root()
    cache_path = cache_path or default_pose_cache()
    labels = build_label_table(root, include_self_labeled=include_self_labeled)

    cached = pd.DataFrame(columns=[PATH_COLUMN, *POSE_COLUMNS])
    if cache_path.is_file():
        cached = pd.read_pickle(cache_path)
        cached = cached.loc[:, ~cached.columns.duplicated()]  # old caches had duplicate columns
        cached[PATH_COLUMN] = cached[PATH_COLUMN].map(posix_path)
        cached = cached.drop_duplicates(PATH_COLUMN, keep="last").reset_index(drop=True)

    missing = labels[~labels[PATH_COLUMN].isin(set(cached[PATH_COLUMN]))]
    if len(missing):
        print(f"Encoding {len(missing)} images ({len(cached)} already in {cache_path.name})")
        encoder = encoder or load_pose_encoder()
        encoded = add_pose_columns(missing, encoder, root, device)
        cached = pd.concat([cached, encoded], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_pickle(cache_path)

    pose = cached.set_index(PATH_COLUMN)[POSE_COLUMNS]
    return labels.join(pose, on=PATH_COLUMN)


# --- 3. Features ---


def features_and_labels(pose_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns X (n, 64) float32, y (n,) int64 in DECODER_CLASSES order, and the kept rows."""
    kept = pose_table[pose_table["kept"].astype(bool)].reset_index(drop=True)
    K = np.stack(kept["keypoints"].to_numpy())  # (n, 20, 3)
    B = np.stack(kept["box"].to_numpy())  # (n, 4)
    X = make_features(K, B)
    y = kept["category"].map(LABEL_INDEX).to_numpy(dtype=np.int64)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite pose features (zero-size box?)")
    return X, y, kept


# --- 4. Split ---


@dataclass
class PoseSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def summary(self) -> pd.DataFrame:
        rows = {}
        for name in ("train", "val", "test"):
            y = getattr(self, f"y_{name}")
            rows[name] = np.bincount(y, minlength=len(DECODER_CLASSES))
        return pd.DataFrame(rows, index=DECODER_CLASSES).T


def split_by_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = SEED,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> PoseSplits:
    """Shuffle the unique groups and give train/val/test whole groups (same as notebook 31).

    Raises ValueError if any split would be empty (too few groups for the fractions).
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_train = int(train_fraction * len(unique_groups))
    n_val = int(val_fraction * len(unique_groups))
    train_mask = np.isin(groups, unique_groups[:n_train])
    val_mask = np.isin(groups, unique_groups[n_train : n_train + n_val])
    test_mask = np.isin(groups, unique_groups[n_train + n_val :])
    sizes = {"train": train_mask.sum(), "val": val_mask.sum(), "test": test_mask.sum()}
    empty = [name for name, size in sizes.items() if size == 0]
    if empty:
        raise ValueError(
            f"Empty split(s) {empty} from {len(unique_groups)} groups "
            f"(train {train_fraction}, val {val_fraction}); need more groups."
        )
    return PoseSplits(
        X[train_mask],
        y[train_mask],
        X[val_mask],
        y[val_mask],
        X[test_mask],
        y[test_mask],
    )
