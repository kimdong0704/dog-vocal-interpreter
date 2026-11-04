from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from woofdata.path import data_path, load_data_dir_from_env_file
from woofdata.tasks import AudioContextSample, AudioEmotionSample, task_rows
from woofdecoder.branches.audio import AudioEncoder
from wooftrainer.torch_datasets import TaskDataset

CACHE_DIR: str = "cache/audio_embeddings"
MANIFEST_LABEL_COLUMNS: list[str] = ["source", "dog_id", "mapped_category"]

def cache_path(task: str, split: str, encoder_name: str) -> Path:
    load_data_dir_from_env_file()
    return data_path(f"{CACHE_DIR}/{encoder_name}/{task}_{split}.npz")

def sample_labels(sample: AudioContextSample | AudioEmotionSample) -> dict[str, str]:
    if isinstance(sample, AudioContextSample):
        return {"mapped_category": sample.mapped_category or "", **sample.labels}
    if isinstance(sample, AudioEmotionSample):
        return {"arousal": sample.arousal, "valence": sample.valence}
    raise TypeError(f"no embedding labels for {type(sample).__name__}")

def save_cache_file(
    path: Path, clip_ids: list[str], embeddings: list[np.ndarray], labels: list[dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        temporary,
        clip_id=np.array(clip_ids, dtype=str),
        embedding=np.stack(embeddings).astype(np.float16),
        labels_json=np.array([json.dumps(item) for item in labels], dtype=str),
    )
    temporary.replace(path)
 
 
def read_cache_file(path: Path) -> tuple[list[str], list[np.ndarray], list[dict]]:
    with np.load(path) as data:
        clip_ids = data["clip_id"].tolist()
        embeddings = list(data["embedding"])
        labels = [json.loads(text) for text in data["labels_json"]]
    return clip_ids, embeddings, labels
 
def cache_embeddings(
    task: str, split: str, encoder: AudioEncoder, save_every: int = 500, verbose: bool = True
) -> Path:
    """Bring the cache of task_rows(task, split) up to date and return its path.
 
    Starts from the cached clips (an interrupted .partial file first, else the finished
    file), drops clips the task no longer has, and embeds only the missing ones with the
    same embed_waveforms the app uses. Nothing to do -> returns at once.
 
    Raises:
        ValueError: If the task split has no rows, or a clip cannot be embedded.
    """
    path = cache_path(task, split, encoder.name)
    partial_path = path.with_name(path.stem + ".partial.npz")
    dataset = TaskDataset(task, split)
 
    cached_ids: list[str] = []
    cached_embeddings: list[np.ndarray] = []
    cached_labels: list[dict] = []
    if partial_path.is_file():
        cached_ids, cached_embeddings, cached_labels = read_cache_file(partial_path)
    elif path.is_file():
        cached_ids, cached_embeddings, cached_labels = read_cache_file(path)
 
    wanted_ids = set(dataset.rows["clip_id"])
    clip_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    labels: list[dict] = []
    for clip_id, embedding, label in zip(cached_ids, cached_embeddings, cached_labels, strict=True):
        if clip_id in wanted_ids:
            clip_ids.append(clip_id)
            embeddings.append(embedding)
            labels.append(label)
    dropped_count = len(cached_ids) - len(clip_ids)
    done = set(clip_ids)
    missing_count = len(wanted_ids) - len(done)
    if verbose:
        print(
            f"{task} {split}: {len(dataset)} clips, {len(done)} cached, "
            f"{missing_count} to embed, {dropped_count} dropped"
        )
    if missing_count == 0 and dropped_count == 0 and path.is_file():
        partial_path.unlink(missing_ok=True)
        return path
 
    new_since_save = 0
    for index in range(len(dataset)):
        clip_id = dataset.rows.at[index, "clip_id"]
        if clip_id in done:
            continue
        sample = dataset[index]
        try:
            embedding = encoder.embed_waveforms([sample.audio])[0]
        except ValueError as error:
            raise ValueError(f"{task} {split} clip {clip_id}: {error}") from error
        clip_ids.append(clip_id)
        embeddings.append(embedding.astype(np.float16))
        labels.append(sample_labels(sample))
        new_since_save += 1
        if new_since_save == save_every:
            save_cache_file(partial_path, clip_ids, embeddings, labels)
            new_since_save = 0
            if verbose:
                print(f"{len(clip_ids)}/{len(dataset)}")
 
    if not clip_ids:
        raise ValueError(f"{task} {split} has no rows")
    save_cache_file(path, clip_ids, embeddings, labels)
    partial_path.unlink(missing_ok=True)
    return path 
 
def load_embeddings(
    task: str, split: str, encoder_name: str
) -> tuple[np.ndarray, pd.DataFrame]:
    path = cache_path(task, split, encoder_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run cache_embeddings({task!r}, {split!r}, encoder) first"
        )
    clip_ids, embeddings, labels = read_cache_file(path)
    table = pd.DataFrame(labels)
    table.insert(0, "clip_id", clip_ids)
    return np.stack(embeddings).astype(np.float32), table

def attach_labels(table: pd.DataFrame, task: str, split: str) -> pd.DataFrame:
    """Return table with source, dog_id and mapped_category from the current manifest.
 
    The cache's own copies of those columns are replaced, so a re-prepare or relabel can
    never train on stale labels. Other cached columns (arousal, valence, labels.csv fields
    such as num_windows) are kept. Row order follows table, so it still matches the
    embeddings array.
 
    Raises:
        ValueError: If a cached clip is not in the task split (re-run cache_embeddings).
    """
    rows = task_rows(task, split)
    kept_columns: list[str] = []
    for column in table.columns:
        if column not in MANIFEST_LABEL_COLUMNS:
            kept_columns.append(column)
    merged = table[kept_columns].merge(
        rows[["clip_id"] + MANIFEST_LABEL_COLUMNS], on="clip_id", how="left", validate="one_to_one"
    )
    missing = merged["source"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} cached clips are not in {task} {split}; "
            "re-run cache_embeddings"
        )
    return merged