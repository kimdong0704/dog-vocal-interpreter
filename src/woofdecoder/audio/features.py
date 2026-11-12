from __future__ import annotations

import numpy as np

from woofdecoder.constants import LAYER_MEAN

Layer = int | str


def check_layer(layer: Layer, num_layers: int | None = None) -> None:
    if layer == LAYER_MEAN:
        return
    if isinstance(layer, bool) or not isinstance(layer, int):
        # ValueError kept on purpose: callers already handle it (TypeError would change that).
        raise ValueError(f"layer must be an int or {LAYER_MEAN!r}, not {layer!r}")  # noqa: TRY004
    if layer < 0 or (num_layers is not None and layer >= num_layers):
        raise ValueError(f"layer {layer} is out of range for {num_layers} layers")


def select_layer(embeddings: np.ndarray, layer: Layer) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim not in (2, 3):
        raise ValueError(
            f"embeddings must haave shape (num_layers, dim) or (num_clips, num_layers, dim), got {embeddings.shape}"
        )

    check_layer(layer, embeddings.shape[-2])
    if layer == LAYER_MEAN:
        return embeddings.mean(axis=-2)
    return embeddings[..., layer, :]
