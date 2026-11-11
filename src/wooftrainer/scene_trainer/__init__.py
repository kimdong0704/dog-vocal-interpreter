"""CLIP segment probability and embedding cache for fusion training."""

from wooftrainer.scene_trainer.embeddings import (
    cache_scene_segments,
    default_scene_cache,
    load_scene_cache,
    save_scene_cache,
)

__all__ = ["cache_scene_segments", "default_scene_cache", "load_scene_cache", "save_scene_cache"]
