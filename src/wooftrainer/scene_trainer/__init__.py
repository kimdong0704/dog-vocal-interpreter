"""Scene embedding cache for fusion training."""

from wooftrainer.scene_trainer.embeddings import (
    cache_scene_segments,
    load_scene_cache,
    save_scene_cache,
)

__all__ = ["cache_scene_segments", "load_scene_cache", "save_scene_cache"]
