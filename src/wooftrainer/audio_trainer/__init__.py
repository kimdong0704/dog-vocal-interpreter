"""Audio head training: embedding cache, linear heads, emotion nudge."""

from wooftrainer.audio_trainer.audio_heads import (
    AudioHeads,
    choose_C,
    choose_nudge_l2,
    fit_linear_head,
    fit_temperature,
    save_audio_heads,
    train_audio_heads,
    train_emotion_head,
)
from wooftrainer.audio_trainer.embeddings import attach_labels, cache_embeddings, load_embeddings

__all__ = [
    "AudioHeads",
    "attach_labels",
    "cache_embeddings",
    "choose_C",
    "choose_nudge_l2",
    "fit_linear_head",
    "fit_temperature",
    "load_embeddings",
    "save_audio_heads",
    "train_audio_heads",
    "train_emotion_head",
]
