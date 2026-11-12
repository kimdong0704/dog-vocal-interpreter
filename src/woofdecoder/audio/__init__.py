"""Audio branch: bark segment -> embedding -> category probabilities."""

from woofdecoder.audio.branch import AudioBranch, load_heads, to_categories_order
from woofdecoder.audio.encoder import (
    AudioEncoder,
    EncoderSettings,
    check_audio,
    encoder_from_config,
    load_audio_encoder,
    split_into_chunks,
)
from woofdecoder.audio.features import Layer, check_layer, select_layer
from woofdecoder.audio.head import (
    EmotionNudge,
    LinearHead,
    emotion_input_names,
    emotion_inputs,
    softmax,
)

__all__ = [
    "AudioBranch",
    "AudioEncoder",
    "EmotionNudge",
    "EncoderSettings",
    "Layer",
    "LinearHead",
    "check_audio",
    "check_layer",
    "emotion_input_names",
    "emotion_inputs",
    "encoder_from_config",
    "load_audio_encoder",
    "load_heads",
    "select_layer",
    "softmax",
    "split_into_chunks",
    "to_categories_order",
]
