"""Audio branch: bark segment -> embedding -> category probabilities."""
 
from woofdecoder.audio.encoder import (
    SAMPLE_RATE,
    AudioEncoder,
    EncoderSettings,
    check_audio,
    encoder_from_config,
    load_audio_encoder,
    split_into_chunks,
)
from woofdecoder.audio.head import (
    AROUSAL_HEAD_FILENAME,
    NUDGE_FILENAME,
    SITUATION_HEAD_FILENAME,
    VALENCE_HEAD_FILENAME,
    EmotionNudge,
    LinearHead,
    emotion_input_names,
    emotion_inputs,
    softmax
)
from woofdecoder.audio.features import LAYER_MEAN, Layer, check_layer, select_layer
from woofdecoder.audio.branch import AudioBranch, load_heads, to_categories_order


__all__ = [
    "LAYER_MEAN",
    "SAMPLE_RATE",
    "AudioEncoder",
    "EmotionNudge",
    "EncoderSettings",
    "Layer",
    "LinearHead",
    "check_audio",
    "check_layer",
    "encoder_from_config",
    "load_audio_encoder",
    "select_layer",
    "softmax",
    "split_into_chunks",
    "emotion_input_names",
    "emotion_inputs",
    "AROUSAL_HEAD_FILENAME",
    "NUDGE_FILENAME",
    "SITUATION_HEAD_FILENAME",
    "VALENCE_HEAD_FILENAME",
    "AudioBranch",
    "load_heads",
    "to_categories_order"
]