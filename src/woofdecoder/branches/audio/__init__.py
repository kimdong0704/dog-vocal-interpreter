"""Audio branch: bark segment -> embedding -> subtype probabilities."""
 
from woofdecoder.branches.audio.encoder import (
    SAMPLE_RATE,
    AudioEncoder,
    EncoderSettings,
    check_audio,
    load_audio_encoder,
    split_into_chunks,
    encoder_from_config
)
 
__all__ = [
    "SAMPLE_RATE",
    "AudioEncoder",
    "EncoderSettings",
    "check_audio",
    "load_audio_encoder",
    "split_into_chunks",
    "encoder_from_config"
]