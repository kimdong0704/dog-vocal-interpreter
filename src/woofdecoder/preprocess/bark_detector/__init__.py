"""Bark detector: finds dog sounds in audio, one score every 40 ms.

load_bark_detector()           model.py           the pretrained frame_mn10
detect_sed_probs(det, audio)   predict.py         audio -> (num_frames, 6) probabilities
find_dog_events(probs, ...)    events.py          probabilities -> cleaned events
PRETRAINED_SED_DIR             pretrained_sed.py  where the third-party repo is cloned
SED_FRAME_S                    constants.py       0.04 s per frame
"""

from woofdecoder.preprocess.bark_detector.events import (
    EVENT_CLASSES,
    DetectionSettings,
    drop_short_events,
    events_to_frames,
    find_dog_events,
    frames_to_events,
    merge_close_events,
    settings_from_config,
    smooth_probs,
    vocal_score,
)
from woofdecoder.preprocess.bark_detector.model import (
    AUDIOSET_CLASSES_PER_SED,
    BarkDetector,
    load_bark_detector,
)
from woofdecoder.preprocess.bark_detector.predict import (
    detect_sed_probs,
    predict_audioset_probs,
    to_sed_probs,
)
from woofdecoder.preprocess.bark_detector.pretrained_sed import PRETRAINED_SED_DIR

__all__ = [
    "AUDIOSET_CLASSES_PER_SED",
    "EVENT_CLASSES",
    "PRETRAINED_SED_DIR",
    "BarkDetector",
    "DetectionSettings",
    "detect_sed_probs",
    "drop_short_events",
    "events_to_frames",
    "find_dog_events",
    "frames_to_events",
    "load_bark_detector",
    "merge_close_events",
    "predict_audioset_probs",
    "settings_from_config",
    "smooth_probs",
    "to_sed_probs",
    "vocal_score",
]
