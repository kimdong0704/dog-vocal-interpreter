"""Shared runtime constants used by the app (not tuned by notebooks).

Edit values here, not in the modules that use them. Settings decided by notebooks
(thresholds, layer, heads_dir, scene temperature, ...) live only in configs/pipeline.yaml;
category and sound-class labels live in woofdecoder/taxonomy.py.
"""

from __future__ import annotations

from woofdecoder.config import load_config
from woofdecoder.taxonomy import CATEGORIES

# --- Read from configs/pipeline.yaml (edit the YAML, not these lines) ---

# Sample rate every audio array is converted to (media.sample_rate); the BEATs encoder and
# the bark detector were trained at 16 kHz, so keep it at 16000.
TARGET_SAMPLE_RATE: int = load_config()["media"]["sample_rate"]


# --- Changing these invalidates saved models/caches: retrain/rebuild after editing ---

# Branch order of the fusion inputs and of the per-branch caches; changing it changes
# FEATURE_NAMES -> re-run notebook 50 (branch-output cache) + 51 (fusion model).
BRANCH_NAMES: tuple[str, ...] = ("audio", "body", "scene")
# Names of the 18 fusion inputs: centred log-probs per branch and category, then a present
# flag per branch; saved in fusion_info.json and checked on load -> re-run notebook 51.
FEATURE_NAMES: list[str] = [
    *(f"{branch}:{category}" for branch in BRANCH_NAMES for category in CATEGORIES),
    *(f"{branch}:present" for branch in BRANCH_NAMES),
]
# Probabilities are clipped to this before log() in the fusion inputs; changes the fusion
# model's inputs -> re-run notebook 51.
PROB_FLOOR: float = 1e-6

# File names of the saved fusion model inside its folder; renaming orphans saved models
# -> re-run notebook 51 (or rename the files in models/fusion).
FUSION_HEAD_FILENAME: str = "fusion_head.npz"
FUSION_INFO_FILENAME: str = "fusion_info.json"

# Encoder / layer labels stored in the fusion LinearHead and checked on load -> re-run notebook 51.
FUSION_ENCODER: str = "fusion_v1"
# Layer choice meaning "average all encoder layers"; stored in saved heads and checked on
# load -> re-run notebooks 23 + 51.
LAYER_MEAN: str = "mean"
FUSION_LAYER: str = LAYER_MEAN

# Version written into every saved LinearHead / EmotionNudge .npz; load() rejects other
# versions -> re-run notebooks 23 (audio heads) + 51 (fusion model).
HEAD_FORMAT_VERSION: int = 1
# File names of the four audio heads inside branches.audio.heads_dir; renaming orphans
# saved heads -> re-run notebook 23.
SITUATION_HEAD_FILENAME: str = "situation_head.npz"
AROUSAL_HEAD_FILENAME: str = "arousal_head.npz"
VALENCE_HEAD_FILENAME: str = "valence_head.npz"
NUDGE_FILENAME: str = "emotion_nudge.npz"

# Bark detector (sound event detection) timing, fixed by the pretrained model: do not change.
# The prepared bark-detector targets use the same frames -> re-run data preparation for
# audioset_strong_eval, esc50 and playpant.
SED_CHUNK_S: int = 10  # seconds of audio per model call; the model was trained on 10 s pieces
SED_FRAMES_PER_CHUNK: int = 250  # output frames per chunk
SED_CHUNK_SAMPLES: int = SED_CHUNK_S * TARGET_SAMPLE_RATE  # 160,000 samples per chunk
SED_SAMPLES_PER_FRAME: int = SED_CHUNK_SAMPLES // SED_FRAMES_PER_CHUNK  # 640 samples
SED_FRAME_S: float = SED_SAMPLES_PER_FRAME / TARGET_SAMPLE_RATE  # 0.04 s per frame

# soxr quality used whenever audio is resampled to TARGET_SAMPLE_RATE; changes prepared WAVs
# -> re-run data preparation for every audio source, then rebuild the audio embedding caches.
RESAMPLE_QUALITY: str = "HQ"


# --- Safe to change: no saved file depends on these ---

# Slack allowed when comparing segment / frame times in seconds (float rounding).
TIME_TOLERANCE_S: float = 1e-3
