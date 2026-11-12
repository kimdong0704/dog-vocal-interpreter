"""Training and evaluation settings: search grids, folds, seeds, bootstrap repeats, cache names.

Edit values here, not in the modules or notebooks that use them. Runtime settings chosen by
the notebooks (layer, heads_dir, thresholds, ...) live only in configs/pipeline.yaml.
"""

from __future__ import annotations

from woofdata.settings import CACHE_DIRNAME

# --- Changing these invalidates saved models/caches: retrain/rebuild after editing ---

# Branch-output cache for fusion (<data dir>/cache/fusion); renaming orphans the cache
# -> re-run notebook 50 (slow: every branch on every video).
FUSION_CACHE_DIR: str = f"{CACHE_DIRNAME}/fusion"
FUSION_SEGMENTS_FILENAME: str = "branch_outputs.csv"  # one row per segment: all branch probs
FUSION_VIDEOS_FILENAME: str = "videos.csv"  # videos already processed (resume list)
FUSION_CACHE_INFO_FILENAME: str = "info.json"  # model versions the cache was built with
# Saved fusion model folder, relative to the repo root -> re-run notebook 51.
FUSION_MODEL_DIR: str = "models/fusion"

# Zero-shot bark-detector scores (<data dir>/cache/bark_detector) -> re-run notebook 12.
BARK_DETECTOR_CACHE_DIR: str = f"{CACHE_DIRNAME}/bark_detector"
# Pose-encoder results in <data dir>/cache/body/ -> re-run notebook 31 (pose cache cell).
POSE_CACHE_FILENAME: str = "pose.pkl"
# Version stored in the scene embedding cache; a mismatch refuses the old cache
# -> rebuild with python -m wooftrainer.scene_trainer.export.
SCENE_CACHE_FORMAT_VERSION: int = 3

# Pose-decoder data split by group; changing these moves images between splits
# -> retrain and re-evaluate the decoder in notebook 31.
POSE_SPLIT_SEED: int = 42
POSE_TRAIN_FRACTION: float = 0.70
POSE_VAL_FRACTION: float = 0.15  # test gets the rest


# --- Audio heads (notebooks 20-24): changing these needs retraining, not a cache rebuild ---

# Inverse regularisation strengths tried for every audio head's logistic regression.
AUDIO_C_VALUES: tuple[float, ...] = (0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0)
NUM_FOLDS: int = 5  # cross-validation folds for C, temperature and out-of-fold scores (also fusion)
CV_SEED: int = 0  # shuffles the cross-validation folds (also fusion and notebook 20)
MAX_ITER: int = 3000  # logistic-regression iteration limit; too low stops before convergence
# Search range for the calibration temperature; a fit at either end is reported.
TEMPERATURE_BOUNDS: tuple[float, float] = (0.25, 10.0)
NUDGE_L2_VALUES: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)  # emotion-nudge L2 strengths tried
# Macro-F1 gain the best nudge needs over "no nudge" to be kept.
MIN_NUDGE_GAIN: float = 0.01
# Probabilities are clipped to this before log() when scoring log-loss (no effect on models).
SCORE_PROB_FLOOR: float = 1e-12
# Bootstrap confidence intervals of macro-F1 in notebooks 20 and 24.
AUDIO_BOOTSTRAP_REPEATS: int = 1000  # resamples; more = steadier intervals, slower
AUDIO_BOOTSTRAP_SEED: int = 0  # seeds those resamples


# --- Fusion (notebooks 51, 52) ---

FUSION_C_VALUES: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)  # C grid for fusion heads
# Manifest splits fusion is fitted and cross-validated on; test stays held out (notebook 52).
TRAINVAL_SPLITS: tuple[str, ...] = ("train", "val")
FUSION_BOOTSTRAP_REPEATS: int = 2000  # bootstrap resamples over videos for the CIs
FUSION_BOOTSTRAP_SEED: int = 0  # seeds those resamples


# --- Scene calibration experiment (notebook 43) ---

SCENE_CV_FOLDS: int = 3  # grouped folds; reduce only if training folds cannot cover all classes
SCENE_CV_SEED: int = 42  # shuffles those folds
SCENE_BOOTSTRAP_REPEATS: int = 1000  # paired grouped bootstrap resamples
SCENE_BOOTSTRAP_SEED: int = 42  # seeds those resamples


# --- Pose decoder (notebook 31) ---

POSE_ENCODER_BATCH_SIZE: int = 32  # images per pose-encoder call (larger overflowed the buffer)
POSE_DECODER_LR: float = 0.003  # learning rate; winner of the sweep in notebook 31
POSE_DECODER_LR_VALUES: tuple[float, ...] = (0.0003, 0.001, 0.003)  # learning rates swept
POSE_DECODER_EPOCHS: int = 250  # training epochs per run
POSE_DECODER_BATCH_SIZE: int = 64  # training batch size
POSE_DECODER_SEED: int = 42  # torch seed for weight init and shuffling
