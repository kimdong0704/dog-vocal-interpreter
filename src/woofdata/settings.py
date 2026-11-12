"""Data preparation settings: folder/file layout, window lengths, split shares and seeds.

Edit values here, not in the modules that use them. After editing a value in the first
section, re-run `python -m woofdata.prepare <source>` for the sources named in its comment;
`python -m woofdata.prepare --check` prints "mismatch" until the saved fingerprints are rebuilt.
"""

from __future__ import annotations

from woofdecoder.constants import SED_CHUNK_S, SED_FRAME_S

# --- Changing these invalidates saved models/caches: retrain/rebuild after editing ---

# Prepared-data layout under <data dir>; renaming orphans existing files ->
# python -m woofdata.prepare (all sources).
PROCESSED_DIRNAME: str = "processed"  # <data dir>/processed/<source>/
MANIFEST_FILENAME: str = "manifest.csv"  # one row per clip, all sources
ROWS_FILENAME: str = "rows.csv"  # each source's manifest rows
LABELS_FILENAME: str = "labels.csv"  # each source's original label columns
EVENTS_FILENAME: str = "events.csv"  # sound events of the bark-detector sources
SED_CLASSES_FILENAME: str = "sed_classes.json"  # which SED classes a source labels
INDEX_FILENAME: str = "index.csv"  # Stanford Dogs image index (also read by dogposecv)
BOXES_FILENAME: str = "boxes.csv"  # Stanford Dogs bounding boxes (also read by dogposecv)
AUDIO_DIRNAME: str = "audio"  # WAV folder inside an audio source's processed folder
TARGETS_DIRNAME: str = "targets"  # per-window frame-target .npy folder (bark-detector sources)
IMAGES_DIRNAME: str = "images"  # image folder inside an image source's processed folder
# <data dir>/cache/<name>: every embedding / branch-output cache; renaming orphans all caches.
CACHE_DIRNAME: str = "cache"

# Window and frame length of the bark-detector targets; they must match the pretrained
# detector, so they come from woofdecoder.constants ->
# python -m woofdata.prepare audioset_strong_eval esc50 playpant.
DEFAULT_WINDOW_S: float = float(SED_CHUNK_S)  # 10 s windows
DEFAULT_FRAME_S: float = SED_FRAME_S  # 0.04 s frames, 250 per window
# Longest audio kept per clip by the clip-level audio tasks (centre of the clip); changes
# what the audio encoder sees -> rebuild audio embedding caches (notebook 20 / 23).
CLIP_MAX_SECONDS: float = DEFAULT_WINDOW_S

# Self-labelled videos -> python -m woofdata.prepare self_labeled, then notebooks 23 + 50.
SELF_LABELED_WINDOW_S: float = DEFAULT_WINDOW_S  # spans are cut into windows of at most this length
SELF_LABELED_MIN_WINDOW_S: float = 1.0  # a leftover piece shorter than this is dropped
BRANCH_ROLE: str = "branch"  # collection used to train the branches
FUSION_ROLE: str = "fusion"  # collection kept for fitting and testing fusion
# Every collection folder must be listed here, so a new one is added on purpose.
COLLECTION_ROLES: dict[str, str] = {
    "audio": BRANCH_ROLE,
    "fusion": FUSION_ROLE,
}
# Role -> (test share, val share) of its dogs; the rest is train. Moves dogs between splits.
SPLIT_SHARES: dict[str, tuple[float, float]] = {
    BRANCH_ROLE: (0.0, 0.2),
    FUSION_ROLE: (0.2, 0.2),
}

# Per-source split assignment; changing any of these moves clips between splits.
# playpant -> python -m woofdata.prepare playpant
PLAYPANT_SPLIT_SEED: int = 0  # shuffles the dogs before the val / test cut
PLAYPANT_NUM_VAL_DOGS: int = 9  # dogs in val; the other 8 go to test
# audioset_strong_eval -> python -m woofdata.prepare audioset_strong_eval
AUDIOSET_NUM_NEGATIVES: int = 300  # clips without a dog event that are kept
AUDIOSET_NEGATIVE_SEED: int = 0  # picks those negatives
# rf_bowl recordings -> python -m woofdata.prepare rf_bowl
ROBOFLOW_BALANCED_TEST_SHARE: float = 0.10  # share of recordings in test
ROBOFLOW_BALANCED_VAL_SHARE: float = 0.10  # share of recordings in val
ROBOFLOW_MIN_EVAL_RECORDINGS: int = 5  # val / test each take this many first; fewer warns
# Stanford Dogs images outside StanfordExtra: sha1(image key) mod HASH_BUCKETS below these
# bounds -> train / val, the rest -> test -> python -m woofdata.prepare stanford_dogs
STANFORD_TRAIN_BELOW: int = 70
STANFORD_VAL_BELOW: int = 85
HASH_BUCKETS: int = 100  # number of hash buckets used by split_from_hash


# --- Safe to change: reports, warnings and loading only ---

# Manifest split values that count as training for every task. train_only rows (labels
# not trusted enough to report on) are trained on but never evaluated.
TRAIN_SPLITS: tuple[str, ...] = ("train", "train_only")
EVAL_SPLITS: tuple[str, ...] = ("val", "test")  # splits that are reported on

MIN_EVAL_DOGS: int = 4  # self_labeled: a val or test split with fewer dogs prints a warning
# self_labeled: how much shorter than its label a cut span may be before end_s counts as
# past the end of the video (an error).
SPAN_TOLERANCE_S: float = 0.1
ROBOFLOW_MIN_EVAL_BOWL_BOXES: int = 50  # rf_bowl: warn when val / test has fewer bowl boxes
DOG_POSE_NUM_SPOT_CHECKS: int = 50  # dog_pose report: images compared against StanfordExtra
DOG_POSE_SPOT_CHECK_SEED: int = 0  # dog_pose report: picks those images
DOG_POSE_MAX_PIXEL_ERROR: float = 2.0  # dog_pose report: largest keypoint error that passes
ACT_ENV_RARE_VALUE_COUNT: int = 5  # act_env report: environments seen fewer times are listed
MACOS_FOLDER: str = "__MACOSX/"  # archive members under this folder are skipped

# Download report: a dataset folder smaller than expected by more than this share is flagged.
DOWNLOAD_SIZE_SHORTFALL_TOLERANCE: float = 0.10
