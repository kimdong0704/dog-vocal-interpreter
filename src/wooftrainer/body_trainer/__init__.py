"""Body (pose) decoder training, from notebooks 31 and 32.

pose_dataset: label table -> pose encoder cache -> 64 features -> group split.
pose_decoder: class-weighted MLP training, learning-rate sweep, evaluation vs baseline.
"""

from wooftrainer.body_trainer.pose_dataset import (
    PoseSplits,
    build_label_table,
    features_and_labels,
    load_or_build_pose_table,
    load_pose_encoder,
    split_by_group,
)
from wooftrainer.body_trainer.pose_decoder import (
    TrainHistory,
    evaluate_pose_decoder,
    load_pose_decoder,
    majority_baseline,
    sweep_learning_rates,
    train_pose_decoder,
)

__all__ = [
    "PoseSplits",
    "TrainHistory",
    "build_label_table",
    "evaluate_pose_decoder",
    "features_and_labels",
    "load_or_build_pose_table",
    "load_pose_decoder",
    "load_pose_encoder",
    "majority_baseline",
    "split_by_group",
    "sweep_learning_rates",
    "train_pose_decoder",
]
