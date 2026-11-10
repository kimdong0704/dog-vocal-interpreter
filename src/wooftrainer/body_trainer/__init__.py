"""Body (pose) decoder training, from notebooks 31 and 32.

pose_dataset: label table -> pose encoder cache -> 64 features -> group split.
pose_decoder: class-weighted MLP training, learning-rate sweep, evaluation vs baseline.
"""
