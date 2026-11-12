"""Training and evaluation code. Models come from woofdecoder; data comes from woofdata.

One sub-package per area, mirroring woofdecoder:
    preprocess_trainer/   bark-detector metrics
    audio_trainer/        audio embedding cache and heads
    body_trainer/         pose dataset and pose decoder
    scene_trainer/        CLIP segment cache and export, calibration experiment
    fusion_trainer/       branch outputs on the fusion videos and the fusion head

Shared modules at this level:
    torch_datasets.py     PyTorch Dataset over a woofdata task
    plots.py              confusion-matrix plot used by the notebooks
"""
