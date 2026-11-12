"""Fusion: the three branch outputs -> one prediction per segment."""

from woofdecoder.fusion.fusers import BRANCH_NAMES, FEATURE_NAMES, FusionModel, fusion_features

__all__ = ["BRANCH_NAMES", "FEATURE_NAMES", "FusionModel", "fusion_features"]