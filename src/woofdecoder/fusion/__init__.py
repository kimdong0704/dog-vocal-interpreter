"""Fusion: the three branch outputs -> one prediction per segment."""

from woofdecoder.fusion.fusers import FusionModel, fusion_features

__all__ = ["FusionModel", "fusion_features"]
