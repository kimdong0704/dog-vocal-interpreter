"""CLIP scene branch: sampled frames -> segment probabilities and features."""

from woofdecoder.scene.branch import SceneBranch
from woofdecoder.scene.encoder import SceneEncoder, encoder_from_config

__all__ = ["SceneBranch", "SceneEncoder", "encoder_from_config"]
