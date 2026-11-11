"""CLIP scene predictions using the same branch interface as audio and body."""

from woofdecoder.base_branch import Branch, BranchOutput
from woofdecoder.config import load_config
from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.scene.encoder import SceneEncoder, encoder_from_config


class SceneBranch(Branch):
    name = "scene"

    def __init__(self, encoder: SceneEncoder):
        self.encoder = encoder

    def predict(self, segment: BarkSegment) -> BranchOutput:
        return self.encoder.predict(segment)

    @classmethod
    def from_config(cls, config: dict | None = None, device: str | None = None):
        return cls(encoder_from_config(load_config() if config is None else config, device))
