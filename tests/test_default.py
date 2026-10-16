"""Basic tests for the base contracts: taxonomy, branch output, and config."""

from __future__ import annotations

import numpy as np
import pytest

from woofdecoder.branches.base import Branch, BranchOutput
from woofdecoder.config import load_config
from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.taxonomy import (
    CATEGORIES,
    DISPLAY_NAMES,
    NUM_CATEGORIES,
    OTHER_CATEGORY,
)


class UniformBranch(Branch):
    """Test branch that always predicts the same probability for every category."""

    name = "uniform"

    def predict(self, segment: BarkSegment) -> BranchOutput:
        """Return uniform probabilities, ignoring the segment."""
        uniform_probs = np.full(NUM_CATEGORIES, 1.0 / NUM_CATEGORIES)  # shape (NUM_CATEGORIES,)
        return BranchOutput(probs=uniform_probs)


def test_taxonomy_is_consistent() -> None:
    """Tests: 5 unique categories, each with a display name, including OTHER_CATEGORY."""
    assert NUM_CATEGORIES == 5
    assert len(CATEGORIES) == len(set(CATEGORIES))
    assert OTHER_CATEGORY in CATEGORIES

    for category_name in CATEGORIES:
        assert category_name in DISPLAY_NAMES


def test_branch_contract() -> None:
    """Test: A Branch returns a valid present output, and missing() is not present."""
    segment = BarkSegment(start_s=0.0, end_s=1.0)

    output = UniformBranch().predict(segment)

    assert output.present
    assert output.probs.shape == (NUM_CATEGORIES,)
    assert BranchOutput.missing().present is False


def test_branch_output_rejects_bad_probs() -> None:
    """Test: BranchOutput rejects probabilities that do not sum to 1."""
    with pytest.raises(ValueError):
        BranchOutput(probs=np.ones(NUM_CATEGORIES))


def test_config_loads() -> None:
    """Test: The default config file loads and has the main sections."""
    config = load_config()

    assert "bark_detector" in config
    assert "branches" in config
    assert "fusion" in config
