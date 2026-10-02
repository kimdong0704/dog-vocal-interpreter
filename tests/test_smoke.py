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
    NUM_SUBTYPES,
    SUBTYPES,
    category_probs,
)


class UniformBranch(Branch):
    """Test branch that always predicts the same probability for every subtype."""

    name = "uniform"

    def predict(self, segment: BarkSegment) -> BranchOutput:
        """Return uniform probabilities, ignoring the segment."""
        uniform_probs = np.full(NUM_SUBTYPES, 1.0 / NUM_SUBTYPES)
        return BranchOutput(probs=uniform_probs)


def test_taxonomy_is_consistent() -> None:
    """There are 9 unique subtypes, and every label has a display name."""
    assert NUM_SUBTYPES == 9
    assert len(SUBTYPES) == len(set(SUBTYPES))

    for subtype_name in SUBTYPES:
        assert subtype_name in DISPLAY_NAMES
    for category_name in CATEGORIES:
        assert category_name in DISPLAY_NAMES


def test_category_probs_sum_to_one() -> None:
    """Category probabilities built from valid subtype probabilities sum to 1."""
    random_generator = np.random.default_rng(seed=0)
    subtype_probs = random_generator.dirichlet(np.ones(NUM_SUBTYPES))

    probs_by_category = category_probs(subtype_probs)

    assert sum(probs_by_category.values()) == pytest.approx(1.0)


def test_branch_contract() -> None:
    """A Branch returns a valid present output, and missing() is not present."""
    segment = BarkSegment(start_s=0.0, end_s=1.0)

    output = UniformBranch().predict(segment)

    assert output.present
    assert output.probs.shape == (NUM_SUBTYPES,)
    assert BranchOutput.missing().present is False


def test_branch_output_rejects_bad_probs() -> None:
    """Probabilities that do not sum to 1 raise ValueError."""
    with pytest.raises(ValueError):
        BranchOutput(probs=np.ones(NUM_SUBTYPES))


def test_config_loads() -> None:
    """The default config file loads and has the main sections."""
    config = load_config()

    assert "bark_detector" in config
    assert "branches" in config
    assert "fusion" in config
