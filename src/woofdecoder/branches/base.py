"""Branch contract: every modality branch implements this.

Branch
takes 1 BarkSegment
returns 1 BranchOutput: probabilities over the shared subtype list from woofdecoder.taxonomy.

The bark detector is NOT a Branch.
It runs before the BarkSegments are consumed by the branches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.taxonomy import NUM_SUBTYPES, SUBTYPES

# How far the probabilities may be from summing to exactly 1.
PROB_SUM_TOLERANCE = 1e-4


@dataclass
class BranchOutput:
    """One branch's prediction for one segment.

    Attributes:
        probs: Subtype probabilities in SUBTYPES order, shape (NUM_SUBTYPES,). Must sum to 1.
        present: False if the modality was unavailable (e.g. no video), so fusion can skip it.
        extras: Evidence for the app's "why" panel (e.g. detected pose, matched scene prompt).
    """

    probs: np.ndarray
    present: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert probs to a float array and check its shape and sum.

        Raises:
            ValueError: If probs has the wrong shape or does not sum to 1.
        """
        self.probs = np.asarray(self.probs, dtype=float)

        expected_shape = (NUM_SUBTYPES,)
        if self.probs.shape != expected_shape:
            raise ValueError(
                f"BranchOutput.probs must have shape {expected_shape} (one value per subtype), "
                f"got shape {self.probs.shape}."
            )

        probs_sum = float(self.probs.sum())
        if not np.isclose(probs_sum, 1.0, atol=PROB_SUM_TOLERANCE):
            raise ValueError(
                f"BranchOutput.probs must sum to 1 (tolerance {PROB_SUM_TOLERANCE}), "
                f"got sum {probs_sum:.6f}."
            )

    @classmethod
    def missing(cls) -> BranchOutput:
        """Build the output for an unavailable modality.

        Returns:
            A BranchOutput with uniform probs and present=False.
        """
        uniform_probs = np.full(NUM_SUBTYPES, 1.0 / NUM_SUBTYPES)
        return cls(probs=uniform_probs, present=False)

    def as_dict(self) -> dict[str, float]:
        """Return the probabilities keyed by subtype name.

        Returns:
            A dict mapping each subtype name to its probability.
        """
        result: dict[str, float] = {}

        for position, subtype_name in enumerate(SUBTYPES):
            result[subtype_name] = float(self.probs[position])
            
        return result


class Branch(ABC):
    """Base class for a modality branch (audio, body, scene).

    Subclasses load their models once in __init__ and implement predict().
    """

    name: str = "branch"

    @abstractmethod
    def predict(self, segment: BarkSegment) -> BranchOutput:
        """Predict subtype probabilities for one segment.

        Args:
            segment: The bark segment to classify.

        Returns:
            The branch's BranchOutput. Use BranchOutput.missing() if the
            modality is not available in this segment.
        """

    def predict_many(self, segments: list[BarkSegment]) -> list[BranchOutput]:
        """Run predict() on each segment in order.

        Args:
            segments: The bark segments to classify.

        Returns:
            One BranchOutput per segment, in the same order.
        """
        outputs: list[BranchOutput] = []
        for segment in segments:
            outputs.append(self.predict(segment))
        return outputs
