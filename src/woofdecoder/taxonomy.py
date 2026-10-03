"""Label taxonomy: the single source of truth for subtypes and categories.

Every model predicts probabilities over the same flat list of subtypes (SUBTYPES).
A category's probability is the sum of its subtypes' probabilities.

Never hard-code subtype or category names anywhere else; import them from here.
If a subtype has too little data and must be merged into its parent category,
that merge is done in this file only.
"""

from __future__ import annotations

import numpy as np

# Each category and the subtypes that belong to it. Order here defines SUBTYPES order.
CATEGORIES: dict[str, list[str]] = {
    "alert_aggressive": ["stranger_alert", "threat", "fearful_defensive"],
    "playful_excited": ["play", "greeting"],
    "distressed_lonely": ["separation", "anxious_whining"],
    "attention_seeking": ["demand", "interaction"],
}

# Flat subtype list and subtype -> category lookup, built from CATEGORIES.
SUBTYPES: list[str] = []
SUBTYPE_TO_CATEGORY: dict[str, str] = {}
for category_name, subtype_names in CATEGORIES.items():
    for subtype_name in subtype_names:
        SUBTYPES.append(subtype_name)
        SUBTYPE_TO_CATEGORY[subtype_name] = category_name

# Position of each subtype in a probability array of shape (NUM_SUBTYPES,).
SUBTYPE_INDEX: dict[str, int] = {}
for position, subtype_name in enumerate(SUBTYPES):
    SUBTYPE_INDEX[subtype_name] = position

NUM_SUBTYPES: int = len(SUBTYPES)

# Human-readable names shown in the app.
DISPLAY_NAMES: dict[str, str] = {
    "alert_aggressive": "Alert / Aggressive",
    "playful_excited": "Playful / Excited",
    "distressed_lonely": "Distressed / Lonely",
    "attention_seeking": "Attention-seeking",
    "stranger_alert": "Stranger alert",
    "threat": "Threat / aggressive",
    "fearful_defensive": "Fearful / defensive",
    "play": "Play",
    "greeting": "Greeting / owner arriving",
    "separation": "Separation / lonely",
    "anxious_whining": "Anxious / whining",
    "demand": "Demand (food, walk, let out)",
    "interaction": "Interaction with owner",
}


def category_probs(subtype_probs: np.ndarray) -> dict[str, float]:
    """Sum subtype probabilities into category probabilities.

    Args:
        subtype_probs: Probabilities in SUBTYPES order, shape (NUM_SUBTYPES,).

    Returns:
        A dict mapping each category name to the sum of its subtypes' probabilities.
    """
    subtype_probs = np.asarray(subtype_probs, dtype=float)  # shape (NUM_SUBTYPES,)

    result: dict[str, float] = {}
    for category_name, subtype_names in CATEGORIES.items():
        category_total = 0.0

        for subtype_name in subtype_names:
            position = SUBTYPE_INDEX[subtype_name]
            category_total += subtype_probs[position]

        result[category_name] = float(category_total)

    return result
