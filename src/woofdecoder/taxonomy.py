"""Label taxonomy: the single source of truth for category names.

Every model predicts probabilities over the same flat list of 5 categories (CATEGORIES).
The order of CATEGORIES is the order of every probability array.

OTHER_CATEGORY means the dog vocalizes, but not in one of the 4 situations
(e.g. resting, bath). It does not mean "the model is unsure".

Never hard-code category names anywhere else; import them from here.
"""

from __future__ import annotations

# The categories every model predicts. Order here defines the probability array order.
CATEGORIES: list[str] = [
    "alert_aggressive",
    "playful_excited",
    "distressed_lonely",
    "attention_seeking",
    "other",
]

# Position of each category in a probability array of shape (NUM_CATEGORIES,).
CATEGORY_INDEX: dict[str, int] = {}
for position, category_name in enumerate(CATEGORIES):
    CATEGORY_INDEX[category_name] = position

NUM_CATEGORIES: int = len(CATEGORIES)

# The vocalization happens outside the 4 situations above.
OTHER_CATEGORY: str = "other"

# Human-readable names shown in the app.
DISPLAY_NAMES: dict[str, str] = {
    "alert_aggressive": "Alert / Aggressive",
    "playful_excited": "Playful / Excited",
    "distressed_lonely": "Distressed / Lonely",
    "attention_seeking": "Attention-seeking",
    "other": "Other",
}
