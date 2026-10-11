"""Sound-event classes: the single source of truth for what the bark detector finds.

Frame targets, events.csv files and the app's bark detector all use SED_CLASSES, in this
order. Each source's labels map onto these classes here and nowhere else; a label that
maps to nothing is a frame with no dog vocalization.
"""

from __future__ import annotations

# Column order of every frame-target array, shape (num_frames, len(SED_CLASSES)).
SED_CLASSES: list[str] = ["bark", "growl", "whine", "howl", "pant", "other_vocal"]

# AudioSet strong-label event names. AudioSet "Pant" is a human breathing class, so it is
# left out on purpose and must never map to pant.
AUDIOSET_TO_SED: dict[str, str] = {
    "Bark": "bark",
    "Yip": "bark",
    "Bow-wow": "bark",
    "Growling": "growl",
    "Whimper (dog)": "whine",
    "Howl": "howl",
    "Bay": "howl",
    "Dog": "other_vocal",
    "Canidae, dogs, wolves": "other_vocal",
}

# PlayPanT letter codes (06 Readme.txt). J (Sneeze), K (Shake) and L (Human
# vocalizations) are left out: they are frames with no dog vocalization.
PLAYPANT_TO_SED: dict[str, str] = {
    "C": "bark",  # Bark / yelp
    "A": "growl",  # Growl
    "B": "whine",  # Whine
    "G": "howl",  # Howl
    "E": "pant",  # Pant
    "D": "other_vocal",  # Moan
    "F": "other_vocal",  # Cough / woof
    "H": "other_vocal",  # Grunts
    "I": "other_vocal",  # Other
}
