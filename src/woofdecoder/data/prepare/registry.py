"""Which datasets have a converter, and the converter of each.

A converter reads <data_dir>/raw/<id>/, writes its outputs into processed/<id>/ and
returns its manifest rows. Add a new source with one import and one CONVERTERS entry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from woofdecoder.data.manifest import ManifestRow
from woofdecoder.data.prepare import (
    act_env,
    audioset_strong_eval,
    emotional_canines,
    esc50,
    playpant,
)

# convert(raw_dir, out_dir) -> rows. out_dir is empty when the converter starts.
Converter = Callable[[Path, Path], list[ManifestRow]]

CONVERTERS: dict[str, Converter] = {
    "playpant": playpant.prepare,
    "audioset_strong_eval": audioset_strong_eval.prepare,
    "esc50": esc50.prepare,
    "act_env": act_env.prepare,
    "emotional_canines": emotional_canines.prepare,
}
