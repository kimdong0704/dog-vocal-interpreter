"""Which datasets have a converter, and the converter of each.

A converter reads <data_dir>/raw/<id>/, writes its outputs into processed/<id>/ and
returns its manifest rows. Add a new source with one import and one CONVERTERS entry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from woofdata.prepare.manifest import ManifestRow
from woofdata.prepare.sources import (
    act_env,
    audioset_strong_eval,
    dog_pose,
    dogposecv,
    emotional_canines,
    esc50,
    playpant,
    roboflow,
    stanford_dogs,
    stanfordextra,
)

# convert(raw_dir, out_dir) -> rows. out_dir is empty when the converter starts.
Converter = Callable[[Path, Path], list[ManifestRow]]

# prepare_all runs sources in configs/datasets.yaml order, where stanford_dogs comes before
# stanfordextra, dog_pose and dogposecv (they read its index.csv and boxes.csv).
CONVERTERS: dict[str, Converter] = {
    "playpant": playpant.prepare,
    "audioset_strong_eval": audioset_strong_eval.prepare,
    "esc50": esc50.prepare,
    "act_env": act_env.prepare,
    "emotional_canines": emotional_canines.prepare,
    "stanford_dogs": stanford_dogs.prepare,
    "stanfordextra": stanfordextra.prepare,
    "dog_pose": dog_pose.prepare,
    "dogposecv": dogposecv.prepare,
    "rf_leash": roboflow.prepare_rf_leash,
    "rf_bowl": roboflow.prepare_rf_bowl,
}
