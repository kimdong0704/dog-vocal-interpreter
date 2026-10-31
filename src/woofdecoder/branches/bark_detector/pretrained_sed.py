from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

from woofdecoder.config import REPO_ROOT

PRETRAINED_SED_DIR: Path = REPO_ROOT / "third_party" / "PretrainedSED"
CLONE_COMMAND: str = (
    "git clone https://github.com/fschmid56/PretrainedSED.git third_party/PretrainedSED"
)
CHECKPOINT_DIR: Path = REPO_ROOT / "models" / "pretrained_sed"
CHECKPOINT_URL: str = "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"


def add_pretrained_sed_to_path()-> None:
    if not PRETRAINED_SED_DIR.is_dir():
        raise FileNotFoundError(f"{PRETRAINED_SED_DIR} is missing; run: {CLONE_COMMAND}")
    if str(PRETRAINED_SED_DIR) not in sys.path:
        sys.path.insert(0, str(PRETRAINED_SED_DIR))

def download_checkpoint(model_name: str, verbose: bool = True) -> Path:
    path = CHECKPOINT_DIR / f"{model_name}_strong_1.pt"
    
    if path.is_file():
        if verbose:
            print(f"Checkpoint already downloaded: {path.name}")
        return path
    if verbose:
        print(f"Downloading {path.name} (about 15 MB)")
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(f"{CHECKPOINT_URL}/{path.name}", str(path))
    return path