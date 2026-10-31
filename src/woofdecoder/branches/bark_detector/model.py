from __future__ import annotations

from dataclasses import dataclass

import torch

from woofdecoder.branches.bark_detector.pretrained_sed import (
    add_pretrained_sed_to_path,
    download_checkpoint
)
from woofdecoder.sed_classes import SED_CLASSES

DEFAULT_MODEL_NAME: str = "frame_mn10"

AUDIOSET_CLASSES_PER_SED: dict[str, list[str]] = {
    "bark": ["Bark", "Yip", "Bow-wow"],
    "growl": ["Growling"],
    "whine": ["Whimper (dog)"],
    "howl": ["Howl"],
    "pant": ["Pant (dog)"],
    "other_vocal": ["Dog", "Canidae, wild dogs, wolves"],
}

@dataclass
class BarkDetector:
    # The loaded SED model + what is needed to read its outputs as we wanted
    
    model: torch.nn.Module
    class_names: list[str] # 447 Audioset strong classes, in output order
    columns_per_sed: dict[str, list[str]] # SED class -> its columns in the 447 outputs 
    device: torch.device


def find_columns(class_names: list[str]) -> dict[str, list[int]]:
    # Return each SED class's column numbers in the model output.
    columns_per_sed: dict[str, list[int]] = {}
    
    for sed_class in SED_CLASSES:
        columns: list[int] = []
        for audioset_name in AUDIOSET_CLASSES_PER_SED[sed_class]:
            if audioset_name not in class_names:
                raise ValueError(f"AudioSet class {audioset_name!r} is not in the model output")
            
            columns.append(class_names.index(audioset_name))
        columns_per_sed[sed_class] = columns
    return columns_per_sed

def load_bark_detector(
    model_name: str = DEFAULT_MODEL_NAME, 
    device: str | None = None,
    verbose: bool = True
) -> BarkDetector:
    add_pretrained_sed_to_path()
    
    from data_util.audioset_classes import as_strong_train_classes
    from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper
    from models.frame_mn.utils import NAME_TO_WIDTH
    from models.prediction_wrapper import PredictionsWrapper
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build Network
    frame_mn = FrameMNWrapper(NAME_TO_WIDTH(model_name))
    embed_dim = frame_mn.state_dict()["frame_mn.features.16.1.bias"].shape[0]
    model = PredictionsWrapper(frame_mn, checkpoint=None, embed_dim=embed_dim)

    # Load weights (the checkpoint has no mel_transform weights, so load non-strict)
    checkpoint_path = download_checkpoint(model_name, verbose)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected or any("mel_transform" not in key for key in missing):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    model.to(device)
    
    class_names = list(as_strong_train_classes)
    columns_per_sed = find_columns(class_names)
    
    if verbose:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{num_params / 1e6:.2f}M parameters, {len(class_names)} classes")
        for sed_class, columns in columns_per_sed.items():
            names = [class_names[c] for c in columns]
            print(f"{sed_class:<12} <- {names}")
    
    return BarkDetector(model, class_names, columns_per_sed, torch.device(device))
