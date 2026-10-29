from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from woofdecoder.config import REPO_ROOT
from woofdecoder.sed_classes import SED_CLASSES

PRETRAINED_SED_DIR: Path = REPO_ROOT / "third_party" / "PretrainedSED"
CLONE_COMMAND: str = (
    "git clone https://github.com/fschmid56/PretrainedSED.git third_party/PretrainedSED"
)
CHECKPOINT_DIR: Path = REPO_ROOT/ "models" / "pretrained_sed"
CHECKPOINT_URL: str = "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1"
DEFAULT_MODEL_NAME: str = "frame_mn10"

SAMPLE_RATE: int = 16_000
CHUNK_S: int = 10 # the model trained on 10 s data
CHUNK_SAMPLES: int = CHUNK_S * SAMPLE_RATE # 160,000
FRAMES_PER_CHUNK: int = 250
SAMPLES_PER_FRAME: int = CHUNK_SAMPLES // FRAMES_PER_CHUNK # 640 samples = 40ms
FRAME_S: float = SAMPLES_PER_FRAME / SAMPLE_RATE # 0.04
CHUNKS_PER_BATCH: int = 8 # 80 s of audio per model to minize the memory

AUDIOSET_CLASSES_PER_SED: dict[str, list[str]] = {
    "bark": ["Bark", "Yip", "Bow-wow"],
    "growl": ["Growling"],
    "whine": ["Whimper (dog)"],
    "howl": ["Howl"],
    "pant": ["Pant (dog)"],
    "other_vocal": ["Dog", "Canidae, wild dogs, wolves"]
}

@dataclass
class BarkDetector:
    # The loaded SED model + what is needed to read its outputs as we wanted
    
    model: torch.nn.Module
    class_names: list[str] # 447 Audioset strong classes, in output order
    columns_per_sed: dict[str, list[str]] # SED class -> its columns in the 447 outputs 
    device: torch.device
    
def add_pretrained_sed_to_path()-> None:
    if not PRETRAINED_SED_DIR.is_dir():
        raise FileNotFoundError(f"{PRETRAINED_SED_DIR} is missing; run: {CLONE_COMMAND}")
    if str(PRETRAINED_SED_DIR) not in sys.path:
        sys.path.insert(0, str(PRETRAINED_SED_DIR))
        
def download_checkpont(model_name: str, verbose: bool = True) -> Path:
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
    checkpoint_path = download_checkpont(model_name, verbose)
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

def predict_audioset_probs(detector: BarkDetector, audio: np.ndarray) -> np.ndarray:
    num_frames = max(1, math.ceil(len(audio) / SAMPLES_PER_FRAME))
    num_chunks = max(1, math.ceil(len(audio) / CHUNK_SAMPLES))
    padded = np.zeros(num_chunks * CHUNK_SAMPLES, dtype=np.float32)
    padded[: len(audio)] = audio
    chunks = padded.reshape(num_chunks, CHUNK_SAMPLES)
    
    chunk_probs: list[np.ndarray] = []
    with torch.no_grad():
        for first in range(0, num_chunks, CHUNKS_PER_BATCH):
            batch = torch.from_numpy(chunks[first: first + CHUNKS_PER_BATCH]).to(detector.device)
            mel = detector.model.mel_forward(batch) # converts to mel spectogram
            logits, _ = detector.model(mel) # throw away whole clip score
            probs = torch.sigmoid(logits).transpose(1, 2) # (batch, 10 seconds clip, 447 sound labels)
            chunk_probs.append(probs.reshape(-1, probs.shape[-1]).cpu().numpy())
            
    all_probs = np.concatenate(chunk_probs, axis=0)
    return all_probs[:num_frames]

def to_sed_probs(detector: BarkDetector, audioset_probs: np.ndarray) -> np.ndarray:
    sed_columns = [
        audioset_probs[:, detector.columns_per_sed[sed_class]].max(axis=1)
        for sed_class in SED_CLASSES
    ]
    return np.stack(sed_columns, axis=1).astype(np.float32)

def detect_sed_probs(detector: BarkDetector, audio: np.ndarray) -> np.ndarray:
    return to_sed_probs(detector, predict_audioset_probs(detector, audio))

def frames_to_events(
    active_frames: np.ndarray, frame_s: float = FRAME_S
) -> list[tuple[float, float]]:
    
    events: list[tuple[float, float]] = []
    run_start: int | None = None
    for index, active in enumerate(active_frames):
        if active and run_start is None:
            run_start = index
        elif not active and run_start is not None:
            events.append((run_start * frame_s, index * frame_s))
            run_start = None
    
    if run_start is not None:
        events.append((run_start * frame_s, len(active_frames) * frame_s))
    return events