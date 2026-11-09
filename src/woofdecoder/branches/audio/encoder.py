from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.config import REPO_ROOT

SAMPLE_RATE: int = 16_000

BEATS_DIR: Path = REPO_ROOT/"third_party"/"unilm"/"beats"
BESTS_CLONE_HINT: str = (
    "git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/unilm.git "
    "third_party/unilm\n  git -C third_party/unilm sparse-checkout set beats"
)

CHECKPOINTS: dict[str, Path] = {
    "beats_iter3_plus_as2m": REPO_ROOT/"models"/"beats"/"BEATs_iter3_plus_AS2M.pt"
}


@dataclass(frozen=True)
class EncoderSettings:
    min_seconds: float = 1.0
    chunk_seconds: float = 10.0
    batch_size: int = 1
    
def check_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"audio must be {SAMPLE_RATE} Hz, got {sample_rate} Hz")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"audio must be monon with shape (num_samples, ), got {audio.shape}")
    if audio.size == 0:
        raise ValueError("audio is empty")
    return audio

def split_into_chunks(audio: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    for start in range(0, len(audio), chunk_samples):
        chunks.append(audio[start:start + chunk_samples])
    return chunks

class AudioEncoder(ABC):
    name: str = "encoder"
    num_layers: int
    dim: int
    
    def __init__(self, settings: EncoderSettings, device: str) -> None:
        self.settings = settings
        self.device = device
        self.min_samples = round(settings.min_seconds * SAMPLE_RATE)
        self.chunk_samples = round(settings.chunk_seconds * SAMPLE_RATE)
        
    @abstractmethod
    def layer_features(
        self,
        audio: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
            returns
                features (num_layers, batch, num_patches, dim)
                padding_mask (batch, num_samples)
            
            Patches are like kernels in which divides the spectogram to manageable sizes for transformers
        """
    
    def embed_waveforms(
        self,
        waveforms: list[np.ndarray],
        sample_rate: int = SAMPLE_RATE
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        owners: list[int] = [] # which waaveform each chunk came from
        
        for index, audio in enumerate(waveforms):
            audio = check_audio(audio, sample_rate)
            for chunk in split_into_chunks(audio, self.chunk_samples):
                chunks.append(chunk)
                owners.append(index)
        
        sums = np.zeros((len(waveforms), self.num_layers, self.dim), dtype=np.float64)
        weights = np.zeros(len(waveforms), dtype=np.float64)
        batch_size = self.settings.batch_size
        
        for start in range(0, len(chunks), batch_size):
            pooled, num_patches = self.embed_chunks(chunks[start:start + batch_size])
            for owner, embedding, count in zip(
                owners[start:start + batch_size], pooled, num_patches, strict=True
            ):
                sums[owner] += embedding * count
                weights[owner] += count
                
        if (weights == 0).any():
            raise ValueError("a waveform is too short to fill one encoder paatch")
        
        return (sums / weights[:, None, None]).astype(np.float32)
    
    def embed_chunks(self, chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        length = max(self.min_samples, max(len(chunk) for chunk in chunks))
        audio = np.zeros((len(chunks), length), dtype=np.float32)
        padding = np.ones((len(chunks), length), dtype=bool)
        
        for row, chunk in enumerate(chunks):
            audio[row, : len(chunk)] = chunk
            padding[row, : len(chunk)] = False
            
        audio_tensor = torch.from_numpy(audio).to(self.device)
        padding_tensor = torch.from_numpy(padding).to(self.device)
        
        with torch.no_grad():
            features, patch_padding = self.layer_features(audio_tensor, padding_tensor)
            
        keep = (~patch_padding).to(features.dtype)
        num_patches = keep.sum(dim=1)
        summed = (features * keep[None, :, :, None]).sum(dim=2)
        pooled = summed / num_patches.clamp(min=1)[None, : , None]
        return pooled.permute(1, 0, 2).cpu().numpy(), num_patches.cpu().numpy()
    
    def embed(self, segment: BarkSegment) -> np.ndarray | None:
        if not segment.has_audio:
            return None
        return self.embed_waveforms([segment.audio], segment.sample_rate)[0]
    
def import_beats() -> tuple[type, type]:
    if not (BEATS_DIR / "BEATs.py").is_file():
        raise FileNotFoundError(f"BEATs code not found in {BEATS_DIR}. Run:\n {BESTS_CLONE_HINT}")
    if str(BEATS_DIR) not in sys.path:
        sys.path.insert(0, str(BEATS_DIR))
    from BEATs import BEATs, BEATsConfig
    
    return BEATs, BEATsConfig

class BeatsEncoder(AudioEncoder):
    def __init__(
      self,
      name: str,
      checkpoint_path: Path,
      settings: EncoderSettings,
      device: str  
    ) -> None:
        super().__init__(settings, device)
        self.name = name
        if not checkpoint_path.is_file():
            raise ValueError(f"BEATs checkpoint not found: {checkpoint_path}")
        
        beats_class, config_class = import_beats()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model = beats_class(config_class(checkpoint["cfg"]))
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.model.to(device)
        
        self.captured: list[torch.Tensor] = []
        for layer in self.model.encoder.layers:
            layer.register_forward_hook(self.capture_layer)
            
        features, _ = self.layer_features(
            torch.zeros(1, SAMPLE_RATE, device=device),
            torch.zeros(1, SAMPLE_RATE, dtype=torch.bool, device=device)
        )
        self.num_layers = features.shape[0]
        self.dim = features.shape[-1]
        
    def capture_layer(self, module, inputs, output) -> None:
        self.captured.append(output[0])
        
    def layer_features(
        self,
        audio: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.captured = []
        with torch.no_grad():
            _, patch_padding = self.model.extract_features(audio, padding_mask=padding_mask)
        # (num_patches, batch, dim) -> (batch, num_patches, dim)
        features = torch.stack([output.transpose(0, 1) for output in self.captured])
        return features, patch_padding
    
def load_audio_encoder(
    name: str, settings: EncoderSettings | None = None,
    device: str | None = None
) -> AudioEncoder:
    if name not in CHECKPOINTS:
        raise ValueError(f"unknown audio encoder {name!r}; choose one of {list(CHECKPOINTS)}")
    if settings is None:
        settings = EncoderSettings()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return BeatsEncoder(name, CHECKPOINTS[name], settings, device)

def encoder_from_config(config: dict, device: str | None = None) -> AudioEncoder:
    section = config["branches"]["audio"]
    if section.get("encoder") is None:
        raise ValueError(f"branches.audio.encoder is not set in configs/pipeline.yaml")
    
    settings = EncoderSettings(
        min_seconds=float(section.get("min_seconds", 1.0)),
        chunk_seconds=float(section.get("chunk_seconds", 10.0)),
        batch_size=int(section.get("batch_size", 1))
    )
    
    return load_audio_encoder(section["encoder"], settings, device)