from __future__ import annotations

import math

import numpy as np
import torch

from woofdecoder.preprocess.bark_detector.model import BarkDetector
from woofdecoder.preprocess.bark_detector.timing import CHUNK_SAMPLES, SAMPLES_PER_FRAME
from woofdecoder.sed_classes import SED_CLASSES

CHUNKS_PER_BATCH: int = 8


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
