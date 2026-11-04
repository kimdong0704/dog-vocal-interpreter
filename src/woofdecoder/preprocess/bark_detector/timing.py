from __future__ import annotations

SAMPLE_RATE: int = 16_000
CHUNK_S: int = 10 # the model was trained on 10 s pieces
CHUNK_SAMPLES: int = CHUNK_S * SAMPLE_RATE # 160,000
FRAMES_PER_CHUNK: int = 250
SAMPLES_PER_FRAME: int = CHUNK_SAMPLES // FRAMES_PER_CHUNK # 640 samples
FRAME_S: float = SAMPLES_PER_FRAME / SAMPLE_RATE # 0.04 s