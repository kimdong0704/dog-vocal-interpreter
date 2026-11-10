"""Body-language branch: pose encoder (YOLO11m-pose) + MLP decoder, decided in notebook 31.

For one BarkSegment:
1. Run the pose encoder on each sampled frame (up to FRAMES_PER_SEGMENT).
2. Keep only frames where exactly 1 dog is found (the same drop rule used in training).
3. Turn each kept frame into 64 features: 20 box-normalised keypoints (x, y, conf) + 4 cues.
4. Run the decoder on each kept frame and average the softmax probabilities (masked mean).

It looks at each frame as a still pose: it sees e.g. tail height, not tail wagging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from woofdecoder.base_branch import Branch, BranchOutput
from woofdecoder.config import REPO_ROOT, load_config
from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.taxonomy import CATEGORIES

BODY_CONFIG = load_config()["branches"]["body"]

# Class order the decoder was trained with (label_map in notebook 31).
# NOT the same as CATEGORIES order: other and attention_seeking are swapped.
DECODER_CLASSES: list[str] = [
    "alert_aggressive",
    "playful_excited",
    "distressed_lonely",
    "other",
    "attention_seeking",
]
NUM_KEYPOINTS = 20  # keypoints 20-23 had no labels in StanfordExtra, so they are dropped
NUM_FEATURES = 64  # 20 keypoints x (x, y, conf) + 4 cues


def pick_device() -> torch.device:
    """Same device order as the lessons: cuda, then mps (Mac), then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PoseClassifier(torch.nn.Module):
    """The decoder MLP from notebook 31: 64 -> 128 -> 5, outputs raw logits."""

    def __init__(self, n_features: int = NUM_FEATURES):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, 128)
        self.fc2 = torch.nn.Linear(128, len(DECODER_CLASSES))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(z1)


def parse_result(r) -> dict:
    """Read one Ultralytics result: keep the frame only if exactly 1 dog was found."""
    n = len(r.boxes)
    if n != 1:
        return dict(n_dogs=n, kept=False, box=None, keypoints=None)
    box = r.boxes.xyxy[0].cpu().numpy()  # (4,) x1, y1, x2, y2 in pixels
    kpts = r.keypoints.data[0, :NUM_KEYPOINTS].cpu().numpy()  # (20, 3) x, y, conf
    return dict(n_dogs=1, kept=True, box=box, keypoints=kpts)


def make_features(K: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Same features as training in notebook 31. If you change them there, change them here.

    Args:
        K: keypoints, shape (n, 20, 3) as x, y, conf in pixels.
        B: boxes, shape (n, 4) as x1, y1, x2, y2 in pixels.

    Returns:
        Features, shape (n, 64), float32.
    """
    x1, y1, x2, y2 = B[:, [0]], B[:, [1]], B[:, [2]], B[:, [3]]  # each (n, 1)
    w = x2 - x1
    h = y2 - y1
    Kn = K.copy()
    Kn[:, :, 0] = (K[:, :, 0] - x1) / w
    Kn[:, :, 1] = (K[:, :, 1] - y1) / h
    X = Kn.reshape(len(Kn), -1)  # (n, 60)

    tail_height = Kn[:, 12, 1] - Kn[:, 13, 1]  # positive = tail up (y grows downward)
    ear_height = ((Kn[:, 14, 1] - Kn[:, 18, 1]) + (Kn[:, 15, 1] - Kn[:, 19, 1])) / 2
    head_height = -Kn[:, 16, 1]
    aspect = (B[:, 2] - B[:, 0]) / (B[:, 3] - B[:, 1])
    cues = np.stack([tail_height, ear_height, head_height, aspect], axis=1)  # (n, 4)
    return np.concatenate([X, cues], axis=1).astype(np.float32)


def to_categories_order(probs: np.ndarray) -> np.ndarray:
    """Reorder decoder probabilities (DECODER_CLASSES order) into CATEGORIES order."""
    out = np.zeros(len(CATEGORIES))
    for position, name in enumerate(DECODER_CLASSES):
        out[CATEGORIES.index(name)] = probs[position]
    return out


class BodyBranch(Branch):
    """Body-language branch. Loads both models once, then predict() per segment."""

    name = "body"

    def __init__(
        self,
        encoder_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
    ):
        from ultralytics import YOLO  # imported here so the rest of the package works without it

        encoder_path = REPO_ROOT / (encoder_path or BODY_CONFIG["pose"])
        decoder_path = REPO_ROOT / (decoder_path or BODY_CONFIG["head"])
        for path in (encoder_path, decoder_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Weights are not in git: copy them in "
                    f"(see configs/pipeline.yaml, branches.body)."
                )

        self.device = pick_device()
        self.encoder = YOLO(str(encoder_path))
        self.decoder = PoseClassifier().to(self.device)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        self.decoder.eval()

    def predict(self, segment: BarkSegment) -> BranchOutput:
        """Average the decoder's probabilities over the frames that show exactly 1 dog.

        Returns BranchOutput.missing() if the segment has no video or no usable frame.
        """
        if not segment.has_video:
            return BranchOutput.missing()

        # Segment frames are RGB; Ultralytics treats numpy images as BGR, so flip channels.
        frames_bgr = [np.ascontiguousarray(frame[:, :, ::-1]) for frame in segment.frames]
        results = self.encoder.predict(source=frames_bgr, device=str(self.device), verbose=False)
        parsed = [parse_result(r) for r in results]
        mask = [p["kept"] for p in parsed]  # True = exactly 1 dog in that frame

        if not any(mask):
            output = BranchOutput.missing()
            output.extras = {"frames_used": mask, "dogs_per_frame": [p["n_dogs"] for p in parsed]}
            return output

        K = np.stack([p["keypoints"] for p in parsed if p["kept"]])  # (n_kept, 20, 3)
        B = np.stack([p["box"] for p in parsed if p["kept"]])  # (n_kept, 4)
        X = make_features(K, B)  # (n_kept, 64)

        with torch.no_grad():
            logits = self.decoder(torch.from_numpy(X).to(self.device))
            probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()  # masked mean

        return BranchOutput(
            probs=to_categories_order(probs),
            present=True,
            extras={
                "frames_used": mask,
                "tail_height": float(X[:, 60].mean()),
                "ear_height": float(X[:, 61].mean()),
                "head_height": float(X[:, 62].mean()),
            },
        )
