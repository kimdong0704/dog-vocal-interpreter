from __future__ import annotations

from pathlib import Path

import numpy as np

from woofdecoder.preprocess.bark_detector import (
    FRAME_S,
    BarkDetector,
    DetectionSettings,
    detect_sed_probs,
    find_dog_events,
    vocal_score
)
from woofdecoder.preprocess.media import TARGET_SAMPLE_RATE, load_audio_16k, sample_frames
from woofdecoder.preprocess.segments import BarkSegment
from woofdecoder.sed_classes import SED_CLASSES

# Other_vocal is the fallback (see pick_event_class).
SPECIFIC_CLASSES: list[str] = ["bark", "growl", "whine", "howl"]


def pick_event_class(event_probs: np.ndarray, threshold: float) -> tuple[str, float]:
    mean_probs = event_probs.mean(axis=0)  # shape (6,)
    best_class = max(SPECIFIC_CLASSES, key=lambda name: mean_probs[SED_CLASSES.index(name)])
    best_score = float(mean_probs[SED_CLASSES.index(best_class)])
    
    if best_score >= threshold:
        return best_class, best_score
    
    return "other_vocal", float(mean_probs[SED_CLASSES.index("other_vocal")])

def cut_audio(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    first = round(start_s * TARGET_SAMPLE_RATE)
    last = round(end_s * TARGET_SAMPLE_RATE)

    return audio[first:last].copy()


def frame_range(start_s: float, end_s: float) -> slice:
    return slice(round(start_s / FRAME_S), round(end_s / FRAME_S))

def build_segments(
    path: str | Path,
    detector: BarkDetector,
    settings: DetectionSettings,
    source_id: str = "",
    verbose: bool = True,
) -> list[BarkSegment]:
    
    path = Path(path)
    if verbose:
        print(f"Building bark segments for {path.name}")

    # audio
    audio = load_audio_16k(path, verbose=False)
    if audio is None:
        if verbose:
            print("No audio track")
        return []
    if verbose:
        print(f"Audio: {len(audio) / TARGET_SAMPLE_RATE:.2f} s")

    # SED class probs
    sed_probs = detect_sed_probs(detector, audio)  # shape (num_frames, 6)
    if verbose:
        print(f"Scored {len(sed_probs)} frames of {FRAME_S * 1000:.0f} ms")

    # events
    events = find_dog_events(vocal_score(sed_probs), settings)
    if verbose:
        print(f"Found {len(events)} dog sounds (threshold {settings.threshold})")

    # one segment per event
    segments: list[BarkSegment] = []
    for number, (start_s, end_s) in enumerate(events, start=1):
        sed_class, score = pick_event_class(sed_probs[frame_range(start_s, end_s)], settings.threshold)
        frames, frame_times_s = sample_frames(path, start_s, end_s, verbose=False)
        
        segment = BarkSegment(
            start_s=start_s,
            end_s=end_s,
            audio=cut_audio(audio, start_s, end_s),
            sample_rate=TARGET_SAMPLE_RATE,
            frames=frames,
            frame_times_s=frame_times_s,
            source_id=source_id or path.stem,
            sed_class=sed_class,
            confidence=score,
        )
        
        segments.append(segment)
        
        if verbose:
            print(f"Segment {number}/{len(events)}:")
            print(f"{start_s:6.2f}-{end_s:6.2f} s {sed_class:<11} score {score:.2f}")
            print(f"{len(frames)} frames")
            
    return segments