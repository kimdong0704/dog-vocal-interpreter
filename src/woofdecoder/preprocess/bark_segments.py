from __future__ import annotations

from itertools import groupby
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
from woofdecoder.preprocess.segments import BarkSegment, SoundSection
from woofdecoder.sed_classes import SED_CLASSES

# Other_vocal is the fallback (see pick_event_class).
SPECIFIC_CLASSES: list[str] = ["bark", "growl", "whine", "howl"]
SPECIFIC_COLUMNS: list[int] = [SED_CLASSES.index(name) for name in SPECIFIC_CLASSES]
MIN_SECTION_FRAMES: int = 2 


def frame_labels(event_probs: np.ndarray, threshold: float) -> list[str]:
    specific = event_probs[:, SPECIFIC_COLUMNS]
    best = specific.argmax(axis=1)
    heard = specific.max(axis=1) >= threshold
    return [SPECIFIC_CLASSES[b] if h else "other_vocal" for b, h  in zip(best, heard)] 

def find_sound_sections(
    event_probs: np.ndarray, event_start_s: float, threshold: float
) -> list[SoundSection]:
    runs: list[list] = []
    first = 0
    
    for label, group in groupby(frame_labels(event_probs, threshold)):
        end = first + len(list(group))
        if runs and (end - first < MIN_SECTION_FRAMES or runs[-1][0] == label):
            runs[-1][2] = end
        else:
            runs.append([label, first, end])
        first = end
        
    if len(runs) > 1 and runs[0][2] - runs[0][1] < MIN_SECTION_FRAMES:
        short_first = runs.pop(0)
        runs[0][1] = short_first[1]
        
    sections: list[SoundSection] = []
    for label, first_frame, end_frame in runs:
        span_s = (
            round(event_start_s + first_frame * FRAME_S, 3),
            round(event_start_s + end_frame * FRAME_S, 3)
        )
        score = float(event_probs[first_frame:end_frame, SED_CLASSES.index(label)].mean())
        sections.append(SoundSection(label, span_s, score))

    return sections

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
        event_probs = sed_probs[frame_range(start_s, end_s)]
        sound_sections = find_sound_sections(event_probs, start_s, settings.threshold)
        confidence = float(vocal_score(event_probs).mean())
        frames, frame_times_s = sample_frames(path, start_s, end_s, verbose=False)
        
        segment = BarkSegment(
            start_s=start_s,
            end_s=end_s,
            audio=cut_audio(audio, start_s, end_s),
            sample_rate=TARGET_SAMPLE_RATE,
            frames=frames,
            frame_times_s=frame_times_s,
            source_id=source_id or path.stem,
            confidence=confidence,
            sound_sections=sound_sections
        )
        
        segments.append(segment)
        
        if verbose:
            print(f"Segment {number}/{len(events)}:")
            print(f"{start_s:6.2f}  -{end_s:6.2f}s")
            print(f"dog vocalizing {confidence:.2f}, {len(frames)} frames")
            
            for label, (section_start_s, section_end_s), score in sound_sections:
                print(f"{label:<11} {section_start_s:6.2f}  -{section_end_s:6.2f}s {score:.2f}")
            
    return segments