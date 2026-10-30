from __future__ import annotations

import numpy as np

from woofdecoder.branches.bark_detector.events import frames_to_events

def safe_divide(top: float, bottom: float) -> float:
    return top / bottom if bottom else 0.0

def f1_from(precision: float, recall: float) -> float:
    return safe_divide(2 * precision * recall, precision + recall)

def frame_scores(true_frames: np.ndarray, predicted_frames: np.ndarray) -> dict[str, float]:
    true_positives = int(np.sum(true_frames & predicted_frames))
    false_positives = int(np.sum(~true_frames & predicted_frames))
    false_negatives = int(np.sum(true_frames & ~predicted_frames))
    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    return {"precision": precision, "recall": recall, "f1": f1_from(precision, recall)}

def overlaps(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[0] < second[1] and second[0] < first[1]

def event_scores(true_frames: np.ndarray, predicted_frames: np.ndarray) -> dict[str, float]:
    num_true = 0
    num_predicted = 0
    true_found = 0
    predicted_correct = 0
    
    for true_row, predicted_row in zip(true_frames, predicted_frames):
        true_events = frames_to_events(true_row)
        predicted_events = frames_to_events(predicted_row)
        num_true += len(true_events)
        num_predicted += len(predicted_events)
        true_found += sum(any(overlaps(t,p) for p in predicted_events) for t in true_events)
        predicted_correct += sum(any(overlaps(p, t) for t in true_events) for p in predicted_events)
        
    precision = safe_divide(predicted_correct, num_predicted)
    recall = safe_divide(true_found, num_true)
    return {"precision": precision, "recall": recall, "f1": f1_from(precision, recall), "num_true": num_true, "num_predicted": num_predicted}
