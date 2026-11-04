from __future__ import annotations

import numpy as np

from woofdecoder.preprocess.bark_detector.events import frames_to_events


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


def matched_event_scores(true_frames: np.ndarray, predicted_frames: np.ndarray) -> dict[str, float]:
    # Like event_scores, but each true event can match at most one predicted event and
    # the other way round. event_scores lets one long predicted event "find" every true
    # event it touches, so merging everything scores well; here it finds only one.
    num_true = 0
    num_predicted = 0
    num_matched = 0

    for true_row, predicted_row in zip(true_frames, predicted_frames):
        true_events = frames_to_events(true_row)
        predicted_events = frames_to_events(predicted_row)
        num_true += len(true_events)
        num_predicted += len(predicted_events)

        # Greedy: pair up the biggest overlaps first
        pairs = sorted(
            (
                (min(t[1], p[1]) - max(t[0], p[0]), t_index, p_index)
                for t_index, t in enumerate(true_events)
                for p_index, p in enumerate(predicted_events)
                if overlaps(t, p)
            ),
            reverse=True,
        )
        used_true: set[int] = set()
        used_predicted: set[int] = set()
        for _, t_index, p_index in pairs:
            if t_index in used_true or p_index in used_predicted:
                continue
            used_true.add(t_index)
            used_predicted.add(p_index)
            num_matched += 1

    precision = safe_divide(num_matched, num_predicted)
    recall = safe_divide(num_matched, num_true)
    return {"precision": precision, "recall": recall, "f1": f1_from(precision, recall), "num_true": num_true, "num_predicted": num_predicted}