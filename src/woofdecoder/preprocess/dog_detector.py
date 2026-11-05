from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from woofdecoder.preprocess.segments import BarkSegment

DOG_CLASS_NAME: str = "dog"

# shape (4,), float32
# x, y, w, h in pixels
Box = np.ndarray 


@dataclass
class DogDetector:
    model: object
    score_threshold: float


def load_dog_detector(
    model_size: str = "nano", score_threshold: float = 0.5, verbose: bool = True
) -> DogDetector:

    from rfdetr import RFDETRNano, RFDETRSmall

    model_classes = {"nano": RFDETRNano, "small": RFDETRSmall}
    
    if model_size not in model_classes:
        raise ValueError(f"model_size must be one of {list(model_classes)}, got {model_size!r}")
    
    if verbose:
        print(f"Loading RF-DETR {model_size} (keeping only '{DOG_CLASS_NAME}')")
    
    return DogDetector(model_classes[model_size](), score_threshold)

def detect_dogs(detector: DogDetector, frames: list[np.ndarray]) -> list[list[tuple[Box, float]]]:
    from rfdetr.assets.coco_classes import COCO_CLASSES

    if not frames:
        return []
    
    results = detector.model.predict(list(frames), threshold=detector.score_threshold)
    
    if len(frames) == 1 and not isinstance(results, list):
        results = [results] 

    dogs_per_frame: list[list[tuple[Box, float]]] = []
    
    for detections in results:
        dogs: list[tuple[Box, float]] = []
        for (x1, y1, x2, y2), class_id, score in zip(
            detections.xyxy, detections.class_id, detections.confidence
        ):
            if COCO_CLASSES.get(int(class_id)) != DOG_CLASS_NAME:
                continue
            
            box = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)  
            dogs.append((box, float(score)))
        dogs_per_frame.append(dogs)
        
    return dogs_per_frame

def pick_dog(dogs: list[tuple[Box, float]]) -> Box | None:
    if not dogs:
        return None

    largest_box, _ = max(dogs, key=lambda dog: dog[0][2] * dog[0][3])

    return largest_box

def add_dog_boxes(segment: BarkSegment, detector: DogDetector, verbose: bool = True) -> BarkSegment:
   
    if not segment.has_video:
        if verbose:
            print(f"{segment.start_s:6.2f} s: no frames, skipped")
        return segment

    dogs_per_frame = detect_dogs(detector, segment.frames)
    boxes = [pick_dog(dogs) for dogs in dogs_per_frame]
    
    if verbose:
        counts = [len(dogs) for dogs in dogs_per_frame]
        found = sum(box is not None for box in boxes)
        print(f"{segment.start_s:6.2f} s: dogs per frame {counts} -> dog in {found}/{len(boxes)} frames")
        
    return replace(segment, dog_boxes=boxes)