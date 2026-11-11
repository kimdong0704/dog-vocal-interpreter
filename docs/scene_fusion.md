# Scene features for fusion

The scene encoder combines frozen prompt-free YOLOE detections with the existing
RF-DETR dog detector. YOLOE uses confidence 0.6; RF-DETR Nano uses confidence 0.5.
RF-DETR dogs replace YOLOE boxes in the same frame when IoU is at least 0.5. All
other YOLOE objects stay. Both models run one frame at a time.

## Runtime interface

```python
from woofdecoder.scene import encoder_from_config
from woofdecoder.config import load_config

encoder = encoder_from_config(load_config())
result = encoder.encode_segment(segment)
scene_vector = result["embedding"]
```

`segment` is the preprocessing team's `BarkSegment`. The result includes source
and interval metadata, frame timestamps, detector provenance, the merged objects,
and availability flags. `encoder.embed(segment)` returns only the vector, or
`None` when the segment has no video. An existing preprocessing `DogDetector` can
be passed as `encoder_from_config(config, dog_detector=detector)` so it is shared.

The current checkpoint has 4,585 classes and ten features per class, producing
45,850 float32 values per segment. The feature order is class-major: ten features
for vocabulary class zero, then ten for class one, and so on. Read the vocabulary
and feature names from the stored schema rather than hardcoding class IDs.

## Export already-preprocessed segments

Install the project in the selected Python 3.12 environment:

```powershell
pip install -e ".[ml,dev]"
```

Export from a JSON manifest without opening a notebook:

```powershell
python -m wooftrainer.scene_export --manifest segments.json  # writes data/cache/scene/scene.npz
```

Optional arguments are `--device cpu` (or `cuda`) and `--config path/to/pipeline.yaml`.
Frame paths are relative to the manifest's folder. Use the same `clip_id` for the
corresponding audio and body segment; two barks in one video need different IDs.
Timestamps are seconds in the source video, within the segment's start/end bounds.

Illustrative manifest structure:

```json
[
  {
    "clip_id": "video_001@1.200-2.400",
    "source_id": "video_001",
    "start_s": 1.2,
    "end_s": 2.4,
    "frames": ["frames/video_001_001.jpg", "frames/video_001_002.jpg"],
    "frame_times_s": [1.4, 2.0],
    "dog_id": "dog_001",
    "split": "train",
    "mapped_category": "playful_excited"
  }
]
```

The label above illustrates the format; supply `mapped_category` only when it is
an actual reviewed label. Other annotation fields are preserved as metadata.
Missing video is represented by `frames: []` and `frame_times_s: []`. The command
validates records and image files before loading the models. It does not decode
videos, find barks, extract frames, create labels, or assign train/test splits.

## Load and align cached features

Fusion training can read a cache without loading either detector or their weights:

```python
from wooftrainer.scene_trainer.embeddings import default_scene_cache, load_scene_cache

scene_x, scene_rows = load_scene_cache(default_scene_cache())
schema_id = scene_rows.attrs["scene_schema_id"]
schema = scene_rows.attrs["scene_schema"]
```

`scene_x` has shape `(N, 45850)` and dtype float32 for the current checkpoint.
Each row has a unique `clip_id`, source/interval metadata, and separate flags:

- `present=False`: no video; the scene vector is zero.
- `present=True, has_detections=False`: frames exist but neither detector found objects.
- `present=True, has_detections=True`: at least one merged object was encoded.

Align scene rows to another modality's explicit segment IDs, rather than relying
on extraction order:

```python
scene_x, scene_rows = load_scene_cache(
    default_scene_cache(),
    clip_ids=audio_rows["clip_id"].tolist(),
    expected_schema_id=schema_id,
)
scene_mask = scene_rows["present"].to_numpy(dtype="float32")[:, None]
```

The loader rejects missing or duplicate requested IDs and returns features in the
requested order. A trained fusion checkpoint should retain its expected scene
schema ID. The cache stores the format version, vocabulary, feature order,
checkpoint hashes, and detector settings. Older scene caches must be regenerated
after this schema/export update.

When segments already exist in Python, call
`cache_scene_segments(path, clip_ids, segments, encoder, metadata=metadata)`.
`segments` may be a generator, and metadata is an optional list of genuine
annotation dictionaries in the same order. Keep dog IDs and the team's fixed
split assignments when constructing fusion batches.

## Handoff boundary

Keep availability masks alongside the vectors, and join all modalities by the
same segment IDs and intervals. Standardization, feature selection, or projection
must be fitted on fusion training data only, then reused for validation and test.
The scene vector is sparse and may benefit from reduction inside the fusion model.

The scene encoder supplies environmental features and detection evidence. The
fusion model learns behavioral categories from labeled examples. Dog correction
does not repair other YOLOE mistakes, and class-center changes do not track
individual objects or establish causes of barking.
