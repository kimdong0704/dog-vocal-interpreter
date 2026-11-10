# CLIP scene predictions for fusion

The scene branch uses frozen `openai/clip-vit-base-patch32`. It runs on the full RGB
frames already sampled inside each `BarkSegment`, using the checkpoint's image
preprocessor. No scene object detector or dog box is needed.

## One prediction per segment

For each frame, CLIP compares the image against five descriptions in `CATEGORIES`
order. Apply softmax to that frame's logits, then average the probability vectors
across all frames. This yields one vector of five scores summing to one. Averaging
retains uncertainty that majority voting discards. Each frame has equal weight;
the existing sampler distributes frames across the bark interval. Batch size does
not change aggregation weights, including for a short final batch.

These are uncalibrated prompt-relative scores, not measured probabilities of a
dog's intent. `other` is an actual category, not an uncertainty bucket. Default
prompts match notebook 40; optional `branches.scene.prompts` must map every taxonomy
category to one description. Tune prompts and temperature on validation data only.

```python
from woofdecoder.scene import SceneBranch

scene = SceneBranch.from_config()
output = scene.predict(segment)
print(output.probs)    # shape (5,), in CATEGORIES order
print(output.present)  # False for missing video
print(output.extras["frame_probs"])  # inspect disagreement across frames
```

Missing video returns uniform placeholder scores and `present=False`; fusion must
use the mask. A visible scene remains present even if no dog is visible.

`scene.encoder.encode_segment(segment)` returns both `probs` and a normalized mean image
`embedding` (512 values for the default checkpoint), plus frame scores, frame count,
timestamps, source ID, interval, and schema ID. `scene.encoder.embed(segment)` still returns the
embedding, or None for missing video. The embedding is a separate optional fusion
input; segment probabilities are computed by averaging frame probabilities.

## Export and load

Install the project's `ml` extra. First use downloads CLIP weights to the Hugging
Face cache. Configuration is in `branches.scene` in `configs/pipeline.yaml`.

```powershell
python -m wooftrainer.scene_export --manifest segments.json --output cache/scene.npz
```

The manifest contains already-preprocessed frames. Paths are relative to its folder:

```json
[
  {
    "clip_id": "video_001@1.200-2.400",
    "source_id": "video_001",
    "start_s": 1.2,
    "end_s": 2.4,
    "frames": ["frames/frame1.jpg", "frames/frame2.jpg"],
    "frame_times_s": [1.4, 2.0],
    "dog_id": "dog_001",
    "split": "train"
  }
]
```

Use identical segment IDs across branches. Supply `mapped_category` only for a
reviewed annotation. Missing video uses empty frame and timestamp lists. Frames
must be chronological and inside the segment. The exporter does not detect barks
or assign labels/splits.

```python
import numpy as np
from wooftrainer.scene_trainer.embeddings import load_scene_cache

features, rows = load_scene_cache("cache/scene.npz", clip_ids=audio_segment_ids)
scene_probs = np.stack(rows["probs"].to_numpy())  # (N, 5)
scene_mask = rows["present"].to_numpy()          # (N,)
schema_id = rows.attrs["scene_schema_id"]
```

The cache stores probabilities, embeddings, masks, metadata and a schema recording
the model revision, preprocessing, prompts, temperature, aggregation and category
order. Save the schema ID with fusion weights and pass `expected_schema_id` when
loading. Duplicate or missing requested IDs are rejected. Format v3 replaces the
object-detector cache; regenerate previous exports. Reading a cache needs no CLIP
weights. Keep fixed dog-disjoint splits for fusion training and evaluation.

The fusion
model and runtime pipeline integration belong to the fusion teammate.

The preserved [YOLOE notebook](../notebooks/41_scene_training.ipynb) owns its
legacy settings and helper code. It is a comparison experiment; its v2 caches
cannot be used as CLIP inputs. Active code loads no YOLOE scene model.
