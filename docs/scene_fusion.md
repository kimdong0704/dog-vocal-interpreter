# CLIP scene predictions for fusion

The scene branch uses frozen `openai/clip-vit-base-patch32`. It runs on the full RGB
frames already sampled inside each `BarkSegment`, using the checkpoint's image
preprocessor. No scene object detector or dog box is needed.

## One prediction per segment

For each frame, CLIP compares the image against five descriptions in `CATEGORIES`
order. Select the segment calculation with `branches.scene.aggregation`:

- `mean_frame_probabilities` (baseline): subtract the class bias from each frame's
  logits, divide by temperature, softmax, and average the five scores across frames.
- `mean_frame_logits`: average raw frame logits, subtract the class bias, divide by
  temperature, and apply one softmax.
- `confidence_weighted_logits` (configured experiment): compute each frame's probabilities as
  `softmax(frame_logits / temperature)`, take the largest probability as its
  confidence `c`, and assign weight `w = (1 - alpha) + alpha * c`. Average the raw
  frame logits using these weights, subtract the class bias, divide by temperature,
  and apply one softmax. The bias does not affect the frame weights.
  `alpha` is `branches.scene.confidence_weight`, a value from 0 to 1; it defaults
  to 0.5. At 0, this method is identical to `mean_frame_logits`. At 1, the weight
  equals the frame's confidence.

All three return one vector of five scores summing to one. The first two give every
frame equal weight. The sampler requests frames across the bark interval, but short
barks can produce repeated or nearly identical frames. Batch size does not change
aggregation weights. The logit methods are experiments, not proven improvements
on this dataset. Compare them using the same labeled, dog-disjoint segments.

```yaml
branches:
  scene:
    aggregation: confidence_weighted_logits  # mean_frame_probabilities is the baseline
    confidence_weight: 0.5  # range 0 to 1; used only by confidence_weighted_logits
    temperature: 1.0
    class_bias: [0.0, 0.0, 0.0, 0.0, 0.0]  # neutral; five values in taxonomy order
```

Change `aggregation` to switch methods without deleting code. After comparing them,
keep the method that works best on held-out groups; the other branches in
`SceneEncoder.encode_frames` can be removed if they are no longer needed.

The existing `temperature` setting must be positive. For `mean_frame_logits`, values
above 1 soften the segment scores and values below 1 sharpen them; the highest
category is unchanged. For `confidence_weighted_logits`, temperature affects both
frame weights and the final softmax, so changing it can change the top category.
The probability-mean method also applies temperature before averaging, so its top
category can change. `1.0` uses CLIP's native logit scale and does not imply the
scores are calibrated. Fit temperature and select aggregation and weighting using
labeled, dog-disjoint validation segments; reserve test dogs for final evaluation.

A high CLIP score is not necessarily evidence that a frame is correct or relevant.
Confidence weighting can amplify a confidently wrong frame. Compare segment-level
macro-F1, per-class recall and probability quality against the unweighted baselines
before using it for fusion.

Averaging logits does not automatically remove class-specific or prompt bias, and
temperature alone does not correct such bias. `class_bias` supplies five finite
values in taxonomy order (`alert_aggressive`, `playful_excited`, `distressed_lonely`,
`attention_seeking`, `other`). Each value is subtracted from its category's logit;
subtracting the same constant from all five has no effect on softmax. All zeros
turn the adjustment off. The default zeros and temperature 1.0 are neutral, not
fitted calibration values. Notebook 43 tests class-bias and temperature fitting;
fit any values on training groups only, and evaluate on separate held-out groups.
The notebook's bias-and-temperature experiment uses mean logits, so its fitted values
should not be assumed optimal for confidence weighting without evaluating that setup.

The frame probabilities in `output.extras["frame_probs"]` are based on uncorrected
frame logits; they are the values used for confidence weights. Segment probabilities
include the configured bias.

These are prompt-relative scores, not measured probabilities of a
dog's intent. `other` is an actual category, not an uncertainty bucket. The
notebook 40 prompts differ slightly from the encoder defaults; optional
`branches.scene.prompts` must map every taxonomy category to one description. Tune prompts and temperature on validation data only.

```python
from woofdecoder.scene import SceneBranch

scene = SceneBranch.from_config()
output = scene.predict(segment)
print(output.probs)    # shape (5,), in CATEGORIES order
print(output.present)  # False for missing video
print(output.extras["frame_probs"])  # inspect disagreement across frames
print(output.extras["frame_logits"]) # raw CLIP logits before temperature
```

Missing video returns uniform placeholder scores and `present=False`; fusion must
use the mask. A visible scene remains present even if no dog is visible.

`scene.encoder.encode_segment(segment)` returns both `probs` and a normalized mean image
`embedding` (512 values for the default checkpoint), plus frame scores, frame count,
timestamps, source ID, interval, and schema ID. `scene.encoder.embed(segment)` still returns the
embedding, or None for missing video. The embedding is a separate optional fusion
input; segment probabilities follow the selected aggregation setting.

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
the model revision, preprocessing, prompts, temperature, class bias, aggregation and
category order. The weighted method also records `confidence_weight` in its schema.
The cache stores segment scores, not frame logits; changing aggregation, confidence
weight, temperature or class bias requires rerunning CLIP to rebuild it. These settings change the
schema ID. Save the schema ID with fusion weights and pass `expected_schema_id` when
loading. Duplicate or missing requested IDs are rejected. Format v3 replaces the
object-detector cache; regenerate previous exports. Reading a cache needs no CLIP
weights. Keep fixed dog-disjoint splits for fusion training and evaluation.

The fusion model and runtime pipeline integration belong to the fusion teammate.


[Notebook 41](../notebooks/41_scene_training.ipynb) preserves the earlier YOLOE comparison as a historical experiment.
