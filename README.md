# WoofDecoder

**Multimodal dog vocalization interpreter**: a web app that estimates what a dog is most
likely communicating when it barks.

| | |
|---|---|
| **Course** | CSSE 473 Deep Learning, course project |
| **Team** | Leo K. · Eden K. · Elijah H. · Rohan M. · Jackson O. |

---

## Disclosure

WoofDecoder estimates the likely *context* of barking from patterns in sound and video.
It does **not** translate dog language.

Some models and datasets are licensed for **non-commercial use only**. This project is for
academic, non-commercial use.

---

## Description

People read a barking dog with more than their ears. WoofDecoder does the same with
three signals:

| Modality | Question it answers | Example cues |
|---|---|---|
| **Audio** | *How* is the dog vocalizing? | bark vs. growl vs. whine, pitch, rhythm |
| **Body language** | *What state* is the dog in? | tail height, ear position, posture |
| **Scene context** | *Why* might it be barking? | person at the door, ball, food bowl |

Each bark is classified into one of 5 categories:

| Category | Meaning |
|---|---|
| Alert / Aggressive | Alerting to a stranger or sound, threat, fearful defense |
| Playful / Excited | Play, greeting |
| Distressed / Lonely | Separation, anxious whining |
| Attention-seeking | Demanding food, a walk or interaction |
| Other | The dog vocalizes, but in none of the 4 situations above (e.g. resting, bath) |

---

## Research Question

Does fusing all three modalities outperform the best single modality?

---

## Model Structure

```
Uploaded video
      │
      ▼
Preprocessing ──► 16 kHz audio · bark detector (PretrainedSED frame_mn10) → bark segments
      │           · frames sampled per segment
      │
      ├──────────────────────┬───────────────────────────┐
      ▼                      ▼                           ▼
 Audio branch           Body branch                 Scene branch
 frozen BEATs           YOLO11m-pose (fine-tuned)   frozen CLIP ViT-B/32
 → linear heads         → keypoint features         → zero-shot scores,
 (situation, arousal,   → pose decoder (MLP)          aggregated over frames
  valence)
      │ 5 probabilities      │ 5 probabilities           │ 5 probabilities
      └──────────────────────┼───────────────────────────┘
                             ▼
     Learned fusion head: logistic regression on the 15 branch probabilities
     + 3 branch-present flags, with a fitted temperature (models/fusion/)
                             ▼
             Probabilities over the 5 categories per bark
```

Every component starts from a **publicly available pretrained model** and is fine-tuned
only as far as needed.

---

## Results

Fusion collection: 100 self-labeled videos, split by dog into 60 train / 20 val / 20 test
videos. All choices (C, temperature, inputs) were made on train + val; test was scored once.
Numbers are copied from the saved outputs of notebooks 51 and 52.

**Train + val, out-of-fold** ([notebook 51](notebooks/51_fusion_fitting.ipynb); 589
segments, 76 videos):

| Method | Segment log-loss | Segment macro-F1 | Video macro-F1 | Video accuracy |
|---|---|---|---|---|
| audio raw | 1.429 | 0.475 | 0.475 | 0.487 |
| body raw | 1.940 | 0.319 | 0.308 | 0.342 |
| scene raw | 1.949 | 0.171 | 0.155 | 0.237 |
| average raw | 1.398 | 0.399 | 0.452 | 0.461 |
| audio learned | 1.310 | 0.425 | 0.511 | 0.526 |
| body learned | 1.484 | 0.333 | 0.362 | 0.355 |
| scene learned | 1.450 | 0.291 | 0.285 | 0.289 |
| **fusion learned** | **1.089** | **0.556** | **0.564** | **0.566** |

Fusion − audio learned, segment log-loss: -0.219 [-0.332, -0.103] (fusion lower in 100% of
2,000 video-level resamples).

**Test, scored once** ([notebook 52](notebooks/52_fusion_test.ipynb); 111 segments,
18 videos with a segment, 2 videos with none):

| Method | Segment log-loss | Segment macro-F1 | Video macro-F1 | Video accuracy |
|---|---|---|---|---|
| audio raw | 1.518 | 0.410 | 0.371 | 0.389 |
| body raw | 2.177 | 0.300 | 0.291 | 0.278 |
| scene raw | 1.667 | 0.427 | 0.508 | 0.611 |
| average raw | 1.386 | 0.418 | 0.549 | 0.556 |
| audio learned | 1.226 | 0.477 | 0.538 | 0.611 |
| body learned | 1.524 | 0.353 | 0.280 | 0.278 |
| scene learned | 1.075 | 0.588 | 0.628 | 0.611 |
| **fusion learned** | **0.834** | 0.581 | **0.687** | **0.667** |

Fusion video macro-F1, 95% CI: [0.362 0.866]. Fusion − audio learned, 95% CI:
[-0.122 0.344] (fusion ahead in 82% of resamples).

---

## Dataset

All raw datasets are listed in [configs/datasets.yaml](configs/datasets.yaml), with the
source link, pinned revision, expected size and license page of each one. The choice of
datasets and what each one is used for are worked out in
[notebook 00](notebooks/00_data_brainstorming.ipynb).

```bash
python -m woofdata.download            # download every dataset (≈ 8 GB)
python -m woofdata.download <id>       # download one dataset, e.g. esc50
```

Files go to `<data_dir>/raw/<id>/`. Kaggle and Roboflow datasets need API keys, and the
`self_labeled` videos are added by hand.

| Stage | Datasets (`id`) |
|---|---|
| Audio | `playpant`, `audioset_strong_eval`, `esc50`, `act_env`, `emotional_canines` |
| Body (pose, posture, emotion) | `stanford_dogs`, `stanfordextra`, `dog_pose`, `dogposecv`, `dog_emotion_kaggle`, `rf_dog_emotion` |
| Scene | `rf_leash`, `rf_bowl` |
| Fusion and missing categories | `self_labeled` |

---

## Repository structure

| Path | Contents |
|---|---|
| `app/` | FastAPI web app: `api/main.py` (upload jobs, progress, results JSON) and `web/` (plain HTML/CSS/JS page): upload a clip, pick a bark, see each step (preprocessing, audio, body, scene, fusion) and the result |
| `configs/` | `pipeline.yaml` (the settings the app reads) and `datasets.yaml` (dataset list) |
| `data/` | Local datasets and caches (not committed), except the shared splits in `data/splits/` |
| `docs/` | Extra documentation; see [docs/INDEX.md](docs/INDEX.md) |
| `models/` | Model weights; only the small trained heads (`*/heads/`) and `models/fusion/` are committed |
| `notebooks/` | Experiments, numbered by area: 0x data, 1x preprocessing, 2x audio, 3x body, 4x scene, 5x fusion |
| `scripts/` | Cluster job scripts (`slurm/train_pose_encoder.sbatch`) |
| `src/woofdecoder/` | What the app runs: preprocessing, the audio / body / scene branches, fusion, `Pipeline` |
| `src/woofdata/` | Dataset download (`python -m woofdata.download`), preparation (`python -m woofdata.prepare`) and loading |
| `src/wooftrainer/` | Training and evaluation code, one sub-package per area (`<area>_trainer/`), plus shared plots |
| `third_party/` | Cloned PretrainedSED and BEATs source code (not committed) |

---

## Limitations

- **Audio source shortcut.** At first, every distressed_lonely training clip came from
  YouTube and nothing else did, so the audio head partly learned "YouTube audio" instead of
  distress. Balancing sources inside each category (notebook 23) cut non-distressed YouTube
  clips called distressed from 74% to 31%, and on new val dogs none were (notebook 24).
  Some lean toward distressed remains for unclear clips.
- **Valence is near chance.** Valence log-loss 1.036 is only 6% better than a blind guess
  (notebook 22); it is kept in `extras` as a weak hint and should not be shown as a result.
- **Raw scene scores are below chance.** Zero-shot CLIP scores on train + val reach a
  segment macro-F1 of 0.171 and a video macro-F1 of 0.155 (notebook 51), below the 0.2 of
  guessing among 5 categories; only the learned scene head and fusion use them well.
- **Small test set.** Test has 18 videos with a segment, so the confidence intervals are wide
  (fusion video macro-F1 95% CI [0.362 0.866]; fusion − audio learned [-0.122 0.344]).
---

## AI Disclosure

We used AI assistants during this project for the tasks below. All code and results were
reviewed and run by the team.

**General**
- Writing this README.
- Organizing the project architecture.
- The dataset downloader (`python -m woofdata.download`).
- Part of the data preparation (`python -m woofdata.prepare`).

**Audio branch and preprocessing**
- Explaining the results of the notebooks used to tune the model.
- Saving and loading the model.
- How to use the encoder model.
- Planning how to deal with class imbalance and the model taking shortcuts.
- Checking for data leakage.
- Building the matplotlib confusion matrix plots.
- Refactoring to move reused constants, functions and classes into shared modules.
- Debugging.

**Body branch**
- Fine-tuning the pose encoder: how to run it on the servers and how big the model
  should be.
- Planning the data frame structure and the normalization step.

**Scene-Context branch**
- Implementation of CLIP model
- Extraction of bias and variations of weight distribution calibration in the scene branch to improve accuracy outside of training

**Fusion**
- Caching branch outputs per segment so the fusion head can be refit without rerunning
  the branches (notebook 50).
- The video-level bootstrap for the confidence intervals (notebooks 51 and 52).
- Saving and loading the fusion head (`models/fusion/`) for the app.
- Debugging.
