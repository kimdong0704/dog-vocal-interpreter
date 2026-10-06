# WoofDecoder

**Multimodal dog vocalization interpreter**: a web app that estimates what a dog is most
likely communicating when it barks.

> **Disclaimer:** WoofDecoder estimates the likely *context* of barking from patterns in
> sound and video. It does **not** translate dog language.

| | |
|---|---|
| **Status** | Early development. No results yet |
| **Course** | CSSE 473 Deep Learning, course project |
| **Team** | Leo K. · Eden K. · Elijah H. · Rohan M. · Jackson O. |

---

## Table of Contents
1. [Motivation](#motivation)
2. [Output Taxonomy](#output-taxonomy)
3. [System Architecture](#system-architecture)
4. [Datasets](#datasets)
5. [Technology Stack](#technology-stack)
6. [Repository Structure](#repository-structure)
7. [License](#license)

---

## Motivation

People read a barking dog with more than their ears. WoofDecoder does the same with
three signals:

| Modality | Question it answers | Example cues |
|---|---|---|
| **Audio** | *How* is the dog vocalizing? | bark vs. growl vs. whine, pitch, rhythm |
| **Body language** | *What state* is the dog in? | tail height, ear position, posture |
| **Scene context** | *Why* might it be barking? | person at the door, ball, food bowl |

**Research question:** does fusing all three modalities outperform the best single modality?

---

## Output Taxonomy

Each bark is classified into a **subtype**. A category's confidence is the sum of its
subtypes' probabilities.

| Category | Subtypes |
|---|---|
| Alert / Aggressive | Stranger alert · Threat · Fearful / defensive |
| Playful / Excited | Play · Greeting |
| Distressed / Lonely | Separation · Anxious / whining |
| Attention-seeking | Demand (food, walk, let out) · Interaction with owner |

---

## System Architecture

```
Uploaded video
      │
      ▼
Preprocessing ──► extract audio · detect bark segments · sample frames per bark
      │
      ├──────────────────────┬───────────────────────┐
      ▼                      ▼                       ▼
 Audio model           Body-language model        Scene model
 pretrained audio      pretrained pose model →    zero-shot image-text model
 encoder               posture features           + object detector
      └──────────────────────┼───────────────────────┘
                             ▼
                Fusion model (+ calibration)
                             ▼
          Subtype and category probabilities per bark
```

Every component starts from a **publicly available pretrained model** and is fine-tuned
only as far as needed.

---

## Datasets

Public datasets for training and evaluation, plus clips collected by the team.
All datasets are used for academic, non-commercial purposes and are **not redistributed**.

### Audio Datasets

| Dataset | Purpose | License |
|---|---|---|
| [Dog play pant](https://zenodo.org/records/18972388) | Bark detector validation · play test set | CC BY-NC 4.0 |
| [AudioSet-Strong-Animals](https://huggingface.co/datasets/enyoukai/AudioSet-Strong-Animals) (eval) | Bark detector validation (secondary) | Labels CC BY 4.0 |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | Non-bark sounds · noise for augmentation | CC BY-NC 3.0 |
| [Barkopedia Activity & Environment](https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env) | Audio model training | MIT |
| [EmotionalCanines](https://huggingface.co/datasets/ArlingtonCL2/BarkopediaDogEmotionClassification_Data) | Audio emotion head | CC BY-NC-SA 4.0 |

### Body-Language Datasets

| Dataset | Purpose | License |
|---|---|---|
| [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) | Images for pose training | ImageNet terms (non-commercial) |
| [StanfordExtra](https://github.com/benjiebob/StanfordExtra) | Dog keypoints (tail, ears) | MIT |
| [Ultralytics Dog-Pose](https://docs.ultralytics.com/datasets/pose/dog-pose/) | Pose baseline | AGPL-3.0 zip (MIT upstream) |
| [DogPoseCV](https://huggingface.co/datasets/stockeh/dog-pose-cv) | Body posture | Apache-2.0 labels |

### Scene-Context Datasets

| Dataset | Purpose | License |
|---|---|---|
| [Roboflow ai-stray-dog-project](https://universe.roboflow.com/rusydan/ai-stray-dog-project) | Leash detection | CC BY 4.0 |
| [Roboflow cat-dog-bowl](https://universe.roboflow.com/eyedetection-s8fd9/cat-dog-bowl-ncx2x) | Food bowl detection | CC BY 4.0 |

### Team-Collected Data
Videos of dogs barking in known situations, collected with owner consent. They are the
**primary test set** and the only data containing all three modalities together.

---

## Technology Stack

| Layer | Tools |
|---|---|
| Models | PyTorch · Hugging Face Transformers · pretrained audio, pose and vision models |
| Backend | FastAPI |
| Frontend | Gradio (prototype) → React (final) |
| Data | [dataset-downloader](https://github.com/kimdong0704/dataset-downloader) (parallel, resumable downloads) |

---

## Repository Structure

```
dog-vocal-interpreter/
├── src/woofdecoder/    # the package the app runs
│   ├── taxonomy.py     #   categories and subtypes
│   ├── pipeline.py     #   video in → predictions out
│   ├── data/           #   dataset download, splits
│   ├── preprocess/     #   audio/frame extraction, bark segments
│   ├── branches/       #   audio, body-language, scene models
│   ├── fusion/         #   combining models, calibration
│   └── eval/           #   metrics
├── app/                # backend and web UI
├── notebooks/          # experiments
├── configs/            # model and dataset settings
├── tests/              # automated tests
└── docs/               # documentation
```

---

## License

No license yet. Some models and datasets are licensed for **non-commercial use only**, so a
license will be chosen before any public release. Until then, this project is for academic,
non-commercial use.