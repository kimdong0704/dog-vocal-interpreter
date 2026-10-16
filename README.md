# WoofDecoder

**Multimodal dog vocalization interpreter**: a web app that estimates what a dog is most
likely communicating when it barks.

| | |
|---|---|
| **Status** | Early development. No results yet |
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
             Probabilities over the 5 categories per bark
```

Every component starts from a **publicly available pretrained model** and is fine-tuned
only as far as needed.
