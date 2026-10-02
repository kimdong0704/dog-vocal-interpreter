# CREMD: Crowd-Sourced Emotional Multimodal Dogs Dataset

- **Authors:** Jinho Baek, Houwei Cao, Kate Blackwell (New York Institute of Technology)
- **Submitted:** 2026-02-17 (arXiv 2602.15349v1, cs.CV)
- **Link:** https://arxiv.org/abs/2602.15349
- **Paper license:** arXiv non-exclusive distribution license (don't redistribute the PDF)
- **Data:** not released. Authors plan an academic release "upon acceptance", with terms
  prohibiting commercial use. Listed as rejected in our dataset report (not public).

## What the dataset is
- 440 videos collected from YouTube, Instagram, TikTok, and Pexels using emotion keywords
  (Table I), about 100 per class; 402 kept after cleaning.
- 351 unique dogs; 89% appear in only one video. Dog metadata: breed, age, color, sex, size.
- Labels: Happy, Angry, Fear, Neutral (+ "Other" with free text). Multi-label, each with
  intensity 1–5.
- Three presentation modes, 923 clips total (average 15.8 s):
  - NCNA, no context, no audio: dog cropped with YOLOv8m, muted (402)
  - YCNA, context, no audio: full frame, muted (402)
  - YCYA, context and audio (119)
  - No "audio without context" mode, because too few videos had clean dog audio.
- 23 annotators (13 F / 10 M, 16 owners, 8 professionals); 12,794 annotations, 15,195 votes;
  over 93% of clips have more than 8 annotations. Custom tool (React + Firebase); 10% of
  clips repeated to measure self-consistency.

## Key findings
- **Visual context improves agreement.** Median agreement 0.69 (NCNA) → 0.71 (YCNA) →
  0.75 (YCYA). NCNA differs from YCNA (p = 0.008) and YCYA (p = 0.003).
- **Audio's effect on agreement is inconclusive.** YCNA vs YCYA is not significant
  (p = 0.173); small audio subset, no audio-only mode.
- **Audio raised confidence:** more intensity-5 ratings for Anger and Fear when audio was present.
- **Common confusions:** Happy ↔ Neutral and Fear ↔ Other without audio; Anger ↔ Fear
  with audio (shared growls, tense posture).
- **Annotator effects:** non-owners agreed more than owners, males more than females, and
  professionals more than non-professionals. Gender and ownership overlap (77% of female
  annotators were owners), so these may be confounded.
- **Self-consistency:** 68% fully consistent, 27% partly, 5% inconsistent.
- Agreement metric: votes for the most common label ÷ total votes, per video.

## What this means for our project
- **Supports multimodal fusion:** context measurably helps humans read dog emotion,
  which backs including the scene branch.
- **Warning for our YouTube plan:** most public videos lacked clean dog audio
  (speech and music overlap). Expect heavy filtering, and run the bark detector before labeling.
- **Threat vs. fearful will be hard:** humans confuse anger and fear even with audio.
  Expect this in our confusion matrix and discuss it in error analysis.
- **Labeling guide:** include at least one non-owner or professional reviewer, measure
  agreement the same way, and repeat ~10% of clips to check consistency.
- **Gap we fill:** CREMD labels emotion as perceived by viewers; we label context from what
  actually happened (owner-reported protocol). Our labels don't depend on how the scene looks.

## Cite for
- Related work (dog emotion datasets, crowd annotation reliability)
- Motivation for multimodal input and for context-based (not perception-based) labels
- Limitations of YouTube audio