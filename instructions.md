# Team Instructions

Follow the sections in order. For branches and pull requests, see [CONTRIBUTING.md](CONTRIBUTING.md).

Some files are **not in git** because they are large or not ours to publish. They are kept on
the **team USB drive**:

| On the USB | What it is | Copied to (Section) |
|---|---|---|
| `third_party/` | Code of the bark detector (PretrainedSED) and the audio encoder (BEATs) | `third_party/` (3) |
| `models/` | The current model weights (BEATs, pose model, bark detector) | `models/` (4) |
| `data/raw/self_labeled/` | Our own labelled videos (`audio/` and `fusion/`) | `<data_dir>/raw/self_labeled/` (5) |

`<USB>` below means the USB's project folder, e.g. `E:\woofdecoder` (Windows) or
`/Volumes/USB/woofdecoder` (macOS). Copy files **from** the USB; never edit them on the USB.

---

## Table of Contents

1. [Setup](#1-setup) (Steps 1–8)
2. [Tokens and data folder](#2-tokens-and-data-folder) (Steps 9–10)
3. [Third-party code](#3-third-party-code) (Step 11)
4. [Model weights](#4-model-weights) (Steps 12–13)
5. [Data download](#5-data-download) (Steps 14–17)
6. [Data preparation](#6-data-preparation) (Steps 18–20)
7. [Caching embeddings and the notebooks](#7-caching-embeddings-and-the-notebooks) (Steps 21–24)
8. [Run the app](#8-run-the-app) (Steps 25–26)

**Only want to run the app?** Do Sections 1–4 and 8. Sections 5–7 are needed only to
download and prepare data, or to retrain and reproduce the results.

---

## 1. Setup

### Step 1: Accept the GitHub invites

Open [github.com/notifications](https://github.com/notifications) and accept the invites for
`dog-vocal-interpreter` and `dataset-downloader`.

### Step 2: Install Python 3.12, Git and VS Code

- [Python 3.12.10](https://www.python.org/downloads/release/python-31210/). **Windows:** keep
  **py launcher** checked. **macOS:** same installer, or `pyenv install 3.12.10`.
- [Git](https://git-scm.com/) (or GitHub Desktop).
- [VS Code](https://code.visualstudio.com/) with the **Python** and **Jupyter** extensions.

**Check:** `py --list` (Windows) or `python3.12 --version` (macOS) shows 3.12.

### Step 3: Clone the repository

```bash
git clone https://github.com/<owner>/dog-vocal-interpreter.git
cd dog-vocal-interpreter
```

Do not clone into a OneDrive or iCloud folder.

### Step 4: Create the virtual environment

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `py -3.12 -m venv .venv` | `python3.12 -m venv .venv` |

### Step 5: Activate it

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

**Check:** the prompt starts with `(.venv)`. Repeat this step in every new terminal.

### Step 6: Install the packages

**With an NVIDIA GPU (Windows/Linux):** install the CUDA build of PyTorch first, using the
command for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
Otherwise pip installs the CPU-only build and everything runs much slower.

Then:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,data,ml,app]"
```

If a browser window asks you to sign in to GitHub, sign in (for `dataset-downloader`).

| Extra | Adds | Needed for |
|---|---|---|
| (core) | av, numpy, pillow, pyyaml, soxr | Always installed |
| `dev` | ruff, Jupyter kernel, nbstripout, matplotlib, tqdm | Everyone |
| `data` | dataset-downloader, pandas, pyarrow, soundfile | Downloading and preparing the data (Sections 5–6) |
| `ml` | torch, torchaudio, torchvision, transformers, ultralytics, rfdetr, scikit-learn, scipy, pandas, matplotlib | The models and training |
| `app` | FastAPI, Uvicorn, python-multipart | Running the app (also needs `ml`) |

**Check:** `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
prints `True` at the end if the GPU is used.

### Step 7: Configure notebooks and VS Code

```bash
nbstripout --install
```

In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one in `.venv`.

### Step 8: Verify the setup

```bash
ruff check .
```

**Check:** ruff reports no problems.

---

## 2. Tokens and data folder

### Step 9: Create `.env`

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `Copy-Item .env.example .env` | `cp .env.example .env` |

### Step 10: Fill in the keys

Keep **every line**. Leave a value empty if you don't need it.

| Entry | Fill in? | How to get it |
|---|---|---|
| `HF_TOKEN` | Recommended | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → **Read** |
| `GITHUB_TOKEN` | No | Leave empty |
| `ZENODO_TOKEN` | No | Leave empty |
| `KAGGLE_USERNAME` | Yes | kaggle.com → Settings → API → **Create Legacy API Key** → `username` in `kaggle.json` |
| `KAGGLE_KEY` | Yes | `key` in the same `kaggle.json`; delete the file afterwards |
| `ROBOFLOW_API_KEY` | Yes | app.roboflow.com → Settings → API Keys → **Private API Key** |
| `WOOFDECODER_DATA_DIR` | Optional | Where data, processed files and caches go, e.g. `D:/woofdecoder-data`; empty uses `data/` in the repo |

**Format:** `HF_TOKEN=hf_abc123`, with no spaces, no quotes and no comment on the same line.
**Never commit `.env`.**

`<data_dir>` below means `WOOFDECODER_DATA_DIR`, or `data/` if it is empty.

---

## 3. Third-party code

### Step 11: Copy `third_party/` from the USB

The bark detector (PretrainedSED) and the audio encoder (BEATs) are other people's code, used
as-is. Copy the whole folder into the repo root:

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `Copy-Item -Recurse <USB>\third_party .\third_party` | `cp -R <USB>/third_party ./third_party` |

**Check:** these two files exist:
- `third_party/PretrainedSED/` (a folder with the detector code)
- `third_party/unilm/beats/BEATs.py`

**No USB?** Get the same code from GitHub:

```bash
git clone https://github.com/fschmid56/PretrainedSED.git third_party/PretrainedSED
git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/unilm.git third_party/unilm
git -C third_party/unilm sparse-checkout set beats
```

---

## 4. Model weights

### Step 12: Copy the current models from the USB

The USB holds the **current version** of every large weight file. Copy its `models/` folder
over the repo's `models/` folder (it only adds files; the committed heads stay):

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `Copy-Item -Recurse -Force <USB>\models\* .\models\` | `cp -R <USB>/models/. ./models/` |

### Step 13: Check every file is in place

| File | What it is | Where it comes from |
|---|---|---|
| `models/audio/beats/BEATs_iter3_plus_AS2M.pt` | Audio encoder (about 360 MB) | USB |
| `models/pose/yolo11m-pose_full/best.pt` | Pose model (about 43 MB) | USB |
| `models/pretrained_sed/frame_mn10_strong_1.pt` | Bark detector (about 15 MB) | USB (also downloads itself on first use) |
| `models/audio/heads/*.npz` | Audio heads (4 files) | Git (committed) |
| `models/pose/heads/pose_decoder_64.pt` | Pose decoder | Git (committed) |
| `models/fusion/fusion_head.npz`, `fusion_info.json` | Fusion model | Git (committed) |
| CLIP `openai/clip-vit-base-patch32`, RF-DETR, Ultralytics base weights | Scene model and detectors | Download automatically on first use (needs internet once) |

The paths the app uses are in `configs/pipeline.yaml`. If you retrain a model, save it to the
same path, and put the new version on the USB so the team gets it.

---

## 5. Data download

### Step 14: Download the public datasets

| Command | Downloads |
|---|---|
| `python -m woofdata.download` | Every public dataset (≈ 8 GB; keep 15 GB free) |
| `python -m woofdata.download <id>` | One dataset (ids are in `configs/datasets.yaml`) |

Files are saved to `<data_dir>/raw/<id>/`. The `self_labeled` entry is skipped: it is not
downloaded (Step 16).

### Step 15: Check the results

Every row in the results table should say `finished` or `already downloaded`. Otherwise:

| Status | What to do |
|---|---|
| `interrupted` | Run the same command again |
| `needs token` | Fill in the `.env` entry named in the reason, then run again |
| `needs manual access` | Open the dataset page in a browser and accept the terms |
| `failed` "Roboflow is preparing the export" | Wait 3–5 minutes, then run that id again |
| `failed` (other) | Post the row in the team chat |

Rerunning is always safe: finished files are skipped and unfinished files resume. The same
rows are saved in `<data_dir>/raw/download_log.csv`.

### Step 16: Copy our self-labelled videos from the USB

Our own labelled videos are stored separately, on the USB:

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `Copy-Item -Recurse <USB>\data\raw\self_labeled <data_dir>\raw\self_labeled` | `cp -R <USB>/data/raw/self_labeled <data_dir>/raw/self_labeled` |

**Check:** the layout is

```
<data_dir>/raw/self_labeled/
  audio/    labels.csv + videos/   audio-branch clips, all 5 categories (train / val by dog)
  fusion/   labels.csv + videos/   fusion videos, 20 per category (train / val / test by dog)
```

`labels.csv` columns: `video_id, source_url, dog_id, start_s, end_s, category, labeller`
(fusion also has `dog_visible`). Video files are `videos/<video_id>.mp4` (lower-case `.mp4`).
To add clips, see the docstring of `src/woofdata/prepare/sources/self_labeled.py`, and copy
the updated folder back to the USB.

### Step 17: Save the license pages

For each public dataset, screenshot its license page (links in the README's *Datasets*
section) and save it as `data/raw/licenses/<id>_<YYYY-MM-DD>.png`.

---

## 6. Data preparation

### Step 18: Prepare the datasets

| Command | Does |
|---|---|
| `python -m woofdata.prepare` | Prepares every dataset, in `configs/datasets.yaml` order (including `self_labeled`) |
| `python -m woofdata.prepare <id>` | Prepares one dataset |
| `python -m woofdata.prepare --dry-run` | Lists each dataset, what is in `raw/` and what would be written; writes nothing |

Each dataset is converted from `<data_dir>/raw/<id>/` into `<data_dir>/processed/<id>/`
(16 kHz WAVs, images, COCO JSON, `labels.csv`, `rows.csv`). Then
`<data_dir>/processed/manifest.csv` and the split files in `data/splits/` are rebuilt.

- `raw/` is never changed, and rerunning replaces only `processed/<id>/`.
- Prepare `stanford_dogs` before `stanfordextra`, `dog_pose` and `dogposecv`: they read its
  `index.csv` and `boxes.csv`. Running without an id does this for you.

### Step 19: Check the results

Each dataset prints one line:

| Line says | What to do |
|---|---|
| `done: … (fingerprint match)` | Nothing: your output is identical to the team's |
| `done: … (fingerprint recorded)` | First time this dataset was prepared; commit `data/splits/` |
| `WARNING: fingerprint differs …` | Your output differs from the team's: don't commit, post the line in the team chat |
| `skipped (raw folder missing …)` | Download (Step 14) or copy (Step 16) it first |
| `no converter yet` | Nothing: this dataset is not prepared yet |

Only after a converter or the self-labelled data was changed **on purpose**, accept the new
output with `python -m woofdata.prepare <id> --update-fingerprints` and commit `data/splits/`.

### Step 20: Verify against the team's fingerprints

```bash
python -m woofdata.prepare --check
```

**Check:** it prints `match`. Otherwise it lists each dataset that differs or was never
prepared.

To load prepared data in Python: `task_rows("<task>", "<split>")` and
`load_sample("<task>", row)` (tasks are listed in `src/woofdata/tasks.py`). For PyTorch,
`TaskDataset` in `wooftrainer` wraps the same rows.

---

## 7. Caching embeddings and the notebooks

Needed only to **retrain** a model or **reproduce** the results; the app does not need any
cache. Caches go to `<data_dir>/cache/` and are never committed. Every caching cell resumes
where it stopped, so rerunning is safe. Use a GPU: on CPU these take hours.

Run the notebooks with the `.venv` kernel, **top to bottom**, and restart the kernel after any
code change in `src/`.

### Step 21: Audio embeddings (notebook 20)

`notebooks/20_audio_encoder_comparison.ipynb`: run the setup cells, the
**Caching resulting embeddings** cell, then **Check if cache match the manifest** (every
`Matches Manifest` and `Finite` is `True`). Skip the **Layer Probe** cells: the layer decision
is already recorded there.

Writes `<data_dir>/cache/audio/beats_iter3_plus_as2m/` (one `.npz` per task and split;
Act_Env takes the longest). After new self-labelled clips are added and prepared, rerun the
caching cell: only the new clips are embedded.

### Step 22: Audio heads (notebooks 22–24)

These read the notebook 20 cache; they cache nothing themselves.

| Notebook | Does |
|---|---|
| `22_audio_heads_trainer_check` | Checks the training code on the cached embeddings |
| `23_train_audio_heads` | Trains the situation, arousal and valence heads (source-balanced) and saves them to `models/audio/heads/` |
| `24_eval_audio_heads` | Scores the saved heads once on val |

### Step 23: Fusion cache, fusion model and test (notebooks 50–52)

| Notebook | Does |
|---|---|
| `50_fusion_dataset_cache` | `cache_branch_outputs()` runs the bark detector and the audio, body and scene branches on every fusion video (15–30 min on a GPU) → `<data_dir>/cache/fusion/` |
| `51_fusion_fitting` | Compares fusion with each single branch on train + val (from the cache) and saves `models/fusion/` |
| `52_fusion_test` | Scores the saved fusion model **once** on the fusion test videos |

If the scene settings (prompts, aggregation, temperature, bias) change, delete
`<data_dir>/cache/fusion/` and rerun notebook 50: the cache refuses to mix versions.

### Step 24: Other branches' notebooks

Bark detector: notebooks 12–15. Body: 30–32. Scene: 40–43. Each notebook's first cell says
what it reads and writes. Two of them also write caches:

| Notebook | Cache |
|---|---|
| `12_bark_detector`, `13_tuning_bark_detector` | Detector scores in `<data_dir>/cache/bark_detector/` (13 makes any that are missing) |
| `31_body_language_probe` | Keypoints in `<data_dir>/cache/body/pose.pkl` |

---

## 8. Run the app

### Step 25: Start it

From the repo root, with the virtual environment active:

```bash
uvicorn app.api.main:app
```

The first start loads every model (and downloads CLIP once), so it takes a while.
Then open <http://127.0.0.1:8000>.

### Step 26: Use it

1. Choose a short video of a dog (or an audio file, up to 200 MB) and press **Analyze**.
   A progress bar shows the current step ("Detecting barks…", "Segment 3/7: scene…").
   Clips run one at a time; a second upload waits until the first is done.
2. The video appears with one row per detected vocalization: its time range, the fused
   category and probability, and which branches (audio, body, scene) could run.
   If no dog sound is found, it says so.
3. Click a row to jump the video to that segment and open its step-by-step explanation:
   - **Preprocessing:** time range, bark-detector confidence, sound types, sampled frames.
   - **Audio:** category, arousal and valence scores (valence is near chance on unseen dogs)
     and the emotion-nudge shifts.
   - **Body:** category scores, or why it was not used (no video, or no frame with exactly one
     dog); which frames it used; tail, ear and head height.
   - **Scene:** category scores, how frames were combined, and the best prompt per frame.
   - **Fusion:** the 18 inputs, how much each branch pushed each category, and
     logits ÷ temperature → probabilities.
   - **Result:** the final category and all 5 probabilities.

Uploaded files and results are deleted after 1 hour (or when the server stops). Test it with
`data/test/media_test.mp4`; `pytest tests/test_pipeline.py` runs the same clip without the page.

Keep demo clips short (under a minute) and **never use dataset clips in a public demo**
(see Data rules).

**Problems?**

| You see | Do |
|---|---|
| `FileNotFoundError` for a model or `third_party` file | Redo Sections 3–4 |
| Very slow | Check `torch.cuda.is_available()` (Step 6) |
| `ModuleNotFoundError` | Activate `.venv` (Step 5) and rerun Step 6 |
| Startup error: fusion head trained on other scene settings | `branches.scene` in `configs/pipeline.yaml` changed since notebook 51: undo it, or rerun notebooks 50–51 (Step 23) |