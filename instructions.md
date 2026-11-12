# Team Instructions

Follow the steps in order. For branches and pull requests, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Table of Contents

1. [Initial Setup](#1-initial-setup) (Steps 1–9)
2. [Environment File](#2-environment-file) (Steps 10–11)
3. [Data Download](#3-data-download) (Steps 12–14)
4. [Data Preparation](#4-data-preparation) (Steps 15–18)
5. [Model Weights](#5-model-weights) (Step 19)
6. [Daily Workflow](#6-daily-workflow)
7. [Data Rules](#7-data-rules)

---

## 1. Initial Setup

### Step 1: Accept the GitHub invites

Open [github.com/notifications](https://github.com/notifications) and accept the invites for
`dog-vocal-interpreter` and `dataset-downloader`.

### Step 2: Install Python 3.12.10

Download it from [python.org](https://www.python.org/downloads/release/python-31210/).

- **Windows:** keep **py launcher** checked.
- **macOS:** same installer, or `pyenv install 3.12.10`.

**Check:** `py --list` shows `-V:3.12`.

### Step 3: Install Git and VS Code

- [Git](https://git-scm.com/) (or GitHub Desktop)
- [VS Code](https://code.visualstudio.com/) with the **Python** and **Jupyter** extensions

### Step 4: Clone the repository

GitHub Desktop: **File → Clone repository → dog-vocal-interpreter**, or:

```bash
git clone https://github.com/<owner>/dog-vocal-interpreter.git
cd dog-vocal-interpreter
```

Do not clone into a OneDrive or iCloud folder.

### Step 5: Create the virtual environment

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `py -3.12 -m venv .venv` | `python3.12 -m venv .venv` |

### Step 6: Activate the virtual environment

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

**Check:** the prompt starts with `(.venv)`. Repeat this step in every new terminal.

### Step 7: Install the package

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,data]"
```

If a browser window asks you to sign in to GitHub, sign in.

| Extra | Adds | Needed for |
|---|---|---|
| (core) | av, numpy, pillow, pyyaml, soxr | Always installed |
| `dev` | ruff, Jupyter kernel, nbstripout, matplotlib, tqdm | Everyone |
| `data` | dataset-downloader, pandas, pyarrow, soundfile | Downloading and preparing the data (Sections 3–4) |
| `ml` | torch, torchaudio, torchvision, transformers, ultralytics, rfdetr, scikit-learn, scipy, pandas, matplotlib | The models (`woofdecoder` branches) and training (`wooftrainer`) |
| `app` | FastAPI, Uvicorn, python-multipart | Running the app (also needs `ml`) |

Add extras as you need them; everything at once is `pip install -e ".[dev,data,ml,app]"`.

### Step 8: Configure notebooks and VS Code

```bash
nbstripout --install
```

In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one in `.venv`.

### Step 9: Verify the setup

```bash
ruff check .
```

**Check:** ruff reports no problems.

---

## 2. Environment File

### Step 10: Create `.env`

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `Copy-Item .env.example .env` | `cp .env.example .env` |

### Step 11: Fill in the keys

Keep **every line**. Leave a value empty if you don't need it.

| Entry | Fill in? | How to get it |
|---|---|---|
| `HF_TOKEN` | Recommended | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → **Read** |
| `GITHUB_TOKEN` | No | Leave empty |
| `ZENODO_TOKEN` | No | Leave empty |
| `KAGGLE_USERNAME` | Yes | kaggle.com → Settings → API → **Create Legacy API Key** → `username` in `kaggle.json` |
| `KAGGLE_KEY` | Yes | `key` in the same `kaggle.json`; delete the file afterwards |
| `ROBOFLOW_API_KEY` | Yes | app.roboflow.com → Settings → API Keys → **Private API Key** |
| `WOOFDECODER_DATA_DIR` | Optional | e.g. `D:/woofdecoder-data`; empty uses `data/` |

**Format:** `HF_TOKEN=hf_abc123`, with no spaces, no quotes and no comment on the same line.
**Never commit `.env`.**

---

## 3. Data Download

### Step 12: Download the datasets

| Command | Downloads |
|---|---|
| `python -m woofdata.download` | Every dataset (≈ 8 GB; keep 15 GB free) |
| `python -m woofdata.download <id>` | One dataset (ids are in `configs/datasets.yaml`) |

Files are saved to `<data_dir>/raw/<id>/`.

### Step 13: Check the results

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

### Step 14: Save the license pages

For each dataset, screenshot its license page (links in the README's *Datasets* section) and
save it as `data/raw/licenses/<id>_<YYYY-MM-DD>.png`.

---

## 4. Data Preparation

### Step 15: Prepare the datasets

| Command | Does |
|---|---|
| `python -m woofdata.prepare` | Prepares every downloaded dataset, in `configs/datasets.yaml` order |
| `python -m woofdata.prepare <id>` | Prepares one dataset |
| `python -m woofdata.prepare --dry-run` | Lists each dataset, what is in `raw/` and what would be written; writes nothing |

Each dataset is converted from `<data_dir>/raw/<id>/` into `<data_dir>/processed/<id>/`
(16 kHz WAVs, images, COCO JSON, `labels.csv`, `rows.csv`). Then
`<data_dir>/processed/manifest.csv` and the split files in `data/splits/` are rebuilt.

- `raw/` is never changed, and rerunning replaces only `processed/<id>/`.
- Prepare `stanford_dogs` before `stanfordextra`, `dog_pose` and `dogposecv`: they read its
  `index.csv` and `boxes.csv`. Running without an id does this for you.

### Step 16: Check the results

Each dataset prints one line:

| Line says | What to do |
|---|---|
| `done: … (fingerprint match)` | Nothing: your output is identical to the team's |
| `done: … (fingerprint recorded)` | First time this dataset was prepared; commit `data/splits/` |
| `WARNING: fingerprint differs …` | Your output differs from the team's: don't commit, post the line in the team chat |
| `skipped (raw folder missing …)` | Download it first (Step 12) |
| `no converter yet` | Nothing: this dataset is not prepared yet |

Only after a converter was changed **on purpose**, accept the new output with
`python -m woofdata.prepare <id> --update-fingerprints` and commit `data/splits/`.

### Step 17: Verify against the team's fingerprints

```bash
python -m woofdata.prepare --check
```

**Check:** it prints `match`. Otherwise it lists each dataset that differs or was never
prepared.

### Step 18: Load a sample

```python
from woofdata.loader import load_sample
from woofdata.tasks import task_rows

rows = task_rows("pose", "train")  # manifest rows of one task and split
sample = load_sample("pose", rows.iloc[0])  # image, box and keypoints
```

Tasks are `bark_detector`, `audio_context`, `audio_emotion`, `pose`, `posture` and
`scene_objects` (see `src/woofdata/tasks.py`). For PyTorch, `wooftrainer.torch_datasets.TaskDataset`
wraps the same rows (needs the `ml` extra).

---

## 5. Model Weights

### Step 19: Get the weights that are not committed

Only small trained files are committed: the audio heads (`models/audio/heads/`), the pose
decoder (`models/pose/heads/pose_decoder_64.pt`) and the fusion head (`models/fusion/`).
Get the rest once:

| File | How to get it |
|---|---|
| `third_party/PretrainedSED/` (bark detector code) | `git clone https://github.com/fschmid56/PretrainedSED.git third_party/PretrainedSED` |
| `models/pretrained_sed/frame_mn10_strong_1.pt` (bark detector, about 15 MB) | Downloaded automatically the first time the bark detector loads |
| `third_party/unilm/beats/` (BEATs code) | `git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/unilm.git third_party/unilm`, then `git -C third_party/unilm sparse-checkout set beats` |
| `models/audio/beats/BEATs_iter3_plus_AS2M.pt` (audio encoder) | Download `BEATs_iter3_plus_AS2M.pt` from the checkpoint table in the [BEATs README](https://github.com/microsoft/unilm/tree/master/beats) and save it at this path |
| `models/pose/yolo11m-pose_full/best.pt` (pose model) | Ask the team for the trained file, or retrain with `scripts/slurm/train_pose_encoder.sbatch` (notebook 30) and copy its `weights/best.pt` to this path |
| CLIP `openai/clip-vit-base-patch32`, RF-DETR, Ultralytics base weights | Downloaded automatically on first use |

**Check:** `uvicorn app.api.main:app` starts and serves
<http://127.0.0.1:8000>.

---

## 6. Daily Workflow

1. Activate the virtual environment (Step 6).
2. Pull `main`, then create a branch for your task.
3. Before every push, run:

```bash
   ruff format .
   ruff check .
```

4. Open a pull request; one teammate reviews before merging.

---

## 7. Data Rules

| Rule | Details |
|---|---|
| **Never commit data** | Check the changed-files list before every commit |
| **Team-only, non-commercial** | Don't share dataset files outside the team |
| **Training only** | `act_env` and `dog_pose` are never used for test results |
| **Secondary only** | `audioset_strong_eval` scores are optimistic |
| **Split by dog** | Never split by clip |