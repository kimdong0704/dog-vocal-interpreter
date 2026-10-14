# Team Instructions

Follow the steps in order. For branches and pull requests, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Table of Contents

1. [Initial Setup](#1-initial-setup) (Steps 1–9)
2. [Environment File](#2-environment-file) (Steps 10–11)
3. [Data Download](#3-data-download) (Steps 12–14)
4. [Daily Workflow](#4-daily-workflow)
5. [Data Rules](#5-data-rules)

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
pip install -e ".[dev]"
```

If a browser window asks you to sign in to GitHub, sign in.

### Step 8: Configure notebooks and VS Code

```bash
nbstripout --install
```

In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one in `.venv`.

### Step 9: Verify the setup

```bash
pytest
ruff check .
```

**Check:** all tests pass and ruff reports no problems.

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

## 4. Daily Workflow

1. Activate the virtual environment (Step 6).
2. Pull `main`, then create a branch for your task.
3. Before every push, run:

```bash
   ruff format .
   ruff check .
   pytest
```

4. Open a pull request; one teammate reviews before merging.

---

## 5. Data Rules

| Rule | Details |
|---|---|
| **Never commit data** | Check the changed-files list before every commit |
| **Team-only, non-commercial** | Don't share dataset files outside the team |
| **Training only** | `act_env` and `dog_pose` are never used for test results |
| **Secondary only** | `audioset_strong_eval` scores are optimistic |
| **Split by dog** | Never split by clip |