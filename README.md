# WoofDecoder: Multimodal Dog Vocalization Interpreter

A web app that estimates what a dog is most likely communicating when it barks. It combines three signals: the **sound** of the bark, the dog's **body language** (tail, ears, posture), and the **scene** (a person at the door, a ball, a food bowl). Each signal has its own model built on publicly available pretrained models, and a fusion model combines them.

> **Disclaimer:** Estimates the likely context of barking from patterns in sound and video. It does not translate dog language.

**Status:** early development (week 1). No results yet.

Deep Learning course team project: Leo K., Eden K., Elijah H., Rohan M., Jackson O.

---

## What it predicts

The models predict a **subtype**. Each category's confidence is the sum of its subtypes' probabilities.

| Category | Subtypes |
|---|---|
| Alert / Aggressive | Stranger alert · Threat · Fearful / defensive |
| Playful / Excited | Play · Greeting |
| Distressed / Lonely | Separation · Anxious / whining |
| Attention-seeking | Demand (food, walk, let out) · Interaction with owner |

The code version of this list lives in `src/woofdecoder/taxonomy.py`.

---

## Setup

You need **Python 3.12.10** (the project requires 3.12, not 3.13 or 3.14), Git, and VS Code with the Python and Jupyter extensions.

### 1. Install Python 3.12.10

- **Windows:** download the 3.12.10 installer from [python.org](https://www.python.org/downloads/release/python-31210/). Keep the **py launcher** option checked. Check that it worked with `py --list`; you should see `-V:3.12`.
- **macOS:** use the 3.12.10 installer from the same page, or `pyenv install 3.12.10`.

### 2. Clone the repo

Use GitHub Desktop (*File → Clone repository*) or:

```bash
git clone https://github.com/<owner>/dog-vocal-interpreter.git
cd dog-vocal-interpreter
```

Avoid cloning into a OneDrive or iCloud folder; syncing large data folders causes problems.

### 3. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

### 4. Install the package

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

This installs `woofdecoder` in editable mode, so code changes take effect without reinstalling. Later in the project you'll also need the model and app extras:

```bash
pip install -e ".[dev,ml,app]"
```

### 5. Turn on notebook output stripping

```bash
nbstripout --install
```

This keeps notebook outputs out of commits. It is set up **per machine**, so everyone has to run it once.

### 6. Select the interpreter in VS Code

Press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac), choose **Python: Select Interpreter**, and pick the one inside `.venv`.

### 7. Check that everything works

```bash
pytest
ruff check .
```

All tests should pass and ruff should report no problems.

---

## Folder guide

```
dog-vocal-interpreter/
├── src/woofdecoder/     # the installable package the app runs
│   ├── taxonomy.py      #   categories and subtypes
│   ├── config.py        #   loads configs/pipeline.yaml
│   ├── pipeline.py      #   end-to-end: video in, predictions out
│   ├── data/            #   downloads, manifest, dog-level splits
│   ├── preprocess/      #   audio/frame extraction, bark segments
│   ├── branches/        #   bark detector, audio, body-language, scene models
│   ├── fusion/          #   combining branches, calibration
│   └── eval/            #   metrics
├── app/
│   ├── api/             # FastAPI backend
│   └── web/             # prototype UI (Gradio)
├── notebooks/           # experiments: compare, test, fine-tune
├── configs/             # pipeline.yaml: chosen models and settings
├── tests/               # pytest tests
├── docs/                # team docs (reference notes)
├── data/                # local data, not committed
└── models/              # local weights, not committed
```

---

## Running tests

```bash
pytest            # run all tests
pytest -v         # show each test name
ruff format .     # format code
ruff check .      # lint
```

Run all three (`ruff format .`, `ruff check .`, `pytest`) before every push.

---

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and pull-request workflow, notebook ownership, and data rules, and [docs/CODINGSTYLE.md](docs/CODINGSTYLE.md) for how we write code.

---

## License

No license yet. Some models and datasets in the planned stack are licensed for non-commercial use only, so the license will be chosen before the repo is made public. Until then, this project is for academic, non-commercial use.