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

## Downloading the datasets

All datasets are downloaded with one command. The downloader is resumable: if it stops, run it again and finished files are skipped.

### 1. Install the download dependencies

With the virtual environment active:

```bash
pip install -e ".[dev,data]"
```

### 2. (Optional) Set up your `.env` file

Copy the template and fill in what you need:

```powershell
Copy-Item .env.example .env     # Windows
```

```bash
cp .env.example .env            # macOS / Linux
```

| Setting | What it does |
|---|---|
| `HF_TOKEN` | Your Hugging Face read token ([create one here](https://huggingface.co/settings/tokens)). Optional: removes the "unauthenticated requests" warning and raises rate limits. |
| `WOOFDECODER_DATA_DIR` | Where to store data, e.g. `D:/woofdecoder-data`. Leave empty to use `data/` inside the repo. |

**Never commit `.env`.** It holds your personal token.

### 3. Download

```bash
python -m woofdecoder.data.download --list          # show available datasets
python -m woofdecoder.data.download                 # download everything (~6 GB)
python -m woofdecoder.data.download --only act_env  # download specific datasets
```

Each dataset goes into `<data_dir>/raw/<dataset_name>/`, and `<data_dir>/raw/download_log.json` records the license, version and files for every download. Files are downloaded as-is (zips stay zipped); extraction is a later preprocessing step.

Make sure you have about **15 GB free**: 6 GB for the downloads plus room to unzip later.

### 4. Save the license pages

On the day you download, screenshot each dataset's license page (links below) into `data/raw/licenses/` as `<dataset_name>_<YYYY-MM-DD>.png`. Licenses can change, and this is our record of the terms we downloaded under.

### Datasets (batch 1, ≈ 6.0 GB total)

| Name | Dataset | Used for | Why we picked it | Size | License |
|---|---|---|---|---|---|
| `playpant` | [Dog play pant](https://zenodo.org/records/18972388) (audio, annotations, metadata) | Bark detector validation; play test set | The only open dataset with expert-labeled bark/growl/whine timings **and** dog IDs (17 dogs), recorded in a lab, so it doesn't overlap with the pretrained models' training data | 0.38 GB | CC BY-NC 4.0 |
| `act_env` | [Barkopedia Activity & Environment](https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env) | Audio branch training | The largest public set of barks labeled by situation (alerting, begging, seeking attention, playing), covering 4 of our 9 subtypes. Labels came from a model watching the video, so it's **training only** | 0.61 GB | MIT |
| `emotional_canines` | [EmotionalCanines](https://huggingface.co/datasets/ArlingtonCL2/BarkopediaDogEmotionClassification_Data) | Audio emotion head (arousal/valence) | The only open dog-audio set labeled by emotion, with different dogs in train and validation | 0.07 GB | CC BY-NC-SA 4.0 |
| `audioset_strong_eval` | [AudioSet-Strong-Animals](https://huggingface.co/datasets/enyoukai/AudioSet-Strong-Animals) (eval split) | Bark detector validation | Precise start/end times for bark, growl, whimper and howl in the same label set our bark detector outputs. Scores will be optimistic, since the detector was trained on AudioSet | 3.0 GB | Labels CC BY 4.0 (YouTube media) |
| `dog_pose` | [Ultralytics Dog-Pose](https://docs.ultralytics.com/datasets/pose/dog-pose/) | Pose baseline | One-file download of StanfordExtra keypoints, including **tail tip and ear tips**, for quick pose testing | 0.35 GB | AGPL-3.0 zip (MIT upstream) |
| `dogposecv` | [DogPoseCV](https://huggingface.co/datasets/stockeh/dog-pose-cv) | Body-language features | Human-labeled standing/sitting/lying posture, a cleaner signal than web "dog emotion" labels | 0.82 GB | Apache-2.0 labels; ImageNet images (non-commercial) |
| `stanford_dogs` | [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) (images + lists) | Images for pose fine-tuning | The source images for StanfordExtra pose labels, 120 breeds | 0.76 GB | ImageNet terms (non-commercial) |

**Left out on purpose:** the Dog play pant raw multi-camera video (43 GB, only needed for a later fusion experiment) and the Barkopedia test split (its labels are hidden).

### Rules for downloaded data

- **Never commit data.** `data/` is gitignored, but check the changed-files list before committing.
- **Non-commercial use only.** Can not be shared outside the team and never be used in the public demo.
- **Training-only sets:** `act_env` and `dog_pose` are never used to report test results (see CONTRIBUTING.md).

### Adding a dataset

Add one `Dataset` entry to `REGISTRY` in `src/woofdecoder/data/download_reg.py`. Nothing else needs to change. Include the license and license URL, and pin a `revision` for Hugging Face datasets when possible.

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