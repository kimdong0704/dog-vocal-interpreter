# Contributing


## 1. Branch and pull-request workflow

**Never commit directly to `main`.** `main` should always work, because the deployed demo and everyone's setup depend on it.

For every task:

1. **Update `main`.** In GitHub Desktop: switch to `main`, then *Fetch origin* → *Pull origin*.
2. **Create a branch** from `main` (*Branch → New branch*). Name it `<type>/<short-description>`:

   | Prefix | Use for | Example |
   |---|---|---|
   | `feature/` | New code or functionality | `feature/bark-detector` |
   | `fix/` | Bug fixes | `fix/segment-off-by-one` |
   | `exp/` | Experiments and notebooks (may never merge) | `exp/audio-linear-probe` |
   | `docs/` | Documentation | `docs/readme-setup` |
   | `chore/` | Setup, config, dependencies, cleanup | `chore/add-github-actions` |
   | `refactor/` | Restructuring code without changing behavior | `refactor/split-scene-branch` |
   | `test/` | Adding or fixing tests only | `test/taxonomy-category-probs` |

   Naming rules:
   - Lowercase letters, numbers and hyphens only: no spaces, capitals or underscores.
   - 2–5 words describing *what* the branch does.
   - If the task has a GitHub Issue, put its number first: `feature/12-bark-detector`.
   - One branch per task. Avoid vague names like `test`, `update` or `leo-stuff`.
3. **Commit and push** to your branch as often as you like.
4. **Before opening a pull request**, run:
```bash
   ruff format .
   ruff check .
   pytest
```
5. **Open a pull request** into `main` (*Branch → Create Pull Request* in GitHub Desktop). Say what changed and why in a few lines.
6. **Get one teammate to review.** The reviewer reads the changes, asks questions, and approves.
7. **Merge**, then delete the branch.

Rules:
- **Never force-push to `main`.**
- Keep pull requests small: one task per PR is easier to review than a week of work.
- If you break `main` by accident, revert the commit (*History → right-click → Revert changes in commit*) and tell the team.

### Commit messages

A short summary in the imperative: `Add bark detector wrapper`, `Fix sample rate in audio loader`. Add a second line with details if needed.

---

## 2. Notebooks decide, the package runs

- **`notebooks/`** are for comparing models, testing ideas, and fine-tuning. They import from `woofdecoder`.
- **`src/woofdecoder/`** holds the code the app actually runs.
- When a notebook settles a choice (which model, which threshold), **move the code into the package** and **put the setting in `configs/pipeline.yaml`**.
- The app reads **only** the package and the config, never a notebook.

---

## 3. Notebook ownership

Notebook numbers show who owns what:

| Range | Area |
|---|---|
| 00 | Setup and data checks |
| 10s | Bark detector |
| 20s | Audio branch |
| 30s | Body language (pose) |
| 40s | Scene context |
| 50s | Fusion |
| 60s | Ablations and report figures |

Ask the owner before editing their notebook. To try something different, copy it instead (for example `21b_audio_finetune_lora.ipynb`).

Notebook outputs are stripped on commit by `nbstripout`. Make sure you ran `nbstripout --install` once on your machine (see the README).

---

## 4. Data rules

- **Never commit data, media, model weights, archives, or keys.** That includes audio, video, images, `.zip`/`.tar`/`.parquet` files, `.pt`/`.ckpt`/`.safetensors` weights, `kaggle.json`, and API tokens. `.gitignore` blocks most of these, but always check `git status` (or the GitHub Desktop file list) before committing.
- **Split by dog, never by clip.** The same dog must never appear in both training and test data. Use the split files in `data/splits/`; don't make your own random splits.
- **YouTube clips:** store only video IDs and timestamps in `data/youtube/ids.csv`. Never commit, upload, or redistribute the clips themselves.
- **Licenses:** on the day you download a dataset or model, screenshot its license page and save it in `data/raw/licenses/` as `<source>_<YYYY-MM-DD>.png`.
- **Non-commercial use:** several datasets and models we use allow academic, non-commercial use only. Don't use them outside this project, and never replay dataset clips in the public demo.