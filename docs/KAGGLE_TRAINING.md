# Sentinel-X — Kaggle Training Guide (Phase A: train on `ctu-13-full`)

Goal: train the world model on **all 13 CTU-13 scenarios** (48k windows, 6 botnet
families, scenario-disjoint split) on a free Kaggle GPU, then bring the checkpoint
back here to evaluate against the baseline bar.

The local dev build (`ctu-13`, scenario 11, 195 windows) stays untouched.

---

## 0. What gets uploaded (and what does NOT)

The trainer reads **only** `windows/*.jsonl` (see `worldmodel/data.py`). So we
skip the big `graphs/*.pt` (~530 MB) and the PCAPs (~72 GB) entirely.

Run locally once:

```
python scripts/pack_kaggle.py --dataset ctu-13-full
```

Produces `dist/kaggle/`:
- `sentinelx_code.zip` (~0.2 MB) — src/, scripts/, configs/, pyproject.toml
- `sentinelx_ctu13full_windows.zip` (~158 MB) — windows/ + metadata/ only

---

## 1. Create the Kaggle Dataset

1. kaggle.com -> Datasets -> **New Dataset**.
2. Drag in BOTH zips. Kaggle auto-unzips them.
3. Title it e.g. `sentinelx-ctu13full`. Create.

Resulting layout on Kaggle (under `/kaggle/input/sentinelx-ctu13full/`):
```
src/ scripts/ configs/ pyproject.toml
data/processed/ctu-13-full/windows/{train,val,test}.jsonl
data/processed/ctu-13-full/metadata/manifest.json
```

---

## 2. Create the Notebook

New Notebook -> **Add Input** -> your `sentinelx-ctu13full` dataset.
Settings -> Accelerator -> **GPU T4 x2** (or P100). Internet: ON (for pip).

### Cell 1 — set up a writable working copy
```python
import shutil, os
SRC = "/kaggle/input/sentinelx-ctu13full"
WORK = "/kaggle/working/sentinelx"
shutil.copytree(SRC, WORK, dirs_exist_ok=True)
os.chdir(WORK)
print(os.listdir(WORK))
print(os.listdir(f"{WORK}/data/processed/ctu-13-full/windows"))
```

### Cell 2 — install deps (CPU torch already on Kaggle; add PyG)
```python
!pip -q install torch-geometric>=2.4
!pip -q install -e .            # installs the sentinelx package (deps=[] core)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
```

### Cell 3 — train on ctu-13-full
```python
!python scripts/train.py \
  --config configs/world_model.yaml \
  --dataset ctu-13-full \
  --combos gat_gru,gat_lstm,graphsage_gru,graphsage_lstm \
  --epochs 50 --seed 42 \
  --processed-root /kaggle/working/sentinelx/data/processed \
  --models-dir /kaggle/working/models/sentinel_x \
  --experiments-dir /kaggle/working/experiments
```

Notes:
- `--processed-root` must point at the folder that CONTAINS `ctu-13-full/`.
- All four combos train and the best (by val future-state loss) is saved.
- If you only want the strongest combo to save time, use `--combos gat_gru`.

### Cell 4 — zip the outputs to download
```python
import shutil
shutil.make_archive("/kaggle/working/sentinelx_trained", "zip",
                    "/kaggle/working", "models")
shutil.make_archive("/kaggle/working/sentinelx_experiments", "zip",
                    "/kaggle/working", "experiments")
print("done — download from the notebook Output panel")
```

Download `sentinelx_trained.zip` (the checkpoint) + `sentinelx_experiments.zip`
(metrics/report) from the right-hand **Output** panel.

---

## 3. Bring it back here (Phase B)

Unzip so you get:
```
models/sentinel_x/{model.pt, config.yaml, metadata.json}
experiments/world_model_{results.csv,report.md,metrics.json}
```
Place them at the repo root (they overwrite the local scenario-11 checkpoint —
back it up first if you want to keep it). Then locally:

```
python scripts/evaluate.py --dataset ctu-13-full --checkpoint models/sentinel_x/model.pt
python scripts/run_experiment.py --dataset ctu-13-full --checkpoint models/sentinel_x/model.pt
```

Compare `experiments/world_model_metrics.json` PR-AUC against the baseline bar
(CTU-13 s11: LogReg 0.917 / GRU 0.955 / GraphSAGE 0.736). Note the target is now
harder (cross-family, scenario-disjoint), so a slightly lower but STABLE PR-AUC
across unseen families is the real win.

---

## Kaggle limitations / shortcomings to know up front

| Limit | Value | Impact on us |
|---|---|---|
| Session runtime | 9h (GPU), 12h (CPU) | 48k windows x 50 epochs x 4 combos may be tight. Mitigation: run 1 combo at a time, or drop to `--epochs 30`. |
| GPU quota | ~30h/week | Fine for a few runs. Don't leave idle sessions open. |
| Output storage | 20 GB /kaggle/working | We only write ~1 MB checkpoints + metrics. No problem. |
| Dataset size | 100+ GB allowed (private) | Our 158 MB is trivial. PCAPs would NOT fit comfortably and aren't needed anyway. |
| No internet mid-run by default | must toggle ON | Needed only for the pip installs in Cell 2. |
| Idle timeout | ~20–40 min interactive | Use "Save Version -> Run All" for long unattended training. |
| CPU RAM | ~13–16 GB | Loading all windows into memory: train.jsonl is 504 MB raw -> fits, but watch peak. If OOM, we can add streaming. |

Biggest real risk: **the 9h GPU cap vs 4 combos x 50 epochs.** Recommendation for
the first run — do a single combo to validate the whole loop end-to-end:

```
--combos gat_gru --epochs 30
```

Once that round-trips cleanly (train -> download -> evaluate here), scale up to
all four combos / 50 epochs.
