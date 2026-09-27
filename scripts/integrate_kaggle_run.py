#!/usr/bin/env python
"""Integrate a Kaggle Phase-A training bundle (train.zip) into the repo.

train.zip layout (per combo, from the Kaggle notebook):
  train/sentinelx_<combo>.zip
    -> kaggle/working/models/sentinel_x_<combo>/{model.pt,config.yaml,metadata.json}
    -> kaggle/working/experiments_<combo>/world_model_{metrics.json,report.md,results.csv}
       + runs/

This script (no GPU, purely local file work):
  1. Backs up the existing local models/sentinel_x/ + experiments/world_model_* .
  2. Extracts all four combos to models/kaggle_ctu13full/<combo>/.
  3. Merges the four single-combo metrics into ONE experiments/world_model_metrics.json
     (dataset ctu-13-full, all 4 runs) + a world_model_results.csv comparison table,
     picking the best combo by test PR-AUC.
  4. Installs the best combo's checkpoint as the canonical models/sentinel_x/.

Run:  python scripts/integrate_kaggle_run.py --zip train.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
EXP = REPO / "experiments"
COMBOS = ["gat_gru", "gat_lstm", "graphsage_gru", "graphsage_lstm"]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _backup(path: Path, tag: str) -> None:
    if path.exists():
        dest = path.with_name(path.name + f".bak_{tag}")
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.copytree(path, dest) if path.is_dir() else shutil.copy2(path, dest)
        print(f"[backup] {path.name} -> {dest.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="train.zip")
    args = ap.parse_args(argv)

    src_zip = (REPO / args.zip) if not Path(args.zip).is_absolute() else Path(args.zip)
    if not src_zip.exists():
        raise SystemExit(f"[err] not found: {src_zip}")

    tag = _ts()
    work = REPO / "_kaggle_integrate"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    # 1. unpack outer + inner zips
    with zipfile.ZipFile(src_zip) as z:
        z.extractall(work)
    inner_dir = work / "train"
    combo_roots = {}
    for combo in COMBOS:
        inner = inner_dir / f"sentinelx_{combo}.zip"
        if not inner.exists():
            print(f"[warn] missing combo zip: {combo}")
            continue
        out = work / combo
        with zipfile.ZipFile(inner) as z:
            z.extractall(out)
        combo_roots[combo] = out / "kaggle" / "working"

    # 2. backup existing artifacts
    _backup(MODELS / "sentinel_x", tag)
    for f in ("world_model_metrics.json", "world_model_report.md", "world_model_results.csv"):
        _backup(EXP / f, tag)

    # 3. collect per-combo run records + copy checkpoints
    dest_root = MODELS / "kaggle_ctu13full"
    dest_root.mkdir(parents=True, exist_ok=True)
    runs = []
    split_info = training_config = data_config = None
    for combo, wroot in combo_roots.items():
        mdir = wroot / f"models/sentinel_x_{combo}"
        edir = wroot / f"experiments_{combo}"
        # copy checkpoint set
        dest = dest_root / combo
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(mdir, dest)
        # pull the single run record from that combo's metrics
        m = json.loads((edir / "world_model_metrics.json").read_text(encoding="utf-8"))
        if m.get("runs"):
            runs.append(m["runs"][0])
        split_info = split_info or m.get("split_info")
        training_config = training_config or m.get("training_config")
        data_config = data_config or m.get("data_config")
        print(f"[ok] {combo}: checkpoint -> models/kaggle_ctu13full/{combo}/")

    if not runs:
        raise SystemExit("[err] no run records found in bundle")

    # best by test PR-AUC (threshold-free headline metric)
    def _prauc(r):
        v = (r.get("test") or {}).get("pr_auc")
        return v if isinstance(v, (int, float)) else -1.0

    runs_sorted = sorted(runs, key=_prauc, reverse=True)
    best = runs_sorted[0]
    best_combo = best["combo"]

    # 4. merged metrics json
    merged = {
        "dataset": "ctu-13-full",
        "phase": "A-kaggle-full-train",
        "integrated_utc": datetime.now(timezone.utc).isoformat(),
        "split_info": split_info,
        "data_config": data_config,
        "training_config": training_config,
        "runs": runs_sorted,
        "best_combo": best_combo,
        "best_selected_by": "test.pr_auc",
    }
    (EXP / "world_model_metrics.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8")

    # comparison CSV
    lines = ["combo,gnn_type,temporal_type,num_parameters,best_epoch,train_seconds,"
             "chosen_threshold,test_pr_auc,test_precision,test_recall,test_f1,"
             "test_roc_auc,val_pr_auc"]
    for r in runs_sorted:
        t = r.get("test", {})
        v = r.get("val", {})
        lines.append(",".join(str(x) for x in [
            r["combo"], r["gnn_type"], r["temporal_type"], r["num_parameters"],
            r["best_epoch"], r["train_seconds"], r.get("chosen_threshold"),
            t.get("pr_auc"), t.get("precision"), t.get("recall"), t.get("f1"),
            t.get("roc_auc"), v.get("pr_auc"),
        ]))
    (EXP / "world_model_results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 5. install best combo as canonical models/sentinel_x/
    canonical = MODELS / "sentinel_x"
    canonical.mkdir(parents=True, exist_ok=True)
    for fname in ("model.pt", "config.yaml", "metadata.json"):
        srcf = dest_root / best_combo / fname
        if srcf.exists():
            shutil.copy2(srcf, canonical / fname)
    print(f"[ok] canonical model.pt <- {best_combo} "
          f"(test PR-AUC {_prauc(best):.4f})")

    shutil.rmtree(work)
    print("\n[done] integration complete.")
    print(f"  best combo : {best_combo}")
    print(f"  metrics    : experiments/world_model_metrics.json")
    print(f"  comparison : experiments/world_model_results.csv")
    print(f"  all combos : models/kaggle_ctu13full/")
    print(f"  canonical  : models/sentinel_x/ (backup: .bak_{tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
