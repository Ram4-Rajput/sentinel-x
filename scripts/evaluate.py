#!/usr/bin/env python
"""Phase-9 unified evaluation entry point (reproducible).

Evaluates a trained Sentinel-X checkpoint on the standard chronological test
split and, optionally, refreshes the machine-readable model comparison. Uses the
checkpoint's own validation-tuned threshold (thresholds are NEVER re-tuned on
the final test set). Records a reproducibility manifest.

    python scripts/evaluate.py
    python scripts/evaluate.py --dataset ctu-13 --ks 1,3,5,10
    python scripts/evaluate.py --checkpoint models/sentinel_x/model.pt --comparison

Reuses the trained checkpoint and the Phase-2 cache; does not retrain.

Outputs (depending on flags):
    experiments/k_step_results.csv|report|metrics   (--ks)
    experiments/model_comparison.csv                (--comparison)
    experiments/runs/evaluate_<timestamp>.json      (reproducibility manifest)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline.config import PROCESSED_ROOT, REPO_ROOT  # noqa: E402
from sentinelx.research import comparison, experiments  # noqa: E402
from sentinelx.research.reproducibility import (  # noqa: E402
    finalize_manifest, new_manifest, seed_everything,
)


def _parse_ks(s):
    if not s:
        return (1, 3, 5, 10)
    return tuple(int(c.strip()) for c in s.split(",") if c.strip()) or (1, 3, 5, 10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sentinel-X unified evaluation (Phase 9).")
    ap.add_argument("--checkpoint",
                    default=str(REPO_ROOT / "models" / "sentinel_x" / "model.pt"))
    ap.add_argument("--dataset", default="ctu-13")
    ap.add_argument("--ks", default=None,
                    help="If set (e.g. 1,3,5,10) run K-step forecasting evaluation.")
    ap.add_argument("--comparison", action="store_true",
                    help="Rebuild experiments/model_comparison.csv from measured results.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--experiments-dir", default=str(REPO_ROOT / "experiments"))
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = Path(args.experiments_dir) / "runs"
    exp_dir = Path(args.experiments_dir)
    ckpt = Path(args.checkpoint)
    manifest = new_manifest("evaluate", f"evaluate:{args.dataset}", args.seed,
                            config={"checkpoint": str(ckpt), "dataset": args.dataset,
                                    "ks": args.ks, "comparison": args.comparison},
                            repo_root=REPO_ROOT)
    outputs = []

    # 1. Known-attack evaluation on the standard split (always).
    known = experiments.run_known_attacks(
        dataset=args.dataset, processed_root=Path(args.processed_root),
        checkpoint_path=ckpt, experiments_dir=exp_dir, seed=args.seed)
    (runs_dir / f"evaluate_{ts}.known_attacks.json").parent.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"evaluate_{ts}.known_attacks.json").write_text(
        json.dumps(known, indent=2, default=str), encoding="utf-8")
    outputs.append(runs_dir / f"evaluate_{ts}.known_attacks.json")
    if known.get("skipped"):
        print(f"[evaluate] known_attacks SKIPPED: {known.get('reason')}")
    else:
        t = known.get("test", {})
        print(f"[evaluate] known_attacks {args.dataset}: pr_auc={t.get('pr_auc')} "
              f"recall={t.get('recall')} f1={t.get('f1')}")

    # 2. Optional K-step forecasting.
    if args.ks:
        experiments.run_kstep(
            dataset=args.dataset, processed_root=Path(args.processed_root),
            checkpoint_path=ckpt, experiments_dir=exp_dir, ks=_parse_ks(args.ks),
            seed=args.seed)
        outputs.append(exp_dir / "k_step_results.csv")
        print(f"[evaluate] k-step -> {exp_dir / 'k_step_results.csv'}")

    # 3. Optional comparison rebuild.
    if args.comparison:
        path = comparison.build_and_write(exp_dir, repo_root=REPO_ROOT)
        outputs.append(path)
        print(f"[evaluate] comparison -> {path}")

    finalize_manifest(manifest, status="completed", outputs=outputs)
    manifest.write(runs_dir / f"evaluate_{ts}.json")
    print(f"[evaluate] manifest -> {runs_dir / f'evaluate_{ts}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
