#!/usr/bin/env python
"""Phase-8 CLI: explainability + propagation + counterfactual + stability.

    python scripts/run_phase8.py
    python scripts/run_phase8.py --k 5 --max-anchors 25
    python scripts/run_phase8.py --checkpoint models/sentinel_x/model.pt --k 5 \
        --stability-trials 16 --stability-epsilon 0.05

Reuses the trained Phase-4 checkpoint and the Phase-5 autoregressive rollout to
layer four analytical capabilities around the world model. No retraining.
Writes:
    experiments/{phase8_results.csv, phase8_report.md, phase8_metrics.json}
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline.config import PROCESSED_ROOT, REPO_ROOT  # noqa: E402
from sentinelx.worldmodel.phase8_experiment import run_phase8_experiment  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Sentinel-X Phase-8: explainability, propagation, "
                    "counterfactual simulation, forecast stability.")
    ap.add_argument("--checkpoint",
                    default=str(REPO_ROOT / "models" / "sentinel_x" / "model.pt"),
                    help="Trained Phase-4 world-model checkpoint.")
    ap.add_argument("--k", type=int, default=5, help="Forecast depth (K steps).")
    ap.add_argument("--max-anchors", type=int, default=25,
                    help="Cap the number of test anchors analysed (default: 25).")
    ap.add_argument("--stability-trials", type=int, default=16)
    ap.add_argument("--stability-epsilon", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--experiments-dir", default=str(REPO_ROOT / "experiments"))
    args = ap.parse_args(argv)

    res = run_phase8_experiment(
        Path(args.checkpoint), Path(args.processed_root),
        K=args.k, max_anchors=args.max_anchors,
        stability_trials=args.stability_trials,
        stability_epsilon=args.stability_epsilon,
        seed=args.seed, experiments_dir=Path(args.experiments_dir),
    )
    if res.get("skipped"):
        print(f"[run_phase8] SKIPPED: {res.get('reason')}")
    else:
        summ = res.get("summary", {})
        print(f"[run_phase8] analysed {summ.get('n_anchors')} anchors | "
              f"mean stability={round(summ.get('mean_stability_score', 0.0), 4)} | "
              f"spreading={summ.get('n_spreading')} | "
              f"counterfactuals={summ.get('n_counterfactuals')}")
        print(f"   experiments -> {args.experiments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
