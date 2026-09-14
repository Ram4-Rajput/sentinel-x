#!/usr/bin/env python
"""Phase-7 CLI: attack trajectory + high-level MITRE ATT&CK interpretation.

    python scripts/run_trajectory.py
    python scripts/run_trajectory.py --k 10 --max-trajectories 5
    python scripts/run_trajectory.py --checkpoint models/sentinel_x/model.pt --k 5

Reuses the trained Phase-4 checkpoint and the Phase-5 autoregressive rollout to
construct interpretable OBSERVED→FORECAST trajectories. No retraining. Writes:
    experiments/{trajectory_results.csv, trajectory_report.md,
                 trajectory_metrics.json}
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline.config import PROCESSED_ROOT, REPO_ROOT  # noqa: E402
from sentinelx.worldmodel.trajectory_experiment import run_trajectory_experiment  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Sentinel-X Phase-7 attack trajectory + MITRE mapping.")
    ap.add_argument("--checkpoint",
                    default=str(REPO_ROOT / "models" / "sentinel_x" / "model.pt"),
                    help="Trained Phase-4 world-model checkpoint.")
    ap.add_argument("--k", type=int, default=5,
                    help="Forecast depth (future stages to project).")
    ap.add_argument("--max-trajectories", type=int, default=None,
                    help="Cap the number of trajectories built (default: all test).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--experiments-dir", default=str(REPO_ROOT / "experiments"))
    args = ap.parse_args(argv)

    res = run_trajectory_experiment(
        Path(args.checkpoint), Path(args.processed_root),
        K=args.k, max_trajectories=args.max_trajectories,
        batch_size=args.batch_size, seed=args.seed,
        experiments_dir=Path(args.experiments_dir),
    )
    if res.get("skipped"):
        print(f"[run_trajectory] SKIPPED: {res.get('reason')}")
    else:
        summ = res.get("summary", {})
        print(f"[run_trajectory] built {summ.get('n_trajectories')} trajectories "
              f"(observed={summ.get('total_observed')}, "
              f"forecast={summ.get('total_forecast')})")
        print(f"   experiments -> {args.experiments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
