#!/usr/bin/env python
"""Phase-6 CLI: Risk + Uncertainty + Calibration + Novelty/OOD.

    python scripts/run_phase6.py
    python scripts/run_phase6.py --mc-passes 50 --ood-percentile 97.5
    python scripts/run_phase6.py --checkpoint models/sentinel_x/model.pt --no-temperature

Loads the trained Phase-4/5 world-model checkpoint and produces four explicitly
separate signals (never collapsed into one confidence number). Writes:
    experiments/{uncertainty_results.csv, calibration_results.csv,
                 ood_results.csv, uncertainty_ood_report.md}
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline.config import PROCESSED_ROOT, REPO_ROOT  # noqa: E402
from sentinelx.worldmodel.phase6_experiment import run_phase6_experiments  # noqa: E402
from sentinelx.worldmodel.uncertainty import DEFAULT_MC_PASSES  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Sentinel-X Phase-6 risk/uncertainty/calibration/OOD.")
    ap.add_argument("--checkpoint",
                    default=str(REPO_ROOT / "models" / "sentinel_x" / "model.pt"),
                    help="Trained world-model checkpoint.")
    ap.add_argument("--mc-passes", type=int, default=DEFAULT_MC_PASSES,
                    help=f"MC-Dropout stochastic passes (default {DEFAULT_MC_PASSES}).")
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--ood-percentile", type=float, default=95.0,
                    help="In-distribution percentile for the OOD threshold.")
    ap.add_argument("--no-temperature", action="store_true",
                    help="Disable the optional temperature-scaling stage.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--experiments-dir", default=str(REPO_ROOT / "experiments"))
    args = ap.parse_args(argv)

    res = run_phase6_experiments(
        Path(args.checkpoint), Path(args.processed_root),
        mc_passes=args.mc_passes, n_calibration_bins=args.calibration_bins,
        ood_percentile=args.ood_percentile,
        use_temperature_scaling=not args.no_temperature,
        batch_size=args.batch_size, seed=args.seed,
        experiments_dir=Path(args.experiments_dir),
    )
    if res.get("skipped"):
        print(f"[run_phase6] SKIPPED: {res.get('reason')}")
    else:
        print(f"[run_phase6] done -> {args.experiments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
