#!/usr/bin/env python
"""Phase-5 CLI: K-step future-state forecasting via autoregressive rollout.

    python scripts/run_kstep_forecast.py
    python scripts/run_kstep_forecast.py --ks 1,3,5,10
    python scripts/run_kstep_forecast.py --checkpoint models/sentinel_x/model.pt --ks 5

Loads the trained Phase-4 world model checkpoint and rolls its learned one-step
latent transition forward K steps (NOT K independent classifiers). Writes:
    experiments/{k_step_results.csv, k_step_report.md, k_step_metrics.json}
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline.config import PROCESSED_ROOT, REPO_ROOT  # noqa: E402
from sentinelx.worldmodel.kstep_experiment import run_kstep_experiments  # noqa: E402


def _parse_ks(s):
    if not s:
        return (1, 3, 5, 10)
    out = []
    for c in s.split(","):
        c = c.strip()
        if c:
            out.append(int(c))
    return tuple(out) or (1, 3, 5, 10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sentinel-X Phase-5 K-step forecasting.")
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "models" / "sentinel_x" / "model.pt"),
                    help="Trained Phase-4 world-model checkpoint.")
    ap.add_argument("--ks", default="1,3,5,10", help="Comma list of horizons, e.g. 1,3,5,10.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--experiments-dir", default=str(REPO_ROOT / "experiments"))
    args = ap.parse_args(argv)

    ks = _parse_ks(args.ks)
    res = run_kstep_experiments(
        Path(args.checkpoint), Path(args.processed_root),
        ks=ks, batch_size=args.batch_size, seed=args.seed,
        experiments_dir=Path(args.experiments_dir),
    )
    if res.get("skipped"):
        print(f"[run_kstep_forecast] SKIPPED: {res.get('reason')}")
    else:
        print(f"[run_kstep_forecast] evaluated K in {list(res.get('ks'))}")
        print(f"   experiments -> {args.experiments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
