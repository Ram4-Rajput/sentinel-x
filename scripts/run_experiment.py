#!/usr/bin/env python
"""Phase-9 experiment orchestrator (reproducible).

Runs one, several, or ALL named research experiments against the trained
Sentinel-X checkpoint + Phase-2 cache, then (by default) assembles the
machine-readable comparison ``experiments/model_comparison.csv`` from the
genuinely measured results.

    python scripts/run_experiment.py --list
    python scripts/run_experiment.py --experiment all
    python scripts/run_experiment.py --experiment early_warning --experiment ood_detection
    python scripts/run_experiment.py --experiment known_attacks --no-comparison

Named experiments (each reuses existing code / checkpoint / cache — nothing is
retrained or rebuilt):
    known_attacks         unseen_attacks        kstep
    early_warning         missing_telemetry     uncertainty_error
    ood_detection         forecast_stability    cross_dataset
    baseline_comparison

Discipline: chronological splits; thresholds never tuned on the final test set;
results never fabricated (unmeasurable cases are skipped, not invented).

Outputs:
    experiments/phase9/<experiment>_result.json   (per experiment)
    experiments/model_comparison.csv              (unless --no-comparison)
    experiments/runs/run_experiment_<timestamp>.json  (reproducibility manifest)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sentinel-X Phase-9 experiment orchestrator.")
    ap.add_argument("--experiment", action="append", default=None,
                    help="Experiment name (repeatable), or 'all'. See --list.")
    ap.add_argument("--list", action="store_true", help="List available experiments and exit.")
    ap.add_argument("--dataset", default="ctu-13")
    ap.add_argument("--checkpoint",
                    default=str(REPO_ROOT / "models" / "sentinel_x" / "model.pt"))
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--k", type=int, default=10,
                    help="Forecast depth for early_warning / forecast_stability.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-comparison", action="store_true",
                    help="Skip rebuilding model_comparison.csv.")
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--experiments-dir", default=str(REPO_ROOT / "experiments"))
    args = ap.parse_args(argv)

    if args.list:
        print("Available experiments:")
        for name in experiments.ALL_EXPERIMENTS:
            print(f"  - {name}")
        return 0

    requested = args.experiment or ["all"]
    if "all" in requested:
        names = list(experiments.ALL_EXPERIMENTS)
    else:
        names = requested
        for n in names:
            if n not in experiments.EXPERIMENT_REGISTRY:
                ap.error(f"unknown experiment '{n}'. Valid: "
                         f"{', '.join(experiments.ALL_EXPERIMENTS)}")

    seed_everything(args.seed)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_dir = Path(args.experiments_dir)
    phase9_dir = exp_dir / "phase9"
    phase9_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = exp_dir / "runs"
    ckpt = Path(args.checkpoint)

    manifest = new_manifest("run_experiment", ",".join(names), args.seed,
                            config={"experiments": names, "dataset": args.dataset,
                                    "checkpoint": str(ckpt)}, repo_root=REPO_ROOT)

    # kwargs shared by every experiment
    base_kwargs = dict(
        dataset=args.dataset, processed_root=Path(args.processed_root),
        checkpoint_path=ckpt, experiments_dir=exp_dir, seed=args.seed,
    )
    ks = tuple(int(c) for c in args.ks.split(",") if c.strip())

    outputs = []
    early_warning_result = None
    results = {}
    for name in names:
        print(f"\n=== experiment: {name} ===")
        kwargs = dict(base_kwargs)
        if name == "kstep":
            kwargs["ks"] = ks
        if name in ("early_warning",):
            kwargs["K"] = args.k
        if name in ("forecast_stability",):
            kwargs["K"] = min(args.k, 5)
        try:
            res = experiments.run_experiment(name, **kwargs)
        except Exception as e:
            res = {"experiment": name, "error": repr(e)}
            print(f"[run_experiment] {name} ERROR: {e!r}")
        results[name] = res
        if name == "early_warning":
            early_warning_result = res
        out_path = phase9_dir / f"{name}_result.json"
        out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        outputs.append(out_path)

    # Assemble the machine-readable comparison (measured results only).
    if not args.no_comparison:
        cmp_path = comparison.build_and_write(
            exp_dir, repo_root=REPO_ROOT,
            early_warning_result=early_warning_result)
        outputs.append(cmp_path)
        print(f"\n[run_experiment] comparison -> {cmp_path}")

    finalize_manifest(manifest, status="completed", outputs=outputs)
    manifest.write(runs_dir / f"run_experiment_{ts}.json")
    print(f"[run_experiment] ran: {', '.join(names)}")
    print(f"[run_experiment] manifest -> {runs_dir / f'run_experiment_{ts}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
