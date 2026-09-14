#!/usr/bin/env python
"""Phase-3 CLI: run all three baselines and write experiments/ outputs.

    python scripts/run_baselines.py --dataset ctu-13
    python scripts/run_baselines.py --dataset ctu-13 --temporal-type lstm --gnn-type gat
    python scripts/run_baselines.py --dataset ctu-13 --dataset unsw-nb15

Reads Phase-2 caches from data/processed/<dataset>/ (build them first).
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.experiments.common import ExperimentConfig  # noqa: E402
from sentinelx.experiments.runner import run_all_baselines, write_outputs  # noqa: E402
from sentinelx.pipeline.config import PROCESSED_ROOT, REPO_ROOT  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run Sentinel-X Phase-3 baselines.")
    ap.add_argument("--dataset", action="append", required=True,
                    help="Dataset key (repeatable). e.g. --dataset ctu-13")
    ap.add_argument("--seq-len", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--temporal-type", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--gnn-type", choices=["sage", "gat"], default="sage")
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--mlflow-uri", default=None,
                    help="MLflow tracking URI (default: file store under experiments/mlruns).")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "experiments"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # MLflow 3.x deprecated the file store -> default to a local SQLite backend.
    default_uri = "sqlite:///" + str((out_dir / "mlflow.db").resolve()).replace("\\", "/")
    mlflow_uri = args.mlflow_uri or default_uri

    all_results = []
    for ds in args.dataset:
        cfg = ExperimentConfig(
            dataset=ds.strip().lower(),
            processed_root=Path(args.processed_root),
            seq_len=args.seq_len, horizon=args.horizon,
            train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed,
        )
        artifacts_uri = "file:///" + str((out_dir / "mlartifacts").resolve()).replace("\\", "/")
        res = run_all_baselines(
            cfg, temporal_type=args.temporal_type, gnn_type=args.gnn_type,
            epochs=args.epochs, mlflow_uri=mlflow_uri, mlflow_artifacts=artifacts_uri,
        )
        all_results.append(res)

    paths = write_outputs(all_results, out_dir)
    print("\n[baselines] wrote:")
    for k, v in paths.items():
        print(f"   {k}: {v}")
    print(f"   mlflow: {mlflow_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
