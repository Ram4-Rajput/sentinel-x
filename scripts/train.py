#!/usr/bin/env python
"""Phase-9 unified training entry point (reproducible).

A single, documented command to (re)train the Sentinel-X world model with full
reproducibility: fixed seed, dependency/version recording, a configuration
snapshot, and a machine-readable run manifest written alongside the outputs.
Model checkpoints + training logs are produced by the existing Phase-4 trainer;
this script wraps it so every run is auditable and reproducible.

    python scripts/train.py --dataset ctu-13
    python scripts/train.py --config configs/world_model.yaml --epochs 50
    python scripts/train.py --dataset ctu-13 --combos graphsage_lstm --seed 42

Reuses the Phase-2 cache in data/processed/<dataset>/ (build it first with
scripts/build_dataset.py). Does NOT rebuild data or retrain unless invoked.

Outputs:
    models/sentinel_x/{model.pt,config.yaml,metadata.json}
    experiments/{world_model_results.csv,world_model_report.md,world_model_metrics.json}
    experiments/runs/train_<timestamp>.json            (reproducibility manifest)
    experiments/runs/train_<timestamp>.config.json     (config snapshot)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline.config import PROCESSED_ROOT, REPO_ROOT  # noqa: E402
from sentinelx.research.reproducibility import (  # noqa: E402
    finalize_manifest, new_manifest, seed_everything, snapshot_config,
)
from sentinelx.worldmodel.config import (  # noqa: E402
    DataConfig, ModelConfig, TrainingConfig, WorldModelConfig, load_config,
)
from sentinelx.worldmodel.experiment import run_world_model_experiments  # noqa: E402


def _parse_combos(s):
    if not s:
        return None
    out = []
    for c in s.split(","):
        c = c.strip().lower()
        if not c:
            continue
        gnn, _, temporal = c.partition("_")
        out.append((gnn, temporal))
    return out or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sentinel-X unified training (Phase 9).")
    ap.add_argument("--config", default=None, help="YAML config path (overrides defaults).")
    ap.add_argument("--dataset", default=None, help="Dataset key, e.g. ctu-13.")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--latent-dim", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--combos", default=None,
                    help="Comma list like gat_gru,gat_lstm,graphsage_gru,graphsage_lstm.")
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--models-dir", default=str(REPO_ROOT / "models" / "sentinel_x"))
    ap.add_argument("--experiments-dir", default=str(REPO_ROOT / "experiments"))
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config)) if args.config else WorldModelConfig(
        model=ModelConfig().normalized(), training=TrainingConfig(), data=DataConfig())

    if args.dataset: cfg.data.dataset = args.dataset.strip().lower()
    if args.seq_len is not None: cfg.data.seq_len = args.seq_len
    if args.horizon is not None: cfg.data.horizon = args.horizon
    if args.epochs is not None: cfg.training.epochs = args.epochs
    if args.batch_size is not None: cfg.training.batch_size = args.batch_size
    if args.learning_rate is not None: cfg.training.learning_rate = args.learning_rate
    if args.hidden_dim is not None: cfg.model.hidden_dim = args.hidden_dim
    if args.latent_dim is not None: cfg.model.latent_dim = args.latent_dim
    if args.seed is not None: cfg.training.seed = args.seed
    cfg.model = cfg.model.normalized()

    # --- reproducibility: seed + manifest + config snapshot ---
    seed_everything(cfg.training.seed)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = Path(args.experiments_dir) / "runs"
    manifest = new_manifest("train", f"world_model:{cfg.data.dataset}",
                            cfg.training.seed, config=cfg.to_dict(),
                            repo_root=REPO_ROOT)
    snapshot_config(cfg.to_dict(), runs_dir / f"train_{ts}.config.json")

    combos = _parse_combos(args.combos)
    try:
        res = run_world_model_experiments(
            cfg, Path(args.processed_root), combos=combos,
            models_dir=Path(args.models_dir),
            experiments_dir=Path(args.experiments_dir),
        )
        status = "skipped" if res.get("skipped") else "completed"
        outputs = [
            Path(args.models_dir) / "model.pt",
            Path(args.experiments_dir) / "world_model_results.csv",
        ]
        finalize_manifest(manifest, status=status, outputs=outputs,
                          notes=res.get("reason"))
    except Exception as e:
        finalize_manifest(manifest, status="error", notes=repr(e))
        manifest.write(runs_dir / f"train_{ts}.json")
        raise

    manifest.write(runs_dir / f"train_{ts}.json")

    if res.get("skipped"):
        print(f"[train] SKIPPED: {res.get('reason')}")
    else:
        print(f"[train] best combo: {res.get('best_combo')}")
        print(f"   models -> {args.models_dir}")
        print(f"   experiments -> {args.experiments_dir}")
        print(f"   manifest -> {runs_dir / f'train_{ts}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
