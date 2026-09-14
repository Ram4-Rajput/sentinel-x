#!/usr/bin/env python
"""Phase-4 CLI: train the Sentinel-X world model and write outputs.

    python scripts/train_world_model.py --dataset ctu-13
    python scripts/train_world_model.py --config configs/world_model.yaml
    python scripts/train_world_model.py --dataset ctu-13 --epochs 50 --combos gat_gru,gat_lstm

Reads Phase-2 caches from data/processed/<dataset>/ (build them first).
Writes:
    models/sentinel_x/{model.pt,config.yaml,metadata.json}
    experiments/{world_model_results.csv,world_model_report.md,world_model_metrics.json}
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline.config import PROCESSED_ROOT, REPO_ROOT  # noqa: E402
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
    ap = argparse.ArgumentParser(description="Train the Sentinel-X world model (Phase 4).")
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

    # CLI overrides (do not hardcode; only override when provided)
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

    combos = _parse_combos(args.combos)

    res = run_world_model_experiments(
        cfg, Path(args.processed_root),
        combos=combos,
        models_dir=Path(args.models_dir),
        experiments_dir=Path(args.experiments_dir),
    )
    if res.get("skipped"):
        print(f"[train_world_model] SKIPPED: {res.get('reason')}")
    else:
        print(f"[train_world_model] best combo: {res.get('best_combo')}")
        print(f"   models -> {args.models_dir}")
        print(f"   experiments -> {args.experiments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
