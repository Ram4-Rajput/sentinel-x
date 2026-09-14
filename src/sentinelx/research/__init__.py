"""Phase-9: Research experiment infrastructure (consolidation layer).

This package does NOT introduce a new model or a new data protocol. It
consolidates the ML experimentation system that Phases 3-8 already built into a
single, reproducible surface:

  * reproducibility.py — deterministic seeding, dependency/version recording,
    configuration snapshots, and a per-run manifest so every result can be
    reproduced and audited.
  * experiments.py     — a registry of named research experiments (known
    attacks, unseen attack types, K=1/3/5/10, early-warning lead time, missing
    telemetry, uncertainty-vs-error, OOD detection, forecast stability,
    cross-dataset generalization, baseline comparison). Each experiment reuses
    the EXISTING phase drivers / trained checkpoint / cached data — nothing is
    retrained or rebuilt unless explicitly requested.
  * comparison.py      — assembles ``experiments/model_comparison.csv`` from the
    genuinely measured results only. Fields that were never measured are left
    blank rather than fabricated.

Reproducible commands (thin CLIs over this package):
    python scripts/train.py ...
    python scripts/evaluate.py ...
    python scripts/run_experiment.py ...
"""

from __future__ import annotations

__all__ = [
    "reproducibility",
    "experiments",
    "comparison",
]
