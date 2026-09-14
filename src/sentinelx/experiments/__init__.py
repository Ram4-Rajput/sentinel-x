"""Sentinel-X Phase-3 scientific baselines.

Purpose: answer "does modelling temporal structure and network topology provide
measurable benefit over simpler approaches?" — BEFORE building the Sentinel-X
world model.

Three baselines, all sharing an identical experiment protocol (same window
sequence, same chronological split, same forecasting target, same seed, same
leakage policy):

    baseline 1  logistic_regression   (aggregated window features -> future attack)
    baseline 2  sequence  (LSTM|GRU)  (feature sequence [t-n..t] -> future attack)
    baseline 3  temporal_gnn (GraphSAGE|GAT) (graph sequence [t-n..t] -> future attack)

The Sentinel-X world model is intentionally NOT implemented in this phase.
"""

from .common import (
    ExperimentConfig,
    WindowSample,
    build_window_samples,
    seed_everything,
)
from .metrics import BinaryMetrics, compute_binary_metrics

__all__ = [
    "ExperimentConfig",
    "WindowSample",
    "build_window_samples",
    "seed_everything",
    "BinaryMetrics",
    "compute_binary_metrics",
]
