"""Sentinel-X world-model core (Phase 4).

Learns P(S_{t+1} | S_t) over a *dynamic network state* rather than classifying
the current traffic window. The pipeline is:

    dynamic graph G_t
        -> GAT / GraphSAGE          (node embeddings, consumes topology + edges)
        -> graph pooling            (graph-level state h_t)
        -> GRU / LSTM               (temporal latent z_t over h_{t-n..t})
        -> forecasting head         (predict next latent state z_{t+1})
        -> (auxiliary) risk head    (predict malicious risk at t+H)

The learned latent state ``z_t`` is exposed on the model API for downstream
reuse (forecasting, novelty/OOD, explainability, trajectory analysis) in later
phases. This phase does NOT implement uncertainty/OOD/MITRE/counterfactuals or
a frontend.
"""

from .config import ModelConfig, TrainingConfig, WorldModelConfig, load_config
from .model import SentinelXWorldModel
from .rollout import RolloutResult, StepPrediction, rollout_latents, validate_k
from .kstep_eval import HorizonMetrics, KStepEvaluation, evaluate_kstep
from .uncertainty import (
    DEFAULT_MC_PASSES,
    UncertaintyEstimate,
    estimate_uncertainty,
    mc_dropout_risk,
)
from .calibration import (
    CalibrationReport,
    TemperatureScaler,
    evaluate_calibration,
)
from .ood import MahalanobisOOD, OODScore, encode_latents, ood_detection_metrics

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "WorldModelConfig",
    "load_config",
    "SentinelXWorldModel",
    # Phase 5: K-step forecasting
    "rollout_latents",
    "validate_k",
    "RolloutResult",
    "StepPrediction",
    "evaluate_kstep",
    "KStepEvaluation",
    "HorizonMetrics",
    # Phase 6: risk / uncertainty / calibration / novelty-OOD
    "DEFAULT_MC_PASSES",
    "UncertaintyEstimate",
    "estimate_uncertainty",
    "mc_dropout_risk",
    "CalibrationReport",
    "TemperatureScaler",
    "evaluate_calibration",
    "MahalanobisOOD",
    "OODScore",
    "encode_latents",
    "ood_detection_metrics",
]
