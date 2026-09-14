"""Phase-7: Attack trajectory construction + high-level MITRE ATT&CK mapping.

This phase turns the EXISTING temporal forecasts (Phase-4 world model +
Phase-5 autoregressive rollout) into an interpretable *attack trajectory* that
strictly separates what the system has **OBSERVED** from what it merely
**FORECASTS**.

    Observed:  Reconnaissance
    Observed:  Suspicious scanning
    Forecast:  Initial Access
    Forecast:  Lateral Movement
    Forecast:  Command & Control

A forecasted stage is NEVER presented as having already happened. The observed
segment is grounded in the real cached window states up to and including the
anchor window ``t``; the forecast segment is grounded in the rolled-out future
latent states / risk for ``t+1 .. t+K`` — predictions, not facts.

What Phase-7 does NOT do
------------------------
* It does not train anything or invent new model outputs. Every number comes
  from the Phase-4 checkpoint and the Phase-5 rollout on real cached windows.
* It does not blindly assign precise ATT&CK technique IDs. Mapping is to
  high-level ATT&CK *stages* (tactics). Where evidence is insufficient it backs
  off to a coarser stage rather than fabricating specificity.

Per-stage return contract (spec)
--------------------------------
    {
      "stage": "...",                 # high-level ATT&CK stage
      "status": "observed|forecast",  # NEVER claim a forecast occurred
      "confidence": 0.0,              # [0,1] — evidence strength / model risk
      "evidence": [ ... ],            # supporting signals (source state, etc.)
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..experiments.common import WINDOW_FEATURE_NAMES, _aggregate_window


# --------------------------------------------------------------------------- #
# High-level MITRE ATT&CK stages (tactics). We intentionally stay at the tactic
# level; the spec forbids blindly assigning precise techniques.
# --------------------------------------------------------------------------- #
STAGE_RECON = "Reconnaissance"
STAGE_INITIAL_ACCESS = "Initial Access"
STAGE_EXECUTION = "Execution"
STAGE_PERSISTENCE = "Persistence"
STAGE_PRIV_ESC = "Privilege Escalation"
STAGE_LATERAL = "Lateral Movement"
STAGE_C2 = "Command & Control"
STAGE_EXFIL = "Exfiltration"

# Coarse fall-back when evidence is present but not specific enough to name a
# precise stage. This satisfies "if evidence is insufficient, return a
# higher-level stage" without fabricating precision.
STAGE_SUSPICIOUS = "Suspicious Activity"
STAGE_BENIGN = "Benign / No Attack Behaviour"

# The canonical, ordered set of stages a caller may see (kill-chain order).
MITRE_STAGES = (
    STAGE_RECON, STAGE_INITIAL_ACCESS, STAGE_EXECUTION, STAGE_PERSISTENCE,
    STAGE_PRIV_ESC, STAGE_LATERAL, STAGE_C2, STAGE_EXFIL,
)

STATUS_OBSERVED = "observed"
STATUS_FORECAST = "forecast"


@dataclass
class TrajectoryStage:
    """One stage of the attack trajectory.

    Beyond the spec's ``{stage, status, confidence, evidence}`` we also carry
    ``horizon`` / ``timestamp`` / ``source`` so the trajectory is temporally
    ordered and every stage is traceable to the state it was derived from.
    """

    stage: str
    status: str                    # "observed" | "forecast"
    confidence: float              # [0, 1]
    evidence: List[str] = field(default_factory=list)
    # --- provenance (temporal ordering + traceability) ---
    horizon: int = 0               # 0 = anchor/observed present, +k = forecast t+k
    window_index: Optional[int] = None
    timestamp: Optional[str] = None
    source: str = ""               # "observed-state" | "forecast-rollout"

    def as_dict(self) -> Dict:
        # The spec's minimal contract, plus provenance fields.
        return {
            "stage": self.stage,
            "status": self.status,
            "confidence": round(float(self.confidence), 6),
            "evidence": list(self.evidence),
            "horizon": self.horizon,
            "window_index": self.window_index,
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass
class AttackTrajectory:
    """An ordered observed→forecast trajectory anchored at window ``t``."""

    t_index: int
    stages: List[TrajectoryStage] = field(default_factory=list)

    @property
    def observed(self) -> List[TrajectoryStage]:
        return [s for s in self.stages if s.status == STATUS_OBSERVED]

    @property
    def forecast(self) -> List[TrajectoryStage]:
        return [s for s in self.stages if s.status == STATUS_FORECAST]

    def as_dict(self) -> Dict:
        return {
            "t_index": self.t_index,
            "n_observed": len(self.observed),
            "n_forecast": len(self.forecast),
            "stages": [s.as_dict() for s in self.stages],
        }


# --------------------------------------------------------------------------- #
# Behaviour → high-level MITRE stage mapping (evidence-gated).
# --------------------------------------------------------------------------- #
def _window_signals(window: Dict) -> Dict[str, float]:
    """Extract interpretable behaviour signals from a cached window dict.

    Uses only quantities already present in the Phase-2 cache (reuses the same
    aggregation the baselines/world-model consume). No packet features are
    fabricated.
    """
    agg = _aggregate_window(window)
    feats = {name: float(v) for name, v in zip(WINDOW_FEATURE_NAMES, agg)}
    feats["label_any_attack"] = float(window.get("label_any_attack", 0))
    return feats


def map_behaviour_to_stage(
    signals: Dict[str, float],
    *,
    risk: Optional[float] = None,
) -> Dict:
    """Map behaviour signals to a HIGH-LEVEL MITRE stage with evidence.

    The mapping is deliberately conservative:

    * Strong, corroborated signals (e.g. a labelled-attack window with a high
      attack ratio and elevated flow volume) map to a specific high-level stage
      such as Command & Control or Lateral Movement.
    * A single weak signal (e.g. a modest rise in distinct hosts contacted) maps
      to a coarse stage — Reconnaissance or the generic ``Suspicious Activity``
      bucket — rather than a precise technique.
    * No behaviour at all maps to the benign stage.

    Returns ``{"stage": str, "confidence": float, "evidence": [str, ...]}``.
    Precise ATT&CK technique IDs are never returned.
    """
    evidence: List[str] = []
    attack_ratio = signals.get("attack_ratio", 0.0)
    edge_attack_fraction = signals.get("edge_attack_fraction", 0.0)
    num_flows = signals.get("num_flows", 0.0)
    num_nodes = signals.get("num_nodes", 0.0)
    num_edges = signals.get("num_edges", 0.0)
    density = signals.get("density", 0.0)
    mean_out = signals.get("mean_node_out_degree", 0.0)
    total_edge_bytes = signals.get("total_edge_bytes", 0.0)
    labelled = signals.get("label_any_attack", 0.0) >= 1.0

    # ---- No evidence of anything -> benign, low confidence in any attack stage.
    if not labelled and attack_ratio <= 0.0 and edge_attack_fraction <= 0.0 and (
        risk is None or risk < 0.5
    ):
        # A quiet, unlabelled window with no malicious edges and low/absent risk.
        conf = 0.0 if risk is None else max(0.0, 0.5 - risk) * 2.0  # -> [0,1]
        evidence.append(
            f"no malicious edges (attack_ratio={attack_ratio:.3f}, "
            f"edge_attack_fraction={edge_attack_fraction:.3f})"
        )
        if risk is not None:
            evidence.append(f"model risk below decision threshold (risk={risk:.3f})")
        return {"stage": STAGE_BENIGN, "confidence": round(conf, 4),
                "evidence": evidence}

    # Accumulate corroborating signals to decide specificity.
    strong_signals = 0

    if labelled:
        strong_signals += 1
        evidence.append("window carries a ground-truth attack label")
    if attack_ratio >= 0.3:
        strong_signals += 1
        evidence.append(f"high attack ratio ({attack_ratio:.3f})")
    elif attack_ratio > 0.0:
        evidence.append(f"some malicious flows (attack_ratio={attack_ratio:.3f})")
    if edge_attack_fraction >= 0.3:
        strong_signals += 1
        evidence.append(f"many malicious edges (fraction={edge_attack_fraction:.3f})")

    # Behaviour-shape signals help distinguish the high-level stage.
    high_fanout = mean_out >= 2.0 or (num_nodes > 1 and num_edges >= 2 * num_nodes)
    high_volume = total_edge_bytes >= 1e4 or num_flows >= 8.0
    many_hosts = num_nodes >= 5.0

    # --- Stage decision (high-level tactic only) ---
    # Insufficient corroboration -> back off to a coarse stage.
    if strong_signals == 0:
        # Only faint signals (e.g. slightly elevated risk / a few flows).
        stage = STAGE_RECON if many_hosts else STAGE_SUSPICIOUS
        if many_hosts:
            evidence.append(f"contact with several hosts (num_nodes={num_nodes:.0f})")
        base = 0.25
        conf = base + (0.15 if risk is not None and risk >= 0.5 else 0.0)
        return {"stage": stage, "confidence": round(min(conf, 0.5), 4),
                "evidence": evidence or ["weak, uncorroborated signal"]}

    # Corroborated malicious behaviour: pick a high-level stage from shape.
    if high_volume and (high_fanout or many_hosts):
        stage = STAGE_C2
        evidence.append(
            f"sustained high-volume fan-out (flows={num_flows:.0f}, "
            f"edge_bytes={total_edge_bytes:.0f}, mean_out_degree={mean_out:.2f})"
        )
    elif high_fanout or many_hosts:
        stage = STAGE_LATERAL
        evidence.append(
            f"spread across hosts (num_nodes={num_nodes:.0f}, "
            f"num_edges={num_edges:.0f}, density={density:.3f})"
        )
    else:
        # Corroborated malicious activity without a clear spread/volume shape.
        # Stay high-level: initial access is the least-specific attack tactic
        # here; we do NOT claim execution/persistence/priv-esc/exfil without
        # evidence that distinguishes them.
        stage = STAGE_INITIAL_ACCESS
        evidence.append("malicious activity without clear spread/volume shape")

    # Confidence grows with corroboration; capped so it is never overstated.
    conf = min(0.5 + 0.2 * strong_signals, 0.95)
    if risk is not None:
        # Blend in model risk but keep it bounded and honest.
        conf = min(0.6 * conf + 0.4 * float(risk) + 0.0, 0.98)
    return {"stage": stage, "confidence": round(conf, 4), "evidence": evidence}


# --------------------------------------------------------------------------- #
# Trajectory construction.
# --------------------------------------------------------------------------- #
def build_observed_stages(observed_windows: Sequence[Dict]) -> List[TrajectoryStage]:
    """Build the OBSERVED segment from real cached window states.

    ``observed_windows`` are the cached graph dicts for [t-L+1 .. t] (the same
    sequence fed to the world model). Each becomes an observed stage mapped to a
    high-level MITRE stage from its real behaviour. Confidence reflects evidence
    strength in the observed data (NOT a model forecast).
    """
    stages: List[TrajectoryStage] = []
    n = len(observed_windows)
    for offset, w in enumerate(observed_windows):
        signals = _window_signals(w)
        mapping = map_behaviour_to_stage(signals, risk=None)
        # horizon is <=0 for observed: 0 is the anchor (t), earlier windows are
        # negative offsets so ordering stays chronological before the forecast.
        horizon = offset - (n - 1)   # ..., -2, -1, 0
        stages.append(TrajectoryStage(
            stage=mapping["stage"],
            status=STATUS_OBSERVED,
            confidence=mapping["confidence"],
            evidence=mapping["evidence"],
            horizon=horizon,
            window_index=w.get("window_index"),
            timestamp=w.get("start"),
            source="observed-state",
        ))
    return stages


def build_forecast_stages(
    forecast_risks: Sequence[float],
    *,
    t_index: int,
    forecast_signals: Optional[Sequence[Optional[Dict[str, float]]]] = None,
    last_observed_signals: Optional[Dict[str, float]] = None,
) -> List[TrajectoryStage]:
    """Build the FORECAST segment from the rolled-out risk trajectory.

    ``forecast_risks[k-1]`` is the model's predicted risk at horizon ``t+k``
    (the Phase-5 rollout output). These are PREDICTIONS: every resulting stage
    has ``status="forecast"`` and is phrased as an expectation, never a fact.

    We do NOT have real future window features (that would be leakage). Instead
    the high-level stage is inferred from the model's forecast risk and, when
    available, the observed present state's behaviour shape (as a prior for what
    kind of progression is plausible). Confidence is the model risk itself,
    which honestly decays as the rollout compounds error at deeper horizons.
    """
    stages: List[TrajectoryStage] = []
    prior_shape = last_observed_signals or {}
    for i, risk in enumerate(forecast_risks):
        k = i + 1
        risk = float(risk)
        sig = None
        if forecast_signals is not None and i < len(forecast_signals):
            sig = forecast_signals[i]

        stage, evidence = _forecast_stage_from_risk(risk, prior_shape, sig)
        stages.append(TrajectoryStage(
            stage=stage,
            status=STATUS_FORECAST,
            confidence=round(risk, 6),
            evidence=evidence,
            horizon=k,
            window_index=(t_index + k) if t_index is not None and t_index >= 0 else k,
            timestamp=None,   # future window has no observed timestamp
            source="forecast-rollout",
        ))
    return stages


def _forecast_stage_from_risk(
    risk: float,
    prior_shape: Dict[str, float],
    sig: Optional[Dict[str, float]],
) -> "tuple[str, List[str]]":
    """Infer a HIGH-LEVEL forecast stage from model risk + behaviour prior.

    Kept deliberately coarse: a forecast is uncertain, so we back off to broad
    stages and never assert a precise technique. Progression logic follows the
    kill chain loosely — higher forecast risk and a spread-like present state
    make later-stage tactics (lateral movement / C2) plausible; low risk keeps
    us at reconnaissance or a generic suspicious/benign expectation.
    """
    evidence: List[str] = [f"model forecast risk at this horizon = {risk:.3f}"]

    # If explicit forecast signals were supplied (e.g. from a decoded latent),
    # let the shared mapper decide, then mark it as a forecast expectation.
    if sig is not None:
        mapping = map_behaviour_to_stage(sig, risk=risk)
        evidence.extend(f"(forecast) {e}" for e in mapping["evidence"])
        return mapping["stage"], evidence

    many_hosts = prior_shape.get("num_nodes", 0.0) >= 5.0
    high_fanout = prior_shape.get("mean_node_out_degree", 0.0) >= 2.0
    high_volume = (prior_shape.get("total_edge_bytes", 0.0) >= 1e4
                   or prior_shape.get("num_flows", 0.0) >= 8.0)

    if risk < 0.5:
        # Model does not expect malicious progression; stay coarse.
        stage = STAGE_RECON if (many_hosts or high_fanout) else STAGE_BENIGN
        if many_hosts or high_fanout:
            evidence.append("present state shows host fan-out (possible recon)")
        else:
            evidence.append("forecast risk below decision threshold")
        return stage, evidence

    # Elevated forecast risk -> plausible attack progression, kept high-level.
    if high_volume and (high_fanout or many_hosts):
        stage = STAGE_C2
        evidence.append("present high-volume fan-out; C2 progression plausible")
    elif high_fanout or many_hosts:
        stage = STAGE_LATERAL
        evidence.append("present host spread; lateral movement plausible")
    else:
        stage = STAGE_INITIAL_ACCESS
        evidence.append("elevated risk without spread shape; initial access plausible")
    return stage, evidence


def build_trajectory(
    observed_windows: Sequence[Dict],
    forecast_risks: Sequence[float],
    *,
    t_index: int,
    forecast_signals: Optional[Sequence[Optional[Dict[str, float]]]] = None,
) -> AttackTrajectory:
    """Assemble a full observed→forecast :class:`AttackTrajectory`.

    Parameters
    ----------
    observed_windows : cached graph dicts for [t-L+1 .. t] (real, observed).
    forecast_risks   : Phase-5 rollout risk per horizon t+1 .. t+K (predicted).
    t_index          : anchor window index t.
    forecast_signals : optional per-horizon decoded behaviour signals; if not
                       given, forecast stages are inferred from risk + the
                       observed present state (no future leakage).

    The returned trajectory is temporally ordered (observed strictly before
    forecast) and every forecast stage is flagged ``status="forecast"``.
    """
    observed = build_observed_stages(observed_windows)
    last_signals = _window_signals(observed_windows[-1]) if observed_windows else {}
    forecast = build_forecast_stages(
        forecast_risks, t_index=t_index,
        forecast_signals=forecast_signals,
        last_observed_signals=last_signals,
    )
    traj = AttackTrajectory(t_index=t_index, stages=list(observed) + list(forecast))
    _assert_temporal_order(traj)
    return traj


def _assert_temporal_order(traj: AttackTrajectory) -> None:
    """Guard: observed stages precede forecast stages and horizons increase.

    This encodes the core Phase-7 invariant — a forecast is never placed before
    an observation, and we never claim a forecasted event already occurred.
    """
    seen_forecast = False
    last_h = None
    for s in traj.stages:
        if s.status == STATUS_FORECAST:
            seen_forecast = True
        elif seen_forecast and s.status == STATUS_OBSERVED:
            raise ValueError("observed stage found after a forecast stage (ordering violation)")
        if last_h is not None and s.horizon < last_h:
            raise ValueError(
                f"non-monotonic horizon ordering: {s.horizon} after {last_h}")
        last_h = s.horizon
