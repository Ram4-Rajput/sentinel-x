"""Phase-8 (2/4): Propagation analysis of suspicious behaviour on the graph.

Analyses **how suspicious behaviour spreads across the network graph** over the
observed window sequence. This is a *supporting analytical layer* that describes
the dynamics of the real cached graphs — it does NOT replace the World Model
with graph-reachability rules, and it does not itself forecast. It answers
"given what the model / labels flag as suspicious, how is that footprint moving
across the graph in time?".

A node is treated as *affected* in a window when it participates in an edge
flagged suspicious. Two suspicion sources are supported and documented:

* ``"label"`` (default) — the cached edge ``contains_attack`` flag (ground truth
  in the graph cache). Purely descriptive of the data.
* ``"model"`` — an externally supplied per-window set of suspicious edges (e.g.
  ranked by the Phase-8 explainer's edge attribution). This lets propagation be
  driven by the *model's* attributed evidence rather than labels, while keeping
  the two sources explicit.

Measured quantities (per the phase brief)
-----------------------------------------
* affected nodes / newly affected nodes (per window and cumulative)
* propagation velocity (new affected nodes per window)
* direction (fan-out vs fan-in, from edge orientation)
* branching (mean out-degree of affected nodes within the suspicious subgraph)
* persistence (fraction of windows a node stays affected)
* growth (trend of the cumulative affected set)

Everything is computed from the cached window dicts (same format the World
Model consumes); nothing is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

SUSPICION_LABEL = "label"
SUSPICION_MODEL = "model"
VALID_SUSPICION = (SUSPICION_LABEL, SUSPICION_MODEL)


@dataclass
class WindowPropagation:
    """Propagation snapshot for one observed window."""

    timestep: int
    window_index: Optional[int]
    affected: List[str]                 # node ids affected in THIS window
    newly_affected: List[str]           # first-time-affected (vs cumulative)
    cumulative_affected: int            # size of cumulative affected set
    suspicious_edges: int               # count of flagged edges this window
    out_edges: int                      # flagged edges leaving affected nodes
    in_edges: int                       # flagged edges entering affected nodes
    branching: float                    # mean out-degree in suspicious subgraph

    def as_dict(self) -> Dict:
        return {
            "timestep": self.timestep,
            "window_index": self.window_index,
            "affected": list(self.affected),
            "newly_affected": list(self.newly_affected),
            "n_affected": len(self.affected),
            "n_newly_affected": len(self.newly_affected),
            "cumulative_affected": self.cumulative_affected,
            "suspicious_edges": self.suspicious_edges,
            "out_edges": self.out_edges,
            "in_edges": self.in_edges,
            "branching": round(float(self.branching), 6),
        }


@dataclass
class PropagationAnalysis:
    """Full propagation analysis across an observed window sequence."""

    suspicion_source: str
    per_window: List[WindowPropagation] = field(default_factory=list)
    total_affected: int = 0
    velocity: float = 0.0               # mean newly-affected per window
    peak_velocity: int = 0              # max newly-affected in any window
    direction: str = "none"             # "fan-out" | "fan-in" | "balanced" | "none"
    direction_ratio: float = 0.0        # out/(out+in) over flagged edges
    mean_branching: float = 0.0
    persistence: float = 0.0            # mean per-node fraction of windows affected
    growth: float = 0.0                 # slope of cumulative affected over time
    is_spreading: bool = False

    def as_dict(self) -> Dict:
        return {
            "suspicion_source": self.suspicion_source,
            "total_affected": self.total_affected,
            "velocity": round(float(self.velocity), 6),
            "peak_velocity": self.peak_velocity,
            "direction": self.direction,
            "direction_ratio": round(float(self.direction_ratio), 6),
            "mean_branching": round(float(self.mean_branching), 6),
            "persistence": round(float(self.persistence), 6),
            "growth": round(float(self.growth), 6),
            "is_spreading": self.is_spreading,
            "per_window": [w.as_dict() for w in self.per_window],
        }


def _validate_source(source: str) -> str:
    s = str(source).lower()
    if s not in VALID_SUSPICION:
        raise ValueError(f"suspicion source must be one of {VALID_SUSPICION}, got {source!r}")
    return s


def _node_id(window: Dict, idx: int) -> str:
    """Resolve a node id for an index, falling back to a synthetic id."""
    ids = window.get("node_ids")
    if ids and 0 <= idx < len(ids):
        return str(ids[idx])
    return f"n{idx}"


def _suspicious_edges_for_window(
    window: Dict,
    source: str,
    model_edges: Optional[Set[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return the list of (src_idx, dst_idx) edges flagged suspicious.

    * label source : edge_features column 2 (``contains_attack``) >= 0.5.
    * model source : membership in the supplied ``model_edges`` set.
    """
    edge_index = window.get("edge_index") or []
    edge_features = window.get("edge_features") or []
    flagged: List[Tuple[int, int]] = []
    for e, (s, d) in enumerate(edge_index):
        if source == SUSPICION_LABEL:
            ef = edge_features[e] if e < len(edge_features) else None
            is_susp = bool(ef is not None and len(ef) >= 3 and ef[2] >= 0.5)
        else:  # model
            is_susp = (int(s), int(d)) in (model_edges or set())
        if is_susp:
            flagged.append((int(s), int(d)))
    return flagged


def _linfit_slope(ys: Sequence[float]) -> float:
    """Least-squares slope of ys against x = 0..n-1 (growth trend).

    Stdlib-only; returns 0.0 for fewer than two points.
    """
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return float(num / den) if den else 0.0


def analyze_propagation(
    windows: Sequence[Dict],
    *,
    suspicion_source: str = SUSPICION_LABEL,
    model_suspicious_edges: Optional[Sequence[Optional[Set[Tuple[int, int]]]]] = None,
) -> PropagationAnalysis:
    """Analyse how suspicious behaviour spreads across ``windows`` in time.

    Parameters
    ----------
    windows : ordered cached graph dicts (e.g. the observed sequence [t-L+1..t]).
    suspicion_source : "label" (edge ``contains_attack``) or "model" (supplied).
    model_suspicious_edges : when source == "model", a per-window set of
        suspicious (src_idx, dst_idx) tuples aligned with ``windows``.

    Returns a :class:`PropagationAnalysis`. Empty input yields an empty, valid
    analysis (all metrics zero / "none") rather than raising.
    """
    source = _validate_source(suspicion_source)

    per_window: List[WindowPropagation] = []
    cumulative: Set[str] = set()
    affected_counts: Dict[str, int] = {}     # node id -> windows it was affected
    total_out = total_in = 0
    branchings: List[float] = []
    cumulative_series: List[float] = []

    for ts, w in enumerate(windows):
        model_edges = None
        if source == SUSPICION_MODEL and model_suspicious_edges is not None \
                and ts < len(model_suspicious_edges):
            model_edges = model_suspicious_edges[ts]
        flagged = _suspicious_edges_for_window(w, source, model_edges)

        affected_idx: Set[int] = set()
        out_deg: Dict[int, int] = {}
        w_out = w_in = 0
        for (s, d) in flagged:
            affected_idx.add(s)
            affected_idx.add(d)
            out_deg[s] = out_deg.get(s, 0) + 1
            w_out += 1     # each directed flagged edge is one out (from s) ...
            w_in += 1      # ... and one in (to d)
        total_out += w_out
        total_in += w_in

        affected_ids = sorted(_node_id(w, i) for i in affected_idx)
        newly = [nid for nid in affected_ids if nid not in cumulative]
        cumulative.update(affected_ids)
        for nid in affected_ids:
            affected_counts[nid] = affected_counts.get(nid, 0) + 1

        # branching: mean out-degree of source nodes within the suspicious
        # subgraph (how many onward hops each active node fans into).
        src_nodes = [n for n in affected_idx if out_deg.get(n, 0) > 0]
        branching = (sum(out_deg[n] for n in src_nodes) / len(src_nodes)
                     if src_nodes else 0.0)
        branchings.append(branching)

        per_window.append(WindowPropagation(
            timestep=ts,
            window_index=w.get("window_index"),
            affected=affected_ids,
            newly_affected=newly,
            cumulative_affected=len(cumulative),
            suspicious_edges=len(flagged),
            out_edges=w_out, in_edges=w_in, branching=branching,
        ))
        cumulative_series.append(float(len(cumulative)))

    n_win = len(per_window)
    total_affected = len(cumulative)
    newly_per_window = [len(w.newly_affected) for w in per_window]
    velocity = (sum(newly_per_window) / n_win) if n_win else 0.0
    peak_velocity = max(newly_per_window) if newly_per_window else 0

    # direction from aggregate edge orientation. With directed flagged edges,
    # out_edges == in_edges by construction (each edge contributes to both), so
    # we instead compare fan-out breadth vs fan-in breadth of distinct nodes.
    direction, direction_ratio = _direction(per_window, windows, source,
                                             model_suspicious_edges)

    mean_branching = (sum(branchings) / len(branchings)) if branchings else 0.0
    # persistence: for nodes that were ever affected, the mean fraction of
    # windows they remained affected.
    persistence = (sum(c / n_win for c in affected_counts.values())
                   / len(affected_counts)) if affected_counts and n_win else 0.0
    growth = _linfit_slope(cumulative_series)
    is_spreading = growth > 0.0 and peak_velocity > 0

    return PropagationAnalysis(
        suspicion_source=source,
        per_window=per_window,
        total_affected=total_affected,
        velocity=velocity,
        peak_velocity=peak_velocity,
        direction=direction,
        direction_ratio=direction_ratio,
        mean_branching=mean_branching,
        persistence=persistence,
        growth=growth,
        is_spreading=is_spreading,
    )


def _direction(
    per_window: Sequence[WindowPropagation],
    windows: Sequence[Dict],
    source: str,
    model_suspicious_edges: Optional[Sequence[Optional[Set[Tuple[int, int]]]]],
) -> Tuple[str, float]:
    """Classify spread direction from source/sink breadth of flagged edges.

    Intuition: a *fan-out* spread has FEW distinct originating nodes reaching
    MANY distinct destinations (one host infecting many); a *fan-in* spread has
    MANY sources converging on FEW destinations (many hosts beaconing to one
    C2). We therefore report ``direction_ratio = sinks/(sources+sinks)`` — the
    breadth of destinations relative to the total. High ratio (many sinks, few
    sources) -> fan-out; low ratio (few sinks, many sources) -> fan-in.
    """
    sources: Set[str] = set()
    sinks: Set[str] = set()
    for ts, w in enumerate(windows):
        model_edges = None
        if source == SUSPICION_MODEL and model_suspicious_edges is not None \
                and ts < len(model_suspicious_edges):
            model_edges = model_suspicious_edges[ts]
        flagged = _suspicious_edges_for_window(w, source, model_edges)
        for (s, d) in flagged:
            sources.add(_node_id(w, s))
            sinks.add(_node_id(w, d))
    total = len(sources) + len(sinks)
    if total == 0:
        return "none", 0.0
    ratio = len(sinks) / total
    if ratio > 0.55:
        return "fan-out", ratio
    if ratio < 0.45:
        return "fan-in", ratio
    return "balanced", ratio
