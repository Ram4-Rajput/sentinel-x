"""Temporal utilities: timestamp normalization, chronological splitting,
temporal window generation, and dynamic-graph (G_t) construction.

Leakage-safety (per sentinel-x-ml-research + model-evaluation skills):
  * chronological_split sorts by time and cuts by time — never shuffles. The
    future never lands in the training split.
  * Datasets without timestamps (CICIoT2023) cannot be windowed/graph-built;
    the functions raise a clear error rather than silently faking an order.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import (
    BinaryLabel,
    CommunicationEdge,
    NetworkEntity,
    TemporalNetworkWindow,
    UnifiedFlowRecord,
)


class TemporalError(ValueError):
    """Raised when a temporal operation is attempted on data that lacks time."""


def normalize_timestamp(ts: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to timezone-aware UTC. Naive datetimes are assumed
    UTC (documented convention). None passes through."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _require_timestamps(records: Sequence[UnifiedFlowRecord]) -> None:
    if not records:
        raise TemporalError("No records provided.")
    missing = sum(1 for r in records if r.timestamp is None)
    if missing:
        raise TemporalError(
            f"{missing}/{len(records)} records have no timestamp; this dataset "
            f"cannot be split/windowed chronologically (e.g. CICIoT2023)."
        )


def chronological_split(
    records: Sequence[UnifiedFlowRecord],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[List[UnifiedFlowRecord], List[UnifiedFlowRecord], List[UnifiedFlowRecord]]:
    """Sort by timestamp and cut into train/val/test by TIME (no shuffle).

    Returns (train, val, test). test_frac = 1 - train - val.
    """
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("Require 0<train_frac<1, 0<=val_frac<1, train+val<1.")
    _require_timestamps(records)

    ordered = sorted(records, key=lambda r: normalize_timestamp(r.timestamp))
    n = len(ordered)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = ordered[:n_train]
    val = ordered[n_train:n_train + n_val]
    test = ordered[n_train + n_val:]
    return train, val, test


def assert_chronological_disjoint(
    train: Sequence[UnifiedFlowRecord],
    val: Sequence[UnifiedFlowRecord],
    test: Sequence[UnifiedFlowRecord],
) -> None:
    """Verify train max-time <= val min-time <= test min-time (no overlap)."""
    def _max(rs):
        return max(normalize_timestamp(r.timestamp) for r in rs) if rs else None

    def _min(rs):
        return min(normalize_timestamp(r.timestamp) for r in rs) if rs else None

    tr_max, va_min, va_max, te_min = _max(train), _min(val), _max(val), _min(test)
    if train and val and tr_max > va_min:
        raise TemporalError("Train overlaps val in time (leakage).")
    if val and test and va_max > te_min:
        raise TemporalError("Val overlaps test in time (leakage).")
    if train and test and not val and tr_max > te_min:
        raise TemporalError("Train overlaps test in time (leakage).")


def build_graph(
    flows: Sequence[UnifiedFlowRecord],
    *,
    dataset: str,
    window_index: int = 0,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> TemporalNetworkWindow:
    """Construct entities + directed communication edges from flows.

    Requires flows with src_ip/dst_ip (CTU-13, UNSW-NB15). Flows lacking entities
    are skipped from the graph but still counted in the window's flow list.
    """
    window = TemporalNetworkWindow(
        dataset=dataset, window_index=window_index, start=start, end=end,
        flows=list(flows),
    )
    for f in flows:
        if not f.has_entities():
            continue
        ts = normalize_timestamp(f.timestamp)
        sent = f.total_fwd_bytes or 0.0
        recv = f.total_bwd_bytes or 0.0

        src = window.entities.setdefault(
            f.src_ip, NetworkEntity(entity_id=f.src_ip, dataset=dataset))
        dst = window.entities.setdefault(
            f.dst_ip, NetworkEntity(entity_id=f.dst_ip, dataset=dataset))
        src.observe(sent_bytes=sent, ts=ts)
        dst.observe(recv_bytes=recv, ts=ts)
        src.out_degree += 1
        dst.in_degree += 1

        key = (f.src_ip, f.dst_ip)
        edge = window.edges.get(key)
        if edge is None:
            edge = CommunicationEdge(src_id=f.src_ip, dst_id=f.dst_ip, dataset=dataset)
            window.edges[key] = edge
        edge.flow_count += 1
        edge.total_bytes += sent + recv
        if f.label_binary == BinaryLabel.ATTACK:
            edge.contains_attack = True
        if f.protocol and f.protocol not in edge.protocols:
            edge.protocols = tuple(sorted(set(edge.protocols) | {f.protocol}))
        if ts is not None:
            if edge.first_seen is None or ts < edge.first_seen:
                edge.first_seen = ts
            if edge.last_seen is None or ts > edge.last_seen:
                edge.last_seen = ts
    return window


def generate_windows(
    records: Sequence[UnifiedFlowRecord],
    *,
    window_seconds: float,
    dataset: Optional[str] = None,
    build_graphs: bool = True,
) -> List[TemporalNetworkWindow]:
    """Slice time-ordered flows into fixed-duration windows [start, start+W).

    Each window optionally builds its own G_t (entities + edges). Windows are
    contiguous and non-overlapping, anchored at the first flow's timestamp.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0.")
    _require_timestamps(records)
    ds = dataset or records[0].dataset

    ordered = sorted(records, key=lambda r: normalize_timestamp(r.timestamp))
    t0 = normalize_timestamp(ordered[0].timestamp)

    windows: List[TemporalNetworkWindow] = []
    bucket: List[UnifiedFlowRecord] = []
    widx = 0
    win_start = t0

    def _flush(w_start, idx, flows):
        w_end = _add_seconds(w_start, window_seconds)
        if build_graphs:
            return build_graph(flows, dataset=ds, window_index=idx,
                               start=w_start, end=w_end)
        return TemporalNetworkWindow(dataset=ds, window_index=idx,
                                     start=w_start, end=w_end, flows=list(flows))

    for r in ordered:
        ts = normalize_timestamp(r.timestamp)
        while ts >= _add_seconds(win_start, window_seconds):
            windows.append(_flush(win_start, widx, bucket))
            widx += 1
            bucket = []
            win_start = _add_seconds(win_start, window_seconds)
        bucket.append(r)
    if bucket:
        windows.append(_flush(win_start, widx, bucket))
    return windows


def _add_seconds(ts: datetime, seconds: float) -> datetime:
    from datetime import timedelta
    return ts + timedelta(seconds=seconds)
