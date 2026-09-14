"""Configurable strided temporal windows (Phase-2 addition).

The existing ``sentinelx.data.generate_windows`` produces contiguous,
non-overlapping windows anchored at t0. Phase 2 needs configurable
``window_size`` AND ``stride`` (to allow overlapping sliding windows), so this
module adds ``generate_windows_strided`` that REUSES the existing
``build_graph`` for each window. The original generator is untouched.

Windowing convention (documented):
  * window i covers [t0 + i*stride, t0 + i*stride + window_size)
  * stride == window_size  -> contiguous, non-overlapping (same as existing)
  * stride <  window_size  -> overlapping sliding windows
  * stride >  window_size  -> gapped windows (rare; allowed)
  * time is normalized to UTC; records are sorted by time (never shuffled)
  * datasets without timestamps raise TemporalError (cannot window)
"""

from __future__ import annotations

from datetime import timedelta
from typing import List, Optional, Sequence

from ..data.schema import TemporalNetworkWindow, UnifiedFlowRecord
from ..data.temporal import build_graph, normalize_timestamp

from ..data.temporal import TemporalError


def generate_windows_strided(
    records: Sequence[UnifiedFlowRecord],
    *,
    window_size: float,
    stride: float,
    dataset: Optional[str] = None,
    build_graphs: bool = True,
) -> List[TemporalNetworkWindow]:
    """Slice time-ordered flows into windows of ``window_size`` seconds spaced
    ``stride`` seconds apart. Reuses ``build_graph`` for each window's G_t."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0.")
    if stride <= 0:
        raise ValueError("stride must be > 0.")
    if not records:
        raise TemporalError("No records provided.")
    missing = sum(1 for r in records if r.timestamp is None)
    if missing:
        raise TemporalError(
            f"{missing}/{len(records)} records lack timestamps; cannot window "
            f"(dataset likely CICIoT2023)."
        )

    ds = dataset or records[0].dataset
    ordered = sorted(records, key=lambda r: normalize_timestamp(r.timestamp))
    t0 = normalize_timestamp(ordered[0].timestamp)
    t_last = normalize_timestamp(ordered[-1].timestamp)
    span = (t_last - t0).total_seconds()

    n_windows = int(span // stride) + 1
    windows: List[TemporalNetworkWindow] = []

    # Precompute normalized epoch seconds once (avoids re-normalizing per window).
    times = [(normalize_timestamp(r.timestamp) - t0).total_seconds() for r in ordered]
    n = len(ordered)
    start_ptr = 0  # first index whose time >= current window start
    for i in range(n_windows):
        w_off = i * stride                      # window start offset (seconds from t0)
        w_end_off = w_off + window_size
        w_start = t0 + timedelta(seconds=w_off)
        w_end = t0 + timedelta(seconds=w_end_off)
        # advance start_ptr to first flow with time >= w_off
        while start_ptr < n and times[start_ptr] < w_off:
            start_ptr += 1
        # collect flows in [w_off, w_end_off) starting from start_ptr
        j = start_ptr
        bucket = []
        while j < n and times[j] < w_end_off:
            bucket.append(ordered[j])
            j += 1
        if build_graphs:
            win = build_graph(bucket, dataset=ds, window_index=i, start=w_start, end=w_end)
        else:
            win = TemporalNetworkWindow(
                dataset=ds, window_index=i, start=w_start, end=w_end, flows=list(bucket))
        windows.append(win)
    return windows
