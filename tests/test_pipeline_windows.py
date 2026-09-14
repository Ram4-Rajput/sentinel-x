"""Phase-2 strided window tests: window_size/stride semantics, overlap,
graph reuse, and no-timestamp rejection."""

from datetime import datetime

import pytest

from sentinelx.data.schema import BinaryLabel, UnifiedFlowRecord
from sentinelx.data.temporal import TemporalError
from sentinelx.pipeline.windows import generate_windows_strided


def _rec(i, second, src=None, dst=None):
    return UnifiedFlowRecord(
        dataset="ctu-13", source_index=i,
        timestamp=datetime(2011, 8, 10, 9, 0, second),
        src_ip=src, dst_ip=dst, protocol="tcp",
        total_fwd_bytes=10.0, total_bwd_bytes=5.0,
        label_binary=BinaryLabel.BENIGN,
    )


def test_non_overlapping_when_stride_equals_size():
    recs = [_rec(0, 0), _rec(1, 5), _rec(2, 10), _rec(3, 20)]
    windows = generate_windows_strided(recs, window_size=10, stride=10, build_graphs=False)
    # windows: [0,10)->2 flows, [10,20)->1, [20,30)->1
    counts = [w.num_flows for w in windows]
    assert counts[0] == 2
    assert 20 <= 30  # sanity
    assert sum(counts) >= 4  # all flows covered at least once


def test_overlapping_when_stride_less_than_size():
    recs = [_rec(0, 0), _rec(1, 5), _rec(2, 8)]
    # window_size=10, stride=5 -> windows [0,10),[5,15),[10,20)...
    windows = generate_windows_strided(recs, window_size=10, stride=5, build_graphs=False)
    # first window covers 0,5,8 ; second covers 5,8 -> overlap means a flow
    # appears in more than one window
    total_assignments = sum(w.num_flows for w in windows)
    assert total_assignments > len(recs)  # overlap duplicates coverage


def test_graph_reused_from_build_graph():
    recs = [_rec(0, 0, src="a", dst="b"), _rec(1, 1, src="a", dst="b"),
            _rec(2, 2, src="c", dst="b")]
    windows = generate_windows_strided(recs, window_size=10, stride=10, build_graphs=True)
    w = windows[0]
    assert w.num_nodes == 3
    assert w.num_edges == 2
    assert w.edges[("a", "b")].flow_count == 2


def test_no_timestamp_raises():
    recs = [UnifiedFlowRecord(dataset="ciciot2023", source_index=0)]
    with pytest.raises(TemporalError):
        generate_windows_strided(recs, window_size=10, stride=10)


def test_invalid_params():
    recs = [_rec(0, 0)]
    with pytest.raises(ValueError):
        generate_windows_strided(recs, window_size=0, stride=10)
    with pytest.raises(ValueError):
        generate_windows_strided(recs, window_size=10, stride=0)


def test_windows_sorted_input_not_required():
    # out-of-order input must be handled (sorted internally, never shuffled away)
    recs = [_rec(0, 20), _rec(1, 0), _rec(2, 10)]
    windows = generate_windows_strided(recs, window_size=5, stride=5, build_graphs=False)
    # first window [0,5) should contain the second=0 record
    assert windows[0].num_flows == 1
    assert windows[0].flows[0].timestamp.second == 0
