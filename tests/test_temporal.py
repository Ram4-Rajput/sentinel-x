"""Temporal tests: timestamp normalization, chronological split (no shuffle),
window generation, and graph construction."""

from datetime import datetime, timezone

import pytest

from sentinelx.data.schema import BinaryLabel, UnifiedFlowRecord
from sentinelx.data.temporal import (
    TemporalError,
    assert_chronological_disjoint,
    build_graph,
    chronological_split,
    generate_windows,
    normalize_timestamp,
)


def _rec(i, second, src=None, dst=None, label=BinaryLabel.BENIGN, fwd=0.0, bwd=0.0):
    return UnifiedFlowRecord(
        dataset="ctu-13", source_index=i,
        timestamp=datetime(2011, 8, 10, 9, 0, second),
        src_ip=src, dst_ip=dst, label_binary=label,
        total_fwd_bytes=fwd, total_bwd_bytes=bwd, protocol="tcp",
    )


def test_normalize_naive_to_utc():
    ts = datetime(2020, 1, 1, 12, 0, 0)
    out = normalize_timestamp(ts)
    assert out.tzinfo == timezone.utc


def test_chronological_split_is_time_ordered_not_shuffled():
    # deliberately out-of-order input
    recs = [_rec(0, 30), _rec(1, 10), _rec(2, 50), _rec(3, 20), _rec(4, 40)]
    train, val, test = chronological_split(recs, train_frac=0.6, val_frac=0.2)
    # 5 records -> 3 train, 1 val, 1 test, all in time order
    assert [r.timestamp.second for r in train] == [10, 20, 30]
    assert [r.timestamp.second for r in val] == [40]
    assert [r.timestamp.second for r in test] == [50]
    assert_chronological_disjoint(train, val, test)  # no overlap


def test_split_without_timestamps_raises():
    recs = [UnifiedFlowRecord(dataset="ciciot2023", source_index=0)]
    with pytest.raises(TemporalError):
        chronological_split(recs)


def test_generate_windows_buckets_by_time():
    # flows at 0,5,10,20 seconds; 10s windows -> [0-10):2, [10-20):1, [20-30):1
    recs = [_rec(0, 0), _rec(1, 5), _rec(2, 10), _rec(3, 20)]
    windows = generate_windows(recs, window_seconds=10, build_graphs=False)
    assert [w.num_flows for w in windows] == [2, 1, 1]


def test_build_graph_entities_and_edges():
    recs = [
        _rec(0, 0, src="10.0.0.1", dst="10.0.0.2", fwd=100.0, bwd=20.0),
        _rec(1, 1, src="10.0.0.1", dst="10.0.0.2", fwd=50.0, bwd=10.0),  # same edge
        _rec(2, 2, src="10.0.0.3", dst="10.0.0.2", label=BinaryLabel.ATTACK, fwd=5.0),
    ]
    w = build_graph(recs, dataset="ctu-13")
    assert w.num_nodes == 3  # .1, .2, .3
    assert w.num_edges == 2  # (.1->.2) and (.3->.2)
    edge = w.edges[("10.0.0.1", "10.0.0.2")]
    assert edge.flow_count == 2
    assert edge.total_bytes == pytest.approx(180.0)  # 100+20+50+10
    attack_edge = w.edges[("10.0.0.3", "10.0.0.2")]
    assert attack_edge.contains_attack is True
    # degrees
    assert w.entities["10.0.0.1"].out_degree == 2
    assert w.entities["10.0.0.2"].in_degree == 3


def test_build_graph_skips_flows_without_entities():
    # CIC/CICIoT flows have no IPs -> contribute to flow list but not the graph
    recs = [UnifiedFlowRecord(dataset="cic-ids2018", source_index=0,
                              timestamp=datetime(2020, 1, 1))]
    w = build_graph(recs, dataset="cic-ids2018")
    assert w.num_flows == 1
    assert w.num_nodes == 0 and w.num_edges == 0


def test_windows_carry_graphs_when_entities_present():
    recs = [_rec(0, 0, src="a", dst="b"), _rec(1, 12, src="b", dst="c")]
    windows = generate_windows(recs, window_seconds=10, build_graphs=True)
    assert windows[0].num_edges == 1
    assert windows[1].num_edges == 1
