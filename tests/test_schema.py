"""Tests for the core representations and availability contract."""

from datetime import datetime

from sentinelx.data.schema import (
    UNIFIED_NUMERIC_FEATURES,
    BinaryLabel,
    CommunicationEdge,
    FeatureAvailability,
    NetworkEntity,
    TemporalNetworkWindow,
    UnifiedFlowRecord,
)


def test_feature_vector_order_and_none_preserved():
    rec = UnifiedFlowRecord(dataset="x", source_index=0, flow_duration=1.5, dst_port=80)
    vec = rec.feature_vector()
    assert len(vec) == len(UNIFIED_NUMERIC_FEATURES)
    # flow_duration is first in canonical order
    assert vec[0] == 1.5
    # ttl_src is unavailable here -> stays None (not fabricated)
    assert vec[UNIFIED_NUMERIC_FEATURES.index("ttl_src")] is None


def test_has_entities_and_time():
    rec = UnifiedFlowRecord(dataset="x", source_index=0)
    assert not rec.has_entities()
    assert not rec.has_time()
    rec.src_ip, rec.dst_ip = "10.0.0.1", "10.0.0.2"
    rec.timestamp = datetime(2020, 1, 1)
    assert rec.has_entities()
    assert rec.has_time()


def test_availability_consistency_detects_gaps():
    # A malformed contract (missing features) must raise.
    av = FeatureAvailability(dataset="bad", present=("flow_duration",))
    try:
        av.assert_consistent()
        assert False, "expected ValueError for unclassified features"
    except ValueError:
        pass


def test_availability_detects_double_classification():
    av = FeatureAvailability(
        dataset="bad",
        present=UNIFIED_NUMERIC_FEATURES + ("protocol",),
        unavailable=("flow_duration",),  # also in present -> conflict
    )
    try:
        av.assert_consistent()
        assert False, "expected ValueError for double classification"
    except ValueError:
        pass


def test_entity_observe_updates_seen_and_bytes():
    e = NetworkEntity(entity_id="10.0.0.1", dataset="x")
    e.observe(sent_bytes=100.0, ts=datetime(2020, 1, 2))
    e.observe(sent_bytes=50.0, ts=datetime(2020, 1, 1))
    assert e.total_bytes_sent == 150.0
    assert e.first_seen == datetime(2020, 1, 1)
    assert e.last_seen == datetime(2020, 1, 2)
    assert e.flow_count == 2


def test_window_attack_ratio():
    flows = [
        UnifiedFlowRecord(dataset="x", source_index=0, label_binary=BinaryLabel.ATTACK),
        UnifiedFlowRecord(dataset="x", source_index=1, label_binary=BinaryLabel.BENIGN),
    ]
    w = TemporalNetworkWindow(dataset="x", window_index=0, start=None, end=None, flows=flows)
    assert w.attack_ratio() == 0.5
    assert w.num_flows == 2


def test_communication_edge_key():
    e = CommunicationEdge(src_id="a", dst_id="b", dataset="x")
    assert e.key == ("a", "b")
