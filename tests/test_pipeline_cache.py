"""Phase-2 graph cache tests: write, reload, manifest, feature consistency,
determinism, and that reload needs no raw CSV."""

from datetime import datetime

from sentinelx.data.schema import BinaryLabel, UnifiedFlowRecord
from sentinelx.pipeline.graph_cache import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    GraphCache,
    GraphSequenceMeta,
    window_to_graph_dict,
)
from sentinelx.pipeline.windows import generate_windows_strided


def _rec(i, second, src, dst, attack=False):
    return UnifiedFlowRecord(
        dataset="ctu-13", source_index=i,
        timestamp=datetime(2011, 8, 10, 9, 0, second),
        src_ip=src, dst_ip=dst, protocol="tcp",
        total_fwd_bytes=100.0, total_bwd_bytes=20.0,
        label_binary=BinaryLabel.ATTACK if attack else BinaryLabel.BENIGN,
    )


def _windows():
    recs = [
        _rec(0, 0, "a", "b"), _rec(1, 1, "a", "b"),
        _rec(2, 2, "c", "b", attack=True),
    ]
    return generate_windows_strided(recs, window_size=10, stride=10, dataset="ctu-13")


def test_window_to_graph_dict_shapes():
    w = _windows()[0]
    g = window_to_graph_dict(w)
    assert g["num_nodes"] == 3
    assert g["num_edges"] == 2
    assert len(g["node_features"][0]) == len(NODE_FEATURE_NAMES)
    assert len(g["edge_features"][0]) == len(EDGE_FEATURE_NAMES)
    assert g["label_any_attack"] == 1  # one attack flow present


def test_cache_write_and_reload_without_raw(tmp_path):
    cache = GraphCache("ctu-13", tmp_path / "ctu-13")
    windows = _windows()
    cache.write_split("train", windows)
    # reload purely from cache (no raw CSV involved)
    graphs = cache.read_split("train")
    assert len(graphs) == len(windows)
    assert graphs[0]["num_nodes"] == 3
    assert cache.cache_size_bytes() > 0


def test_manifest_roundtrip(tmp_path):
    cache = GraphCache("ctu-13", tmp_path / "ctu-13")
    meta = GraphSequenceMeta(
        dataset="ctu-13", split="train", dataset_version="v1",
        schema_version="1.0", preprocessing_version="1.0", pipeline_version="2.0",
        window_size=10, stride=10, num_windows=1, total_nodes=3, total_edges=2,
        total_flows=3, timestamp_min="2011-08-10T09:00:00", timestamp_max="2011-08-10T09:00:02",
        class_distribution={"benign": 2, "attack": 1},
    )
    cache.write_split("train", _windows())
    cache.write_manifest([meta], extra={"temporal_capable": True})
    man = cache.read_manifest()
    assert man["dataset"] == "ctu-13"
    assert man["splits"]["train"]["window_size"] == 10
    assert man["node_feature_names"] == NODE_FEATURE_NAMES
    assert man["temporal_capable"] is True


def test_deterministic_serialization(tmp_path):
    # Same windows serialize to identical graph dicts (deterministic processing).
    w = _windows()[0]
    g1 = window_to_graph_dict(w)
    g2 = window_to_graph_dict(w)
    assert g1 == g2


def test_reload_missing_split_raises(tmp_path):
    import pytest
    cache = GraphCache("ctu-13", tmp_path / "ctu-13")
    with pytest.raises(FileNotFoundError):
        cache.read_split("test")
