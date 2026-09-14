"""Phase-2 end-to-end build tests: streaming -> split -> window -> graph ->
cache, plus temporal-leakage gate and the non-temporal (CICIoT2023-like) path.
Uses synthetic CTU-13-style .binetflow and CICIoT-style CSVs in a temp dir."""

import csv
from pathlib import Path

import pytest

from sentinelx.pipeline.build import build_dataset
from sentinelx.pipeline.config import PipelineConfig
from sentinelx.pipeline.graph_cache import GraphCache


CTU_HEADER = ["StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr",
              "Dport", "State", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes", "Label"]


def _write_ctu(path: Path, n: int):
    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CTU_HEADER)
        for i in range(n):
            # spread over time so windows form; vary bytes so keys differ
            sec = i % 60
            minute = (i // 60) % 60
            src = hosts[i % len(hosts)]
            dst = hosts[(i + 1) % len(hosts)]
            label = "flow=From-Botnet-V42" if i % 10 == 0 else "flow=Background"
            w.writerow([f"2011/08/10 09:{minute:02d}:{sec:02d}.000000", "1.0",
                        "tcp", src, 1000 + i, "->", dst, 80, "CON",
                        "0", "0", i + 1, 100 + i, 40 + i, label])


CICIOT_HEADER = ["flow_duration", "Header_Length", "Protocol Type", "Rate", "Srate",
                 "syn_count", "ack_count", "fin_count", "rst_count",
                 "Tot sum", "IAT", "Number", "label"]


def _write_ciciot(path: Path, n: int):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CICIOT_HEADER)
        for i in range(n):
            label = "DDoS-SYN_Flood" if i % 3 else "BenignTraffic"
            w.writerow([0.01 * i, 54, 6, 1.2, 1.2, i % 2, 0, 0, 0,
                        500 + i, 83000000 + i, 9.5, label])


def test_build_ctu_end_to_end(tmp_path):
    raw = tmp_path / "ctu"
    raw.mkdir()
    _write_ctu(raw / "capture.binetflow", 600)
    cache_root = tmp_path / "processed"
    cfg = PipelineConfig(dataset="ctu-13", window_size=30, stride=30,
                         processed_root=cache_root)
    report = build_dataset(cfg, raw_path=str(raw), log=lambda m: None)

    assert report.records == 600
    assert report.temporal_capable is True
    assert report.leakage_ok is True          # temporal gate passes
    assert report.windows > 0
    assert report.graphs == report.windows
    assert report.nodes > 0 and report.edges > 0
    assert report.temporal_min < report.temporal_max
    assert report.cache_location is not None

    # cache reloads without raw
    cache = GraphCache("ctu-13", cache_root / "ctu-13")
    man = cache.read_manifest()
    assert set(man["splits"]).issubset({"train", "val", "test"})
    train_graphs = cache.read_split("train")
    assert len(train_graphs) >= 1


def test_build_ctu_overlapping_windows(tmp_path):
    raw = tmp_path / "ctu"
    raw.mkdir()
    _write_ctu(raw / "capture.binetflow", 300)
    cfg = PipelineConfig(dataset="ctu-13", window_size=30, stride=10,
                         processed_root=tmp_path / "p")
    report = build_dataset(cfg, raw_path=str(raw), log=lambda m: None)
    # overlapping stride < size still builds and stays leakage-safe on hard gates
    assert report.windows > 0
    assert report.leakage_ok is True


def test_build_ciciot_no_timestamp_is_graceful(tmp_path):
    raw = tmp_path / "ciciot"
    raw.mkdir()
    _write_ciciot(raw / "train.csv", 200)
    cfg = PipelineConfig(dataset="ciciot2023", processed_root=tmp_path / "p")
    report = build_dataset(cfg, raw_path=str(raw), log=lambda m: None)
    assert report.records == 200
    assert report.temporal_capable is False    # no timestamps
    assert report.windows == 0                 # no fake ordering
    assert any("no temporal" in lim.lower() for lim in report.limitations)
    # manifest still written
    cache = GraphCache("ciciot2023", (tmp_path / "p") / "ciciot2023")
    man = cache.read_manifest()
    assert man["temporal_capable"] is False


def test_max_records_smoke(tmp_path):
    raw = tmp_path / "ctu"
    raw.mkdir()
    _write_ctu(raw / "capture.binetflow", 1000)
    cfg = PipelineConfig(dataset="ctu-13", window_size=30, stride=30,
                         max_records=100, processed_root=tmp_path / "p")
    report = build_dataset(cfg, raw_path=str(raw), log=lambda m: None)
    assert report.records == 100  # capped


def test_deterministic_report(tmp_path):
    raw = tmp_path / "ctu"
    raw.mkdir()
    _write_ctu(raw / "capture.binetflow", 300)
    cfg = PipelineConfig(dataset="ctu-13", window_size=30, stride=30,
                         processed_root=tmp_path / "p1")
    r1 = build_dataset(cfg, raw_path=str(raw), log=lambda m: None)
    cfg2 = PipelineConfig(dataset="ctu-13", window_size=30, stride=30,
                          processed_root=tmp_path / "p2")
    r2 = build_dataset(cfg2, raw_path=str(raw), log=lambda m: None)
    # deterministic structural outputs
    assert r1.records == r2.records
    assert r1.windows == r2.windows
    assert r1.nodes == r2.nodes
    assert r1.edges == r2.edges
    assert r1.class_distribution == r2.class_distribution
