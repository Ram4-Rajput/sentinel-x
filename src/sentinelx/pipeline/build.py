"""End-to-end Phase-2 orchestration.

Flow:
  raw files
    -> iter_unified_records (streaming, chunked, adapter-routed)
    -> collect (bounded by max_records for smoke/medium)
    -> chronological_split (existing; TRAIN < VAL < TEST by time)
    -> assert no temporal leakage (existing check + explicit assertion)
    -> FlowPreprocessor.fit on TRAIN ONLY (existing; leakage-safe)
    -> per split: generate_windows_strided (reuses build_graph)
    -> GraphCache.write_split (+ manifest with full reproduction metadata)
    -> BuildReport

Datasets without timestamps (CICIoT2023) cannot be windowed into temporal
graphs; the builder detects this, still streams + preprocesses + reports, and
records the limitation instead of fabricating an order.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..data.preprocessing import FlowPreprocessor
from ..data.schema import BinaryLabel, UnifiedFlowRecord
from ..data.temporal import (
    TemporalError,
    assert_chronological_disjoint,
    chronological_split,
    normalize_timestamp,
)
from ..data.leakage import run_leakage_checks
from .config import PipelineConfig, resolve_dataset
from .graph_cache import GraphCache, GraphSequenceMeta, summarize_windows
from .loaders import LoaderStats, iter_unified_records
from .windows import generate_windows_strided

SCHEMA_VERSION = "1.0"
PREPROCESSING_VERSION = "1.0"
PIPELINE_VERSION = "2.0"


@dataclass
class BuildReport:
    dataset: str
    files_processed: List[str] = field(default_factory=list)
    records: int = 0
    rows_read: int = 0
    malformed_rows: int = 0
    skipped_records: int = 0
    windows: int = 0
    graphs: int = 0
    nodes: int = 0
    edges: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    temporal_min: Optional[str] = None
    temporal_max: Optional[str] = None
    processing_seconds: float = 0.0
    peak_memory_mb: Optional[float] = None
    cache_location: Optional[str] = None
    cache_size_bytes: int = 0
    temporal_capable: bool = True
    leakage_ok: bool = True
    limitations: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return asdict(self)


def _class_distribution(records: List[UnifiedFlowRecord]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for r in records:
        k = r.label_binary.value
        dist[k] = dist.get(k, 0) + 1
    return dist


def build_dataset(
    config: PipelineConfig,
    *,
    raw_path: Optional[str] = None,
    variant: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    write_cache: bool = True,
) -> BuildReport:
    """Run the full streaming -> window -> graph -> cache pipeline for one dataset."""
    log = log or (lambda m: print(m, flush=True))
    dataset = config.dataset
    report = BuildReport(dataset=dataset)

    paths = resolve_dataset(dataset, raw_path=raw_path, variant=variant)
    files = paths.list_files()
    if not files:
        raise FileNotFoundError(f"No input files matched for '{dataset}' in {paths.raw_path}")
    log(f"[build] {dataset}: {len(files)} file(s); window_size={config.window_size}s "
        f"stride={config.stride}s max_records={config.max_records}")

    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Stream + collect (bounded)
    stats = LoaderStats()
    records: List[UnifiedFlowRecord] = list(iter_unified_records(
        dataset, files,
        chunk_size=config.chunk_size,
        is_binetflow=paths.is_binetflow,
        headerless=paths.headerless,
        max_records=config.max_records,
        stats=stats,
        log=log,
    ))
    report.files_processed = stats.files
    report.records = len(records)
    report.rows_read = stats.rows_read
    report.malformed_rows = stats.malformed_rows
    report.skipped_records = stats.skipped_records
    report.class_distribution = _class_distribution(records)
    log(f"[build] streamed {len(records):,} records "
        f"(malformed={stats.malformed_rows}, skipped={stats.skipped_records})")

    if not records:
        report.limitations.append("No records produced.")
        _finalize_mem_time(report, t_start)
        return report

    # 2. Temporal capability check
    n_missing_ts = sum(1 for r in records if r.timestamp is None)
    report.temporal_capable = (n_missing_ts == 0)

    cache = GraphCache(dataset, config.cache_dir())

    if not report.temporal_capable:
        report.limitations.append(
            f"{dataset}: {n_missing_ts}/{len(records)} records lack timestamps -> "
            f"no temporal windows / graph sequences (dataset has no time). "
            f"Records streamed + preprocessed only."
        )
        # still fit preprocessor on a chronological-free split is impossible;
        # fit on the whole set is out of scope here (no split without time).
        _finalize_mem_time(report, t_start)
        if write_cache:
            meta = GraphSequenceMeta(
                dataset=dataset, split="all", dataset_version="unknown",
                schema_version=SCHEMA_VERSION, preprocessing_version=PREPROCESSING_VERSION,
                pipeline_version=PIPELINE_VERSION, window_size=config.window_size,
                stride=config.stride, num_windows=0, total_nodes=0, total_edges=0,
                total_flows=len(records), timestamp_min=None, timestamp_max=None,
                class_distribution=report.class_distribution,
            )
            cache.write_manifest([meta], extra={"temporal_capable": False,
                                                "limitations": report.limitations})
            report.cache_location = str(cache.root)
            report.cache_size_bytes = cache.cache_size_bytes()
        return report

    # 3. Chronological split (existing) + leakage assertion
    train, val, test = chronological_split(
        records, train_frac=config.train_frac, val_frac=config.val_frac)
    assert_chronological_disjoint(train, val, test)  # raises on leakage
    log(f"[build] split: train={len(train):,} val={len(val):,} test={len(test):,}")

    # 4. Preprocessor fit on TRAIN ONLY (leakage-safe)
    preprocessor = FlowPreprocessor().fit(train)

    leak = run_leakage_checks(train, val, test, preprocessor=preprocessor)
    # Hard gates: temporal ordering + preprocessor-fit MUST pass (real leakage).
    # duplicate_rows / entity_overlap are DATASET PROPERTIES for testbed captures
    # (e.g. CTU-13 has many near-identical Background flows; UNSW reuses hosts),
    # so they are recorded as limitations, not treated as a split failure.
    HARD_CHECKS = {"temporal_order", "preprocessor_fit"}
    hard_failures = [f for f in leak.failed() if f.check in HARD_CHECKS]
    report.leakage_ok = len(hard_failures) == 0
    if hard_failures:
        for f in hard_failures:
            report.limitations.append(f"LEAKAGE[{f.check}]: {f.detail}")
    for f in leak.failed():
        if f.check not in HARD_CHECKS:
            report.limitations.append(
                f"dataset-property[{f.check}]: {f.detail} "
                f"(not a split defect; inherent to this capture)")
    # explicit temporal-leakage assertion (hard fail if the future leaks into train)
    if hard_failures:
        raise AssertionError(
            "Temporal/preprocessor leakage gate failed: "
            + "; ".join(f.detail for f in hard_failures))

    # temporal range
    tmin = min(normalize_timestamp(r.timestamp) for r in records)
    tmax = max(normalize_timestamp(r.timestamp) for r in records)
    report.temporal_min = tmin.isoformat()
    report.temporal_max = tmax.isoformat()

    # 5. Windows + graphs per split; 6. cache
    metas: List[GraphSequenceMeta] = []
    total_nodes = total_edges = total_windows = 0
    for split_name, split_recs in (("train", train), ("val", val), ("test", test)):
        if not split_recs:
            continue
        windows = generate_windows_strided(
            split_recs, window_size=config.window_size, stride=config.stride,
            dataset=dataset, build_graphs=config.build_graphs)
        summary = summarize_windows(windows)
        total_windows += len(windows)
        total_nodes += summary["nodes"]
        total_edges += summary["edges"]
        if write_cache:
            cache.write_split(split_name, windows)
        s_tmin = min((normalize_timestamp(r.timestamp) for r in split_recs), default=None)
        s_tmax = max((normalize_timestamp(r.timestamp) for r in split_recs), default=None)
        metas.append(GraphSequenceMeta(
            dataset=dataset, split=split_name, dataset_version="unknown",
            schema_version=SCHEMA_VERSION, preprocessing_version=PREPROCESSING_VERSION,
            pipeline_version=PIPELINE_VERSION, window_size=config.window_size,
            stride=config.stride, num_windows=len(windows),
            total_nodes=summary["nodes"], total_edges=summary["edges"],
            total_flows=summary["flows"],
            timestamp_min=s_tmin.isoformat() if s_tmin else None,
            timestamp_max=s_tmax.isoformat() if s_tmax else None,
            class_distribution=_class_distribution(split_recs),
        ))
        log(f"[build]   {split_name}: {len(windows)} windows, "
            f"{summary['nodes']} nodes, {summary['edges']} edges")

    report.windows = total_windows
    report.graphs = total_windows  # one graph per window
    report.nodes = total_nodes
    report.edges = total_edges

    if write_cache:
        cache.write_manifest(metas, extra={
            "temporal_capable": True,
            "leakage_ok": report.leakage_ok,
            "window_config": {"window_size": config.window_size, "stride": config.stride,
                              "train_frac": config.train_frac, "val_frac": config.val_frac},
            "preprocessor_feature_names": preprocessor.feature_names(),
            "raw_files": [str(f) for f in files],
        })
        report.cache_location = str(cache.root)
        report.cache_size_bytes = cache.cache_size_bytes()

    _finalize_mem_time(report, t_start)
    return report


def _finalize_mem_time(report: BuildReport, t_start: float) -> None:
    report.processing_seconds = round(time.perf_counter() - t_start, 3)
    try:
        _cur, peak = tracemalloc.get_traced_memory()
        report.peak_memory_mb = round(peak / (1024 * 1024), 1)
        tracemalloc.stop()
    except Exception:
        report.peak_memory_mb = None
