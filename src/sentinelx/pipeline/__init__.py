"""Sentinel-X Phase-2 data pipeline.

Builds ON TOP of the existing unified data layer (sentinelx.data) — it does not
replace adapters, schema, preprocessing, temporal split, window generation, or
graph construction. It adds:

    config        - dataset path resolution (no hardcoded paths) + versions
    loaders       - streaming/chunked readers -> existing adapters
    windows       - configurable window_size/stride overlapping windows
                    (reuses data.build_graph); the existing fixed generator stays
    graph_cache   - training-friendly cache of graph sequences + reload
    build         - end-to-end orchestration (stream -> split -> window -> graph
                    -> cache) with a processing report
"""

from .config import DatasetPaths, PipelineConfig, resolve_dataset
from .loaders import iter_unified_records, iter_raw_chunks, LoaderStats
from .windows import generate_windows_strided
from .graph_cache import GraphCache, GraphSequenceMeta
from .build import build_dataset, BuildReport

__all__ = [
    "DatasetPaths",
    "PipelineConfig",
    "resolve_dataset",
    "iter_unified_records",
    "iter_raw_chunks",
    "LoaderStats",
    "generate_windows_strided",
    "GraphCache",
    "GraphSequenceMeta",
    "build_dataset",
    "BuildReport",
]

SCHEMA_VERSION = "1.0"
PREPROCESSING_VERSION = "1.0"
PIPELINE_VERSION = "2.0"
