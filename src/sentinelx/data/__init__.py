"""Sentinel-X unified data layer.

Public API:
    Core representations:
        UnifiedFlowRecord, PacketFeatureRecord, NetworkEntity,
        CommunicationEdge, TemporalNetworkWindow, FeatureAvailability

    Adapters (dataset -> UnifiedFlowRecord stream):
        get_adapter, list_adapters, BaseDatasetAdapter

    Preprocessing (train-only fit/transform):
        FlowPreprocessor

    Temporal utilities:
        normalize_timestamp, chronological_split, generate_windows,
        build_graph

    Leakage checks:
        run_leakage_checks
"""

from .schema import (
    UNIFIED_NUMERIC_FEATURES,
    CommunicationEdge,
    FeatureAvailability,
    NetworkEntity,
    PacketFeatureRecord,
    TemporalNetworkWindow,
    UnifiedFlowRecord,
)
from .adapters import BaseDatasetAdapter, get_adapter, list_adapters
from .preprocessing import FlowPreprocessor
from .temporal import (
    build_graph,
    chronological_split,
    generate_windows,
    normalize_timestamp,
)
from .leakage import LeakageReport, run_leakage_checks

__all__ = [
    "UNIFIED_NUMERIC_FEATURES",
    "UnifiedFlowRecord",
    "PacketFeatureRecord",
    "NetworkEntity",
    "CommunicationEdge",
    "TemporalNetworkWindow",
    "FeatureAvailability",
    "BaseDatasetAdapter",
    "get_adapter",
    "list_adapters",
    "FlowPreprocessor",
    "normalize_timestamp",
    "chronological_split",
    "generate_windows",
    "build_graph",
    "run_leakage_checks",
    "LeakageReport",
]
