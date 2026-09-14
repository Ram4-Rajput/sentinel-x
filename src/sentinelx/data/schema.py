"""Common internal representation for the Sentinel-X unified data layer.

Design rules (from sentinel-x-dataset-research + Phase-1 audit):
  * A feature that a dataset genuinely lacks is represented as ``None`` and
    recorded in ``FeatureAvailability``. It is NEVER fabricated / imputed with a
    fake constant at the adapter stage. (Train-only imputation happens later, in
    preprocessing, and only for features the dataset actually provides.)
  * Packet-level features are only populated when a real packet source exists.
    No packet feature is ever synthesised from flow statistics.
  * Label taxonomies are kept per-dataset (``label_multiclass``); a coarse
    ``label_binary`` {benign, attack} view is provided for cross-dataset work.

The five required representations:
  UnifiedFlowRecord       - one network flow, mapped to the canonical schema
  PacketFeatureRecord     - packet-level features (only when a packet source exists)
  NetworkEntity           - a host/node in the dynamic graph G_t
  CommunicationEdge       - a directed communication between two entities
  TemporalNetworkWindow   - a time-bounded slice: flows + entities + edges
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical numeric feature order. Adapters populate a subset of these; the
# preprocessor consumes exactly this ordered list so every dataset produces a
# feature vector with identical column semantics (missing -> None -> handled
# by the preprocessor's train-only imputation, never by the adapter).
# ---------------------------------------------------------------------------
UNIFIED_NUMERIC_FEATURES: Tuple[str, ...] = (
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "flow_bytes_per_s",
    "flow_pkts_per_s",
    "flow_iat_mean",
    "fwd_iat_mean",
    "syn_flag_count",
    "ack_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "init_win_bytes_fwd",
    "ttl_src",
    "ttl_dst",
    "header_length",
    "src_port",
    "dst_port",
)

# Categorical fields carried through the pipeline (encoded by the preprocessor).
UNIFIED_CATEGORICAL_FEATURES: Tuple[str, ...] = ("protocol",)


class BinaryLabel(str, Enum):
    """Coarse cross-dataset label view. 'unknown' is used for CTU-13
    Background flows, which the audit flags as NOT confirmed-benign."""

    BENIGN = "benign"
    ATTACK = "attack"
    UNKNOWN = "unknown"


@dataclass
class FeatureAvailability:
    """Honest record of which unified features a source actually provides.

    ``present`` -> mapped from a real column.
    ``derived``  -> computed from other present columns (documented per adapter).
    ``unavailable`` -> the source genuinely lacks it; value stays ``None``.

    This object travels with every batch so downstream code (and audits) can
    verify that no unavailable feature was silently invented.
    """

    dataset: str
    present: Tuple[str, ...] = ()
    derived: Tuple[str, ...] = ()
    unavailable: Tuple[str, ...] = ()
    # Free-text, per-field notes documenting each transformation.
    transform_notes: Dict[str, str] = field(default_factory=dict)

    def status(self, feature: str) -> str:
        if feature in self.present:
            return "present"
        if feature in self.derived:
            return "derived"
        if feature in self.unavailable:
            return "unavailable"
        return "unknown"

    def assert_consistent(self) -> None:
        """All canonical features must be classified exactly once."""
        classified = set(self.present) | set(self.derived) | set(self.unavailable)
        canonical = set(UNIFIED_NUMERIC_FEATURES) | set(UNIFIED_CATEGORICAL_FEATURES)
        missing = canonical - classified
        if missing:
            raise ValueError(
                f"{self.dataset}: features not classified as present/derived/"
                f"unavailable: {sorted(missing)}"
            )
        overlap_pd = set(self.present) & set(self.derived)
        overlap_pu = set(self.present) & set(self.unavailable)
        overlap_du = set(self.derived) & set(self.unavailable)
        if overlap_pd or overlap_pu or overlap_du:
            raise ValueError(
                f"{self.dataset}: features double-classified: "
                f"{sorted(overlap_pd | overlap_pu | overlap_du)}"
            )


@dataclass
class UnifiedFlowRecord:
    """One network flow mapped onto the canonical Sentinel-X schema.

    Every numeric feature is ``Optional[float]``: ``None`` means the source
    dataset does not provide it (see the owning ``FeatureAvailability``).
    """

    # -- provenance --
    dataset: str
    source_index: int  # row index within the source file (for traceability)

    # -- entity / graph identity (None when the dataset lacks IPs) --
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None

    # -- time (None when the dataset has no timestamp, e.g. CICIoT2023) --
    timestamp: Optional[datetime] = None

    # -- protocol --
    protocol: Optional[str] = None

    # -- flow volume / duration --
    flow_duration: Optional[float] = None
    total_fwd_packets: Optional[float] = None
    total_bwd_packets: Optional[float] = None
    total_fwd_bytes: Optional[float] = None
    total_bwd_bytes: Optional[float] = None

    # -- rates --
    flow_bytes_per_s: Optional[float] = None
    flow_pkts_per_s: Optional[float] = None

    # -- inter-arrival time --
    flow_iat_mean: Optional[float] = None
    fwd_iat_mean: Optional[float] = None

    # -- TCP flag counts --
    syn_flag_count: Optional[float] = None
    ack_flag_count: Optional[float] = None
    fin_flag_count: Optional[float] = None
    rst_flag_count: Optional[float] = None

    # -- header / window / TTL (mostly UNSW-NB15 only) --
    init_win_bytes_fwd: Optional[float] = None
    ttl_src: Optional[float] = None
    ttl_dst: Optional[float] = None
    header_length: Optional[float] = None

    # -- labels --
    label_multiclass: Optional[str] = None  # native per-dataset taxonomy
    label_binary: BinaryLabel = BinaryLabel.UNKNOWN

    def feature_vector(self, feature_order: Tuple[str, ...] = UNIFIED_NUMERIC_FEATURES):
        """Return numeric features in canonical order (None preserved)."""
        return [getattr(self, name) for name in feature_order]

    def has_entities(self) -> bool:
        return self.src_ip is not None and self.dst_ip is not None

    def has_time(self) -> bool:
        return self.timestamp is not None


@dataclass
class PacketFeatureRecord:
    """Packet-level features. Only created when a REAL packet source (PCAP)
    is parsed. There is deliberately no adapter path that fabricates these
    from flow statistics."""

    dataset: str
    flow_source_index: int
    packet_index: int
    timestamp: Optional[datetime] = None
    packet_length: Optional[int] = None
    ttl: Optional[int] = None
    tcp_flags: Optional[str] = None
    payload_bytes: Optional[int] = None
    source: str = "pcap"  # provenance; must be a real capture, never synthetic


@dataclass
class NetworkEntity:
    """A node in the dynamic network graph G_t (typically a host IP)."""

    entity_id: str  # canonical id (usually the IP string)
    dataset: str
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    out_degree: int = 0
    in_degree: int = 0
    total_bytes_sent: float = 0.0
    total_bytes_received: float = 0.0
    flow_count: int = 0

    def observe(self, *, sent_bytes: float = 0.0, recv_bytes: float = 0.0,
                ts: Optional[datetime] = None) -> None:
        self.total_bytes_sent += sent_bytes
        self.total_bytes_received += recv_bytes
        self.flow_count += 1
        if ts is not None:
            if self.first_seen is None or ts < self.first_seen:
                self.first_seen = ts
            if self.last_seen is None or ts > self.last_seen:
                self.last_seen = ts


@dataclass
class CommunicationEdge:
    """A directed communication between two entities within a window."""

    src_id: str
    dst_id: str
    dataset: str
    flow_count: int = 0
    total_bytes: float = 0.0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    # True if any flow on this edge carried an attack label.
    contains_attack: bool = False
    protocols: Tuple[str, ...] = ()

    @property
    def key(self) -> Tuple[str, str]:
        return (self.src_id, self.dst_id)


@dataclass
class TemporalNetworkWindow:
    """A time-bounded slice of the network: the flows in [start, end), plus the
    entities and directed edges induced by those flows. This is the unit fed to
    the Temporal GNN + GRU/LSTM in later phases."""

    dataset: str
    window_index: int
    start: Optional[datetime]
    end: Optional[datetime]
    flows: List[UnifiedFlowRecord] = field(default_factory=list)
    entities: Dict[str, NetworkEntity] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], CommunicationEdge] = field(default_factory=dict)

    @property
    def num_flows(self) -> int:
        return len(self.flows)

    @property
    def num_nodes(self) -> int:
        return len(self.entities)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def attack_ratio(self) -> float:
        if not self.flows:
            return 0.0
        n_attack = sum(1 for f in self.flows if f.label_binary == BinaryLabel.ATTACK)
        return n_attack / len(self.flows)


def flow_field_names() -> Tuple[str, ...]:
    """All field names on UnifiedFlowRecord (used by schema validation)."""
    return tuple(f.name for f in fields(UnifiedFlowRecord))
