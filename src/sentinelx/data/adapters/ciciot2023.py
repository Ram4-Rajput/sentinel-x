"""CICIoT2023 adapter (47-column engineered flow schema).

Audit facts honoured here:
  * NO timestamp column -> timestamp UNAVAILABLE (cannot form temporal windows
    or graphs from this dataset alone).
  * NO src/dst IP, no explicit port column -> entity fields UNAVAILABLE.
  * 'Protocol Type' is a numeric proto id; protocol-presence flags (TCP/UDP/...)
    exist as separate columns but we map the single numeric id to `protocol`.
  * Single aggregate 'IAT' -> mapped to flow_iat_mean (documented approximation).
  * flag *_count columns map to syn/ack/fin/rst counts.
  * 'BenignTraffic' -> binary benign; else attack.

TTL/window are genuinely absent and are NOT fabricated.
"""

from __future__ import annotations

from typing import Dict

from ..schema import FeatureAvailability, UnifiedFlowRecord
from .base import BaseDatasetAdapter, to_float


class CICIoT2023Adapter(BaseDatasetAdapter):
    dataset_name = "ciciot2023"
    required_columns = ("flow_duration", "Protocol Type", "IAT", "label")

    def availability(self) -> FeatureAvailability:
        present = (
            "flow_duration", "flow_bytes_per_s", "flow_pkts_per_s",
            "flow_iat_mean", "syn_flag_count", "ack_flag_count",
            "fin_flag_count", "rst_flag_count", "header_length",
            "total_fwd_packets", "total_fwd_bytes", "protocol",
        )
        unavailable = (
            "total_bwd_packets", "total_bwd_bytes", "fwd_iat_mean",
            "init_win_bytes_fwd", "ttl_src", "ttl_dst",
            "src_port", "dst_port",
        )
        notes = {
            "flow_iat_mean": "single aggregate 'IAT' mapped to flow_iat_mean (approx; no fwd/bwd split).",
            "flow_bytes_per_s": "'Rate' used as bytes/s proxy (units approximate).",
            "flow_pkts_per_s": "'Srate' used as pkts/s proxy (units approximate).",
            "total_fwd_packets": "'Number' used as packet-count proxy.",
            "total_fwd_bytes": "'Tot sum' used as aggregate-byte proxy.",
            "header_length": "'Header_Length' direct.",
            "protocol": "'Protocol Type' numeric proto id (string-cast).",
            "timestamp": "UNAVAILABLE: dataset has no timestamp column.",
            "src_ip": "UNAVAILABLE: no IP columns.",
            "dst_ip": "UNAVAILABLE: no IP columns.",
            "src_port": "UNAVAILABLE: only protocol-presence flags.",
            "dst_port": "UNAVAILABLE: only protocol-presence flags.",
            "ttl_src": "UNAVAILABLE.",
            "ttl_dst": "UNAVAILABLE.",
            "total_bwd_packets": "UNAVAILABLE: no directional split.",
            "total_bwd_bytes": "UNAVAILABLE: no directional split.",
        }
        return FeatureAvailability(
            dataset=self.dataset_name, present=present, derived=(),
            unavailable=unavailable, transform_notes=notes,
        )

    def map_row(self, row: Dict[str, object], index: int) -> UnifiedFlowRecord:
        proto = row.get("Protocol Type")
        proto_str = None if proto in (None, "") else str(proto).strip()

        label = row.get("label")
        label_str = None if label is None else str(label).strip()

        return UnifiedFlowRecord(
            dataset=self.dataset_name,
            source_index=index,
            src_ip=None,
            dst_ip=None,
            src_port=None,
            dst_port=None,
            timestamp=None,  # dataset has no time
            protocol=proto_str,
            flow_duration=to_float(row.get("flow_duration")),
            total_fwd_packets=to_float(row.get("Number")),
            total_bwd_packets=None,
            total_fwd_bytes=to_float(row.get("Tot sum")),
            total_bwd_bytes=None,
            flow_bytes_per_s=to_float(row.get("Rate")),
            flow_pkts_per_s=to_float(row.get("Srate")),
            flow_iat_mean=to_float(row.get("IAT")),
            fwd_iat_mean=None,
            syn_flag_count=to_float(row.get("syn_count")),
            ack_flag_count=to_float(row.get("ack_count")),
            fin_flag_count=to_float(row.get("fin_count")),
            rst_flag_count=to_float(row.get("rst_count")),
            init_win_bytes_fwd=None,
            ttl_src=None,
            ttl_dst=None,
            header_length=to_float(row.get("Header_Length")),
            label_multiclass=label_str,
            label_binary=self._binary_from_benign_token(label_str, ["BenignTraffic", "Benign"]),
        )
