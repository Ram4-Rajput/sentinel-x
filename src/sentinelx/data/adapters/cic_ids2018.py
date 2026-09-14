"""CIC-IDS2018 adapter (genuine CICFlowMeter processed CSVs, 80 columns).

Audit facts honoured here:
  * NO Src/Dst IP in the ML CSVs -> src_ip/dst_ip are UNAVAILABLE (graph cannot
    be built from these files alone). Only Dst Port + Protocol exist.
  * Timestamp present as 'dd/MM/yyyy HH:mm:ss' -> parsed to datetime.
  * Flow Duration is in MICROSECONDS -> converted to seconds (documented).
  * Flow Byts/s and Flow Pkts/s can be Inf/NaN -> to_float() returns None so the
    preprocessor treats them as missing (we do NOT clip/impute in the adapter).
  * Label 'Benign' -> binary benign; anything else -> attack.

Every mapping decision is recorded in ``transform_notes``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from ..schema import BinaryLabel, FeatureAvailability, UnifiedFlowRecord
from .base import BaseDatasetAdapter, to_float, to_int

_TS_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %I:%M:%S %p")


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


class CICIDS2018Adapter(BaseDatasetAdapter):
    dataset_name = "cic-ids2018"
    required_columns = ("Dst Port", "Protocol", "Timestamp", "Flow Duration", "Label")

    def availability(self) -> FeatureAvailability:
        present = (
            "flow_duration", "total_fwd_packets", "total_bwd_packets",
            "total_fwd_bytes", "total_bwd_bytes", "flow_bytes_per_s",
            "flow_pkts_per_s", "flow_iat_mean", "fwd_iat_mean",
            "syn_flag_count", "ack_flag_count", "fin_flag_count",
            "rst_flag_count", "init_win_bytes_fwd", "header_length",
            "dst_port", "protocol",
        )
        unavailable = ("ttl_src", "ttl_dst", "src_port")
        derived = ()
        notes = {
            "flow_duration": "'Flow Duration' microseconds -> seconds (/1e6).",
            "flow_bytes_per_s": "'Flow Byts/s'; Inf/NaN -> None (missing), not clipped.",
            "flow_pkts_per_s": "'Flow Pkts/s'; Inf/NaN -> None (missing).",
            "header_length": "'Fwd Header Len' + 'Bwd Header Len' summed.",
            "src_ip": "UNAVAILABLE: IPs dropped from CICFlowMeter ML CSVs.",
            "dst_ip": "UNAVAILABLE: IPs dropped from CICFlowMeter ML CSVs.",
            "src_port": "UNAVAILABLE: not present in ML CSVs (only Dst Port).",
            "ttl_src": "UNAVAILABLE: TTL not exported by CICFlowMeter.",
            "ttl_dst": "UNAVAILABLE: TTL not exported by CICFlowMeter.",
            "timestamp": "'Timestamp' parsed dd/MM/yyyy HH:mm:ss.",
        }
        return FeatureAvailability(
            dataset=self.dataset_name, present=present, derived=derived,
            unavailable=unavailable, transform_notes=notes,
        )

    def map_row(self, row: Dict[str, object], index: int) -> UnifiedFlowRecord:
        dur_us = to_float(row.get("Flow Duration"))
        flow_duration = (dur_us / 1e6) if dur_us is not None else None

        fwd_hdr = to_float(row.get("Fwd Header Len"))
        bwd_hdr = to_float(row.get("Bwd Header Len"))
        header_length = None
        if fwd_hdr is not None or bwd_hdr is not None:
            header_length = (fwd_hdr or 0.0) + (bwd_hdr or 0.0)

        label = row.get("Label")
        label_str = None if label is None else str(label).strip()

        return UnifiedFlowRecord(
            dataset=self.dataset_name,
            source_index=index,
            src_ip=None,
            dst_ip=None,
            src_port=None,
            dst_port=to_int(row.get("Dst Port")),
            timestamp=_parse_ts(row.get("Timestamp")),
            protocol=(None if row.get("Protocol") in (None, "") else str(row.get("Protocol")).strip()),
            flow_duration=flow_duration,
            total_fwd_packets=to_float(row.get("Tot Fwd Pkts")),
            total_bwd_packets=to_float(row.get("Tot Bwd Pkts")),
            total_fwd_bytes=to_float(row.get("TotLen Fwd Pkts")),
            total_bwd_bytes=to_float(row.get("TotLen Bwd Pkts")),
            flow_bytes_per_s=to_float(row.get("Flow Byts/s")),
            flow_pkts_per_s=to_float(row.get("Flow Pkts/s")),
            flow_iat_mean=to_float(row.get("Flow IAT Mean")),
            fwd_iat_mean=to_float(row.get("Fwd IAT Mean")),
            syn_flag_count=to_float(row.get("SYN Flag Cnt")),
            ack_flag_count=to_float(row.get("ACK Flag Cnt")),
            fin_flag_count=to_float(row.get("FIN Flag Cnt")),
            rst_flag_count=to_float(row.get("RST Flag Cnt")),
            init_win_bytes_fwd=to_float(row.get("Init Fwd Win Byts")),
            ttl_src=None,
            ttl_dst=None,
            header_length=header_length,
            label_multiclass=label_str,
            label_binary=self._binary_from_benign_token(label_str, ["Benign"]),
        )
