"""CTU-13 adapter (bidirectional NetFlow .binetflow, 15 columns).

Audit facts honoured here:
  * FULL entities: SrcAddr/DstAddr/Sport/Dport -> graph-constructable.
  * StartTime present ('YYYY/MM/DD HH:MM:SS.ffffff') -> parsed to datetime.
  * total_fwd_bytes <- SrcBytes (direct); total_bwd_bytes <- (TotBytes - SrcBytes)
    DERIVED (documented). total_fwd_packets <- TotPkts is a TOTAL, not a fwd
    split -> we DO NOT claim it as fwd; instead we leave fwd/bwd packet splits
    UNAVAILABLE and expose the total via flow-level bytes only. (Conservative:
    never mislabel a total as a directional count.)
  * flow_bytes_per_s / flow_pkts_per_s DERIVED as TotBytes/Dur and TotPkts/Dur
    (guarded against divide-by-zero -> None).
  * Label is a free-text 'flow=...' string -> parsed to Background/Normal/Botnet;
    Botnet -> attack, Normal -> benign, Background -> UNKNOWN (audit: Background
    is NOT confirmed benign).
  * TCP flags are only in the 'State' field (e.g. S_RA); we parse SYN presence
    into a 0/1 syn_flag_count and leave ack/fin/rst UNAVAILABLE (State is not a
    reliable per-flag count). syn_flag_count is therefore DERIVED.

TTL / window / header_length are absent and NOT fabricated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from ..schema import BinaryLabel, FeatureAvailability, UnifiedFlowRecord
from .base import BaseDatasetAdapter, to_float, to_int

_TS_FORMATS = ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S")


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


def _parse_label(raw: Optional[str]):
    """Return (multiclass, binary) from CTU-13 free-text flow label."""
    if raw is None:
        return None, BinaryLabel.UNKNOWN
    s = str(raw).strip()
    low = s.lower()
    if "botnet" in low:
        return s, BinaryLabel.ATTACK
    if "normal" in low:
        return s, BinaryLabel.BENIGN
    if "background" in low:
        # Audit: Background is 'unknown', not confirmed benign.
        return s, BinaryLabel.UNKNOWN
    return s, BinaryLabel.UNKNOWN


class CTU13Adapter(BaseDatasetAdapter):
    dataset_name = "ctu-13"
    required_columns = (
        "StartTime", "Dur", "Proto", "SrcAddr", "Sport",
        "DstAddr", "Dport", "State", "TotPkts", "TotBytes", "SrcBytes", "Label",
    )

    def availability(self) -> FeatureAvailability:
        present = ("flow_duration", "total_fwd_bytes", "src_port", "dst_port", "protocol")
        derived = (
            "total_bwd_bytes",   # TotBytes - SrcBytes
            "flow_bytes_per_s",  # TotBytes / Dur
            "flow_pkts_per_s",   # TotPkts / Dur
            "syn_flag_count",    # parsed from State
        )
        unavailable = (
            "total_fwd_packets",  # TotPkts is a TOTAL, not a fwd split
            "total_bwd_packets",
            "flow_iat_mean", "fwd_iat_mean",
            "ack_flag_count", "fin_flag_count", "rst_flag_count",
            "init_win_bytes_fwd", "ttl_src", "ttl_dst", "header_length",
        )
        notes = {
            "flow_duration": "'Dur' direct (seconds).",
            "total_fwd_bytes": "'SrcBytes' direct.",
            "total_bwd_bytes": "DERIVED = TotBytes - SrcBytes (>=0 clamp).",
            "flow_bytes_per_s": "DERIVED = TotBytes / Dur (Dur<=0 -> None).",
            "flow_pkts_per_s": "DERIVED = TotPkts / Dur (Dur<=0 -> None).",
            "syn_flag_count": "DERIVED: 1.0 if 'S' in State src/dst flags else 0.0.",
            "total_fwd_packets": "UNAVAILABLE: TotPkts is a total, not a fwd split (not mislabeled).",
            "ack_flag_count": "UNAVAILABLE: State is not a reliable per-flag count.",
            "fin_flag_count": "UNAVAILABLE.",
            "rst_flag_count": "UNAVAILABLE.",
            "ttl_src": "UNAVAILABLE.",
            "ttl_dst": "UNAVAILABLE.",
            "header_length": "UNAVAILABLE.",
            "src_ip": "'SrcAddr' direct (entity).",
            "dst_ip": "'DstAddr' direct (entity).",
            "timestamp": "'StartTime' parsed YYYY/MM/DD HH:MM:SS.ffffff.",
        }
        return FeatureAvailability(
            dataset=self.dataset_name, present=present, derived=derived,
            unavailable=unavailable, transform_notes=notes,
        )

    def map_row(self, row: Dict[str, object], index: int) -> UnifiedFlowRecord:
        dur = to_float(row.get("Dur"))
        tot_bytes = to_float(row.get("TotBytes"))
        src_bytes = to_float(row.get("SrcBytes"))
        tot_pkts = to_float(row.get("TotPkts"))

        # derived bwd bytes (clamp >= 0)
        total_bwd_bytes = None
        if tot_bytes is not None and src_bytes is not None:
            total_bwd_bytes = max(0.0, tot_bytes - src_bytes)

        # derived rates (guard divide-by-zero)
        flow_bps = (tot_bytes / dur) if (tot_bytes is not None and dur and dur > 0) else None
        flow_pps = (tot_pkts / dur) if (tot_pkts is not None and dur and dur > 0) else None

        # derived syn flag from State
        state = row.get("State")
        syn = None
        if state is not None:
            syn = 1.0 if "S" in str(state).upper() else 0.0

        multiclass, binary = _parse_label(row.get("Label"))

        src_ip = row.get("SrcAddr")
        dst_ip = row.get("DstAddr")
        return UnifiedFlowRecord(
            dataset=self.dataset_name,
            source_index=index,
            src_ip=None if src_ip in (None, "") else str(src_ip).strip(),
            dst_ip=None if dst_ip in (None, "") else str(dst_ip).strip(),
            src_port=to_int(row.get("Sport")),
            dst_port=to_int(row.get("Dport")),
            timestamp=_parse_ts(row.get("StartTime")),
            protocol=(None if row.get("Proto") in (None, "") else str(row.get("Proto")).strip()),
            flow_duration=dur,
            total_fwd_packets=None,
            total_bwd_packets=None,
            total_fwd_bytes=src_bytes,
            total_bwd_bytes=total_bwd_bytes,
            flow_bytes_per_s=flow_bps,
            flow_pkts_per_s=flow_pps,
            flow_iat_mean=None,
            fwd_iat_mean=None,
            syn_flag_count=syn,
            ack_flag_count=None,
            fin_flag_count=None,
            rst_flag_count=None,
            init_win_bytes_fwd=None,
            ttl_src=None,
            ttl_dst=None,
            header_length=None,
            label_multiclass=multiclass,
            label_binary=binary,
        )
