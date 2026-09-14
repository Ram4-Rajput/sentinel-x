"""UNSW-NB15 adapter (49-feature Argus/Bro flow schema).

Audit facts honoured here:
  * FULL entities: srcip/sport/dstip/dsport -> graph-constructable.
  * The ONLY local source with TTL (sttl/dttl), TCP window (swin/dwin), and
    retransmission (sloss/dloss). These map to ttl_src/ttl_dst/init_win_bytes_fwd.
  * Timestamps Stime/Ltime are epoch seconds -> converted to datetime.
  * Raw files UNSW-NB15_1..4.csv are HEADER-LESS: this adapter accepts rows as
    dicts, so the caller MUST attach the 49 canonical names (see
    ``RAW_COLUMN_ORDER``) before feeding rows. The partition CSVs
    (training-set/testing-set) already have headers but use slightly different
    names (lowercase, add 'id'/'service'); this adapter reads via a tolerant
    key lookup that accepts either naming.
  * attack_cat -> multiclass; Label 0/1 (and 'Normal' attack_cat) -> binary.

No feature is fabricated; flags not present as counts stay UNAVAILABLE.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from ..schema import BinaryLabel, FeatureAvailability, UnifiedFlowRecord
from .base import BaseDatasetAdapter, to_float, to_int

# Canonical column order of the header-less raw UNSW-NB15_1..4.csv files
# (from NUSW-NB15_features.csv, 49 fields).
RAW_COLUMN_ORDER = (
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label",
)


def attach_raw_header(row_values):
    """Zip a header-less raw row (sequence of 49 values) with canonical names.
    Helper for callers reading UNSW-NB15_1..4.csv."""
    values = list(row_values)
    if len(values) != len(RAW_COLUMN_ORDER):
        raise ValueError(
            f"UNSW-NB15 raw row has {len(values)} fields, expected "
            f"{len(RAW_COLUMN_ORDER)}. Header-less file may be misaligned."
        )
    return dict(zip(RAW_COLUMN_ORDER, values))


def _get(row: Dict[str, object], *names):
    """Tolerant lookup: try several possible column names (raw vs partition)."""
    for n in names:
        if n in row:
            return row[n]
    # case-insensitive fallback
    lower = {str(k).lower(): v for k, v in row.items()}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _parse_epoch(value) -> Optional[datetime]:
    f = to_float(value)
    if f is None:
        return None
    try:
        return datetime.fromtimestamp(f, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


class UNSWNB15Adapter(BaseDatasetAdapter):
    dataset_name = "unsw-nb15"
    # Only the fields we truly rely on are required; partition CSVs may lack
    # srcip/Stime, so we require the always-present core + label.
    required_columns = ("proto", "dur", "sbytes", "Label")
    # Partition CSVs use lowercase 'label' etc; raw uses 'Label'. Tolerate both.
    case_insensitive_schema = True

    def availability(self) -> FeatureAvailability:
        present = (
            "flow_duration", "total_fwd_packets", "total_bwd_packets",
            "total_fwd_bytes", "total_bwd_bytes", "flow_bytes_per_s",
            "flow_pkts_per_s", "fwd_iat_mean", "init_win_bytes_fwd",
            "ttl_src", "ttl_dst", "src_port", "dst_port", "protocol",
        )
        derived = ()
        unavailable = (
            "flow_iat_mean",  # only per-direction Sintpkt/Dintpkt exist
            "syn_flag_count", "ack_flag_count", "fin_flag_count",
            "rst_flag_count", "header_length",
        )
        notes = {
            "flow_duration": "'dur' direct (seconds).",
            "total_fwd_packets": "'Spkts' direct.",
            "total_bwd_packets": "'Dpkts' direct.",
            "total_fwd_bytes": "'sbytes' direct.",
            "total_bwd_bytes": "'dbytes' direct.",
            "flow_bytes_per_s": "'Sload' (source bits/s) as bytes/s proxy.",
            "flow_pkts_per_s": "derived-in-adapter n/a; 'Sintpkt' -> fwd_iat_mean; pkts/s left as Sload-based proxy None if absent.",
            "fwd_iat_mean": "'Sintpkt' source interpacket time (ms).",
            "init_win_bytes_fwd": "'swin' source TCP window advertisement.",
            "ttl_src": "'sttl' source TTL (UNIQUE to UNSW-NB15).",
            "ttl_dst": "'dttl' destination TTL.",
            "protocol": "'proto' string.",
            "src_ip": "'srcip' (raw files); may be absent in partition CSVs.",
            "dst_ip": "'dstip' (raw files); may be absent in partition CSVs.",
            "flow_iat_mean": "UNAVAILABLE as a single flow IAT (only Sintpkt/Dintpkt per-direction).",
            "syn_flag_count": "UNAVAILABLE: no per-flag counts (only 'state').",
            "ack_flag_count": "UNAVAILABLE.",
            "fin_flag_count": "UNAVAILABLE.",
            "rst_flag_count": "UNAVAILABLE.",
            "header_length": "UNAVAILABLE.",
            "label": "attack_cat -> multiclass; Label 0/1 (attack_cat Normal/empty -> benign).",
        }
        return FeatureAvailability(
            dataset=self.dataset_name, present=present, derived=derived,
            unavailable=unavailable, transform_notes=notes,
        )

    def map_row(self, row: Dict[str, object], index: int) -> UnifiedFlowRecord:
        src_ip = _get(row, "srcip")
        dst_ip = _get(row, "dstip")

        # Label: prefer explicit binary Label; classify via attack_cat.
        attack_cat = _get(row, "attack_cat")
        cat_str = None if attack_cat in (None, "") else str(attack_cat).strip()
        label_bin_raw = to_int(_get(row, "Label", "label"))
        if label_bin_raw is not None:
            binary = BinaryLabel.ATTACK if label_bin_raw == 1 else BinaryLabel.BENIGN
        else:
            # fall back to attack_cat text
            if cat_str is None or cat_str.lower() in {"normal", ""}:
                binary = BinaryLabel.BENIGN
            else:
                binary = BinaryLabel.ATTACK
        # normalise multiclass: empty/Normal -> "Normal"
        multiclass = cat_str if cat_str else ("Normal" if binary == BinaryLabel.BENIGN else None)

        # source bits/s -> approximate bytes/s (/8). Documented proxy.
        sload = to_float(_get(row, "Sload"))
        flow_bps = (sload / 8.0) if sload is not None else None

        return UnifiedFlowRecord(
            dataset=self.dataset_name,
            source_index=index,
            src_ip=None if src_ip in (None, "") else str(src_ip).strip(),
            dst_ip=None if dst_ip in (None, "") else str(dst_ip).strip(),
            src_port=to_int(_get(row, "sport")),
            dst_port=to_int(_get(row, "dsport")),
            timestamp=_parse_epoch(_get(row, "Stime")),
            protocol=(None if _get(row, "proto") in (None, "") else str(_get(row, "proto")).strip()),
            flow_duration=to_float(_get(row, "dur")),
            total_fwd_packets=to_float(_get(row, "Spkts", "spkts")),
            total_bwd_packets=to_float(_get(row, "Dpkts", "dpkts")),
            total_fwd_bytes=to_float(_get(row, "sbytes")),
            total_bwd_bytes=to_float(_get(row, "dbytes")),
            flow_bytes_per_s=flow_bps,
            flow_pkts_per_s=None,
            flow_iat_mean=None,
            fwd_iat_mean=to_float(_get(row, "Sintpkt", "sinpkt")),
            syn_flag_count=None,
            ack_flag_count=None,
            fin_flag_count=None,
            rst_flag_count=None,
            init_win_bytes_fwd=to_float(_get(row, "swin")),
            ttl_src=to_float(_get(row, "sttl")),
            ttl_dst=to_float(_get(row, "dttl")),
            header_length=None,
            label_multiclass=multiclass,
            label_binary=binary,
        )
