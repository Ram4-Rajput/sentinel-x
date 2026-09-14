"""Adapter tests: schema validation, mapping correctness, availability honesty,
and the no-fabrication guard. Expected values are hand-derived from the tiny
fixtures in conftest.py."""

import math
from datetime import datetime

import pytest

from sentinelx.data.adapters import get_adapter, list_adapters
from sentinelx.data.adapters.base import SchemaValidationError
from sentinelx.data.adapters.unsw_nb15 import attach_raw_header, RAW_COLUMN_ORDER
from sentinelx.data.schema import BinaryLabel


def test_registry_lists_four_datasets():
    assert set(list_adapters()) == {"cic-ids2018", "ciciot2023", "ctu-13", "unsw-nb15"}


def test_get_adapter_case_insensitive():
    assert get_adapter("CIC-IDS2018").dataset_name == "cic-ids2018"


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        get_adapter("kddcup99")


# ----------------------------- CIC-IDS2018 -----------------------------
def test_cic_schema_validation_missing_column():
    ad = get_adapter("cic-ids2018")
    with pytest.raises(SchemaValidationError):
        ad.validate_schema(["Dst Port", "Protocol"])  # missing Timestamp/Label/...


def test_cic_mapping_units_and_labels(cic_rows):
    ad = get_adapter("cic-ids2018")
    recs = list(ad.iter_records(cic_rows, header=cic_rows[0].keys()))
    r0, r1 = recs
    # microseconds -> seconds
    assert r0.flow_duration == pytest.approx(112641719 / 1e6)
    assert r0.timestamp == datetime(2018, 2, 14, 8, 31, 1)
    assert r0.label_binary == BinaryLabel.BENIGN
    # header length = fwd+bwd
    assert r1.header_length == 60.0
    assert r1.label_binary == BinaryLabel.ATTACK
    assert r1.label_multiclass == "SSH-Bruteforce"
    # IPs are unavailable for CIC
    assert r0.src_ip is None and r0.dst_ip is None


def test_cic_inf_nan_become_none_not_clipped(cic_rows):
    ad = get_adapter("cic-ids2018")
    r1 = list(ad.iter_records(cic_rows))[1]
    # 'Infinity'/'NaN' -> None (missing), NOT a clipped finite value
    assert r1.flow_bytes_per_s is None
    assert r1.flow_pkts_per_s is None


def test_cic_no_fabrication_of_ttl(cic_rows):
    ad = get_adapter("cic-ids2018")
    for r in ad.iter_records(cic_rows):
        assert r.ttl_src is None and r.ttl_dst is None  # never invented


# ----------------------------- CICIoT2023 -----------------------------
def test_ciciot_no_timestamp_no_entities(ciciot_rows):
    ad = get_adapter("ciciot2023")
    recs = list(ad.iter_records(ciciot_rows, header=ciciot_rows[0].keys()))
    for r in recs:
        assert r.timestamp is None       # dataset has no time
        assert r.src_ip is None and r.dst_ip is None
        assert r.dst_port is None        # only protocol-presence flags
    assert recs[0].label_binary == BinaryLabel.BENIGN
    assert recs[1].label_binary == BinaryLabel.ATTACK
    assert recs[1].label_multiclass == "DDoS-SYN_Flood"


def test_ciciot_iat_maps_to_flow_iat_mean(ciciot_rows):
    ad = get_adapter("ciciot2023")
    r0 = list(ad.iter_records(ciciot_rows))[0]
    assert r0.flow_iat_mean == pytest.approx(83340575.0)
    assert r0.header_length == 757.0


# ----------------------------- CTU-13 -----------------------------
def test_ctu_entities_time_and_derived(ctu_rows):
    ad = get_adapter("ctu-13")
    recs = list(ad.iter_records(ctu_rows, header=ctu_rows[0].keys()))
    r0, r1, r2 = recs
    assert r0.src_ip == "147.32.84.59" and r0.dst_ip == "147.32.84.229"
    assert r0.timestamp == datetime(2011, 8, 10, 9, 46, 59, 607825)
    # derived bwd bytes = TotBytes - SrcBytes = 276-156 = 120
    assert r0.total_bwd_bytes == 120.0
    # derived rate = TotBytes/Dur = 276/1.026539
    assert r0.flow_bytes_per_s == pytest.approx(276 / 1.026539)
    # zero-duration flow -> rate None (no divide-by-zero)
    assert r2.flow_bytes_per_s is None
    # TotPkts is a TOTAL -> fwd packets left unavailable (not mislabeled)
    assert r0.total_fwd_packets is None


def test_ctu_freetext_label_parsing(ctu_rows):
    ad = get_adapter("ctu-13")
    r0, r1, r2 = list(ad.iter_records(ctu_rows))
    assert r0.label_binary == BinaryLabel.UNKNOWN   # Background
    assert r1.label_binary == BinaryLabel.ATTACK    # Botnet
    assert r2.label_binary == BinaryLabel.BENIGN    # Normal


def test_ctu_syn_flag_derived_from_state(ctu_rows):
    ad = get_adapter("ctu-13")
    r0, r1, r2 = list(ad.iter_records(ctu_rows))
    assert r0.syn_flag_count == 1.0   # 'S_RA' contains S
    assert r1.syn_flag_count == 0.0   # 'CON' has no S
    # ack/fin/rst remain unavailable (not fabricated)
    assert r0.ack_flag_count is None and r0.fin_flag_count is None


# ----------------------------- UNSW-NB15 -----------------------------
def test_unsw_attach_raw_header_length_guard():
    with pytest.raises(ValueError):
        attach_raw_header(["only", "three", "values"])


def test_unsw_raw_row_ttl_and_entities(unsw_raw_values):
    ad = get_adapter("unsw-nb15")
    row = attach_raw_header(unsw_raw_values)
    assert len(RAW_COLUMN_ORDER) == 49
    rec = ad.map_row(row, 0)
    assert rec.src_ip == "59.166.0.0" and rec.dst_ip == "149.171.126.6"
    assert rec.ttl_src == 31.0 and rec.ttl_dst == 29.0   # UNIQUE TTL feature
    assert rec.init_win_bytes_fwd == 255.0               # swin
    assert rec.timestamp is not None                     # Stime epoch parsed
    # empty attack_cat + Label 0 -> benign
    assert rec.label_binary == BinaryLabel.BENIGN


def test_unsw_partition_rows_labeling(unsw_partition_rows):
    ad = get_adapter("unsw-nb15")
    r0, r1 = [ad.map_row(r, i) for i, r in enumerate(unsw_partition_rows)]
    assert r0.label_binary == BinaryLabel.BENIGN and r0.label_multiclass == "Normal"
    assert r1.label_binary == BinaryLabel.ATTACK and r1.label_multiclass == "Exploits"
    # partition rows have no srcip -> entities None, but ttl present
    assert r0.src_ip is None
    assert r0.ttl_src == 254.0


def test_unsw_flags_unavailable_not_fabricated(unsw_partition_rows):
    ad = get_adapter("unsw-nb15")
    rec = ad.map_row(unsw_partition_rows[0], 0)
    assert rec.syn_flag_count is None and rec.ack_flag_count is None


# ----------------------------- availability contracts -----------------------------
@pytest.mark.parametrize("name", ["cic-ids2018", "ciciot2023", "ctu-13", "unsw-nb15"])
def test_all_availability_contracts_consistent(name):
    ad = get_adapter(name)
    ad.get_availability().assert_consistent()  # raises if malformed


@pytest.mark.parametrize("name", ["cic-ids2018", "ciciot2023"])
def test_ttl_declared_unavailable_for_non_unsw(name):
    ad = get_adapter(name)
    assert "ttl_src" in ad.get_availability().unavailable
