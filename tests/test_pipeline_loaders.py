"""Phase-2 loader tests: chunked streaming, malformed rows, timestamp parsing,
adapter routing, header-less handling, max_records cap. Uses tiny synthetic CSV
files in a temp dir (no dependence on the large real datasets)."""

import csv
from pathlib import Path

import pytest

from sentinelx.pipeline.loaders import (
    LoaderStats,
    iter_raw_chunks,
    iter_unified_records,
)
from sentinelx.data.adapters.unsw_nb15 import RAW_COLUMN_ORDER


CIC_HEADER = ["Dst Port", "Protocol", "Timestamp", "Flow Duration",
              "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
              "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Fwd IAT Mean",
              "SYN Flag Cnt", "ACK Flag Cnt", "FIN Flag Cnt", "RST Flag Cnt",
              "Init Fwd Win Byts", "Fwd Header Len", "Bwd Header Len", "Label"]


def _write_cic_csv(path: Path, n_rows: int, with_malformed: bool = False):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CIC_HEADER)
        for i in range(n_rows):
            sec = 1 + (i % 50)
            w.writerow([80, 6, f"14/02/2018 08:31:{sec:02d}", 1000000,
                        3, 1, 100, 50, 0.5, 0.1, 1000, 900,
                        0, 1, 0, 0, 8192, 40, 20, "Benign"])
        if with_malformed:
            # too few fields -> DictReader tolerates (missing -> None); write a
            # row with EXTRA fields to trigger the None-overflow malformed path
            fh.write(",".join(["x"] * (len(CIC_HEADER) + 3)) + "\n")


def test_chunked_reading_yields_expected_chunks(tmp_path):
    p = tmp_path / "cic.csv"
    _write_cic_csv(p, 250)
    stats = LoaderStats()
    chunks = list(iter_raw_chunks([p], chunk_size=100, stats=stats))
    # 250 rows / 100 -> chunks of 100, 100, 50
    assert [len(c) for c in chunks] == [100, 100, 50]
    assert stats.rows_read == 250


def test_iter_unified_records_routes_to_adapter_and_parses_ts(tmp_path):
    p = tmp_path / "cic.csv"
    _write_cic_csv(p, 10)
    recs = list(iter_unified_records("cic-ids2018", [p], chunk_size=4))
    assert len(recs) == 10
    assert recs[0].dataset == "cic-ids2018"
    assert recs[0].timestamp is not None          # timestamp parsed
    assert recs[0].flow_duration == pytest.approx(1.0)  # 1e6 us -> 1s
    assert recs[0].label_binary.value == "benign"


def test_malformed_rows_counted_not_fatal(tmp_path):
    p = tmp_path / "cic_bad.csv"
    _write_cic_csv(p, 5, with_malformed=True)
    stats = LoaderStats()
    recs = list(iter_unified_records("cic-ids2018", [p], chunk_size=100, stats=stats))
    # 5 good rows mapped; the overflow row is flagged malformed and skipped
    assert len(recs) == 5
    assert stats.malformed_rows >= 1


def test_max_records_cap(tmp_path):
    p = tmp_path / "cic.csv"
    _write_cic_csv(p, 500)
    recs = list(iter_unified_records("cic-ids2018", [p], chunk_size=50, max_records=120))
    assert len(recs) == 120


def test_schema_validation_rejects_wrong_header(tmp_path):
    from sentinelx.data.adapters.base import SchemaValidationError
    p = tmp_path / "wrong.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["a", "b", "c"])
        csv.writer(fh).writerow([1, 2, 3])
    with pytest.raises(SchemaValidationError):
        list(iter_unified_records("cic-ids2018", [p]))


def test_headerless_unsw_raw_attaches_columns(tmp_path):
    p = tmp_path / "UNSW-NB15_1.csv"
    row = ["59.166.0.0", "1390", "149.171.126.6", "53", "udp", "CON", "0.001",
           "132", "164", "31", "29", "0", "0", "dns", "500", "600",
           "2", "2", "255", "255", "0", "0", "66", "82", "0", "0", "0.01", "0.01",
           "1421927414", "1421927414", "0.5", "0.4", "0", "0", "0", "0", "3", "0",
           "0", "0", "3", "7", "1", "3", "1", "1", "1", "", "0"]
    assert len(row) == len(RAW_COLUMN_ORDER)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(row)
    recs = list(iter_unified_records("unsw-nb15", [p], headerless=True))
    assert len(recs) == 1
    assert recs[0].src_ip == "59.166.0.0"
    assert recs[0].ttl_src == 31.0


def test_multiple_files_accumulate(tmp_path):
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_cic_csv(p1, 30)
    _write_cic_csv(p2, 20)
    stats = LoaderStats()
    recs = list(iter_unified_records("cic-ids2018", [p1, p2], chunk_size=100, stats=stats))
    assert len(recs) == 50
    assert stats.files_processed == 2
