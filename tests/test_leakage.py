"""Leakage-check tests: temporal order, entity overlap, duplicate rows,
preprocessor-fit guard."""

from datetime import datetime

import pytest

from sentinelx.data.leakage import run_leakage_checks
from sentinelx.data.preprocessing import FlowPreprocessor
from sentinelx.data.schema import UnifiedFlowRecord


def _rec(i, minute, src=None, fd=1.0, proto="tcp"):
    return UnifiedFlowRecord(
        dataset="x", source_index=i,
        timestamp=datetime(2020, 1, 1, 0, minute % 60, minute // 60),
        src_ip=src, flow_duration=fd, total_fwd_bytes=fd, total_bwd_bytes=0.0,
        dst_port=80, protocol=proto,
    )


def test_temporal_order_pass():
    train = [_rec(0, 0), _rec(1, 1)]
    val = [_rec(2, 5)]
    test = [_rec(3, 9)]
    report = run_leakage_checks(train, val, test)
    finding = next(f for f in report.findings if f.check == "temporal_order")
    assert finding.passed


def test_temporal_order_detects_future_in_train():
    train = [_rec(0, 100)]   # train is AFTER test
    val = []
    test = [_rec(1, 1)]
    report = run_leakage_checks(train, val, test)
    finding = next(f for f in report.findings if f.check == "temporal_order")
    assert not finding.passed
    assert not report.passed


def test_duplicate_rows_flagged():
    # identical feature key present in both train and test
    dup = _rec(0, 0, fd=42.0)
    train = [dup]
    test = [_rec(1, 9, fd=42.0)]  # same fd/bytes/port/proto -> same key
    report = run_leakage_checks(train, [], test)
    finding = next(f for f in report.findings if f.check == "duplicate_rows")
    assert not finding.passed


def test_entity_overlap_is_warning_by_default():
    train = [_rec(0, 0, src="10.0.0.1")]
    test = [_rec(1, 9, src="10.0.0.1")]  # shared host
    report = run_leakage_checks(train, [], test)
    finding = next(f for f in report.findings if f.check == "entity_overlap")
    assert finding.severity == "medium"
    assert finding.passed  # warn-only by default (still surfaced in detail)
    assert "GroupKFold" in finding.detail


def test_entity_overlap_can_hard_fail():
    train = [_rec(0, 0, src="10.0.0.1")]
    test = [_rec(1, 9, src="10.0.0.1")]
    report = run_leakage_checks(train, [], test, entity_overlap_warn_only=False)
    finding = next(f for f in report.findings if f.check == "entity_overlap")
    assert not finding.passed


def test_preprocessor_fit_guard():
    train = [_rec(0, 0)]
    test = [_rec(1, 9)]
    pp = FlowPreprocessor()  # NOT fitted
    report = run_leakage_checks(train, [], test, preprocessor=pp)
    finding = next(f for f in report.findings if f.check == "preprocessor_fit")
    assert not finding.passed
    pp.fit(train)
    report2 = run_leakage_checks(train, [], test, preprocessor=pp)
    finding2 = next(f for f in report2.findings if f.check == "preprocessor_fit")
    assert finding2.passed


def test_raise_if_failed():
    train = [_rec(0, 100)]
    test = [_rec(1, 1)]
    report = run_leakage_checks(train, [], test)
    with pytest.raises(AssertionError):
        report.raise_if_failed()


def test_ciciot_no_timestamp_temporal_check_skipped():
    # records without timestamps -> temporal check reports 'info', passes
    recs = [UnifiedFlowRecord(dataset="ciciot2023", source_index=i) for i in range(3)]
    report = run_leakage_checks(recs[:1], recs[1:2], recs[2:])
    finding = next(f for f in report.findings if f.check == "temporal_order")
    assert finding.severity == "info" and finding.passed
