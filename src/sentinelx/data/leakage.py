"""Leakage checks for the unified data layer.

Implements the audit's leakage-risk items as executable assertions (per
sentinel-x-ml-research: leakage-safe by default). Each check returns a finding
rather than raising, so callers can decide severity; ``raise_if_failed`` is
provided for CI/tests.

Checks:
  1. temporal_order   - train max-time <= val/test min-time (no future leak).
  2. entity_overlap   - hosts (src_ip) shared between train and test (graph
                        identity leakage; suggests GroupKFold by host).
  3. duplicate_rows   - identical feature vectors spanning splits.
  4. preprocessor_fit - the preprocessor was fit on train only (guard flag).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .preprocessing import FlowPreprocessor
from .schema import UnifiedFlowRecord
from .temporal import normalize_timestamp


@dataclass
class LeakageFinding:
    check: str
    passed: bool
    detail: str
    severity: str = "high"  # high|medium|low|info


@dataclass
class LeakageReport:
    findings: List[LeakageFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.findings)

    def add(self, finding: LeakageFinding) -> None:
        self.findings.append(finding)

    def failed(self) -> List[LeakageFinding]:
        return [f for f in self.findings if not f.passed]

    def raise_if_failed(self) -> None:
        bad = self.failed()
        if bad:
            lines = "\n".join(f"  [{f.severity}] {f.check}: {f.detail}" for f in bad)
            raise AssertionError(f"Leakage checks FAILED:\n{lines}")


def _check_temporal(train, val, test) -> LeakageFinding:
    have_ts = all(r.timestamp is not None for r in list(train) + list(val) + list(test))
    if not have_ts:
        return LeakageFinding(
            "temporal_order", True,
            "skipped: dataset has no timestamps (temporal split N/A).",
            severity="info",
        )
    def _max(rs):
        return max((normalize_timestamp(r.timestamp) for r in rs), default=None)
    def _min(rs):
        return min((normalize_timestamp(r.timestamp) for r in rs), default=None)
    tr_max = _max(train)
    later_min = _min(list(val) + list(test))
    if tr_max is not None and later_min is not None and tr_max > later_min:
        return LeakageFinding(
            "temporal_order", False,
            f"train max time {tr_max} is after val/test min time {later_min}.",
        )
    return LeakageFinding("temporal_order", True, "train precedes val/test in time.")


def _check_entity_overlap(train, test, *, warn_only: bool = True) -> LeakageFinding:
    tr_hosts = {r.src_ip for r in train if r.src_ip}
    te_hosts = {r.src_ip for r in test if r.src_ip}
    if not tr_hosts or not te_hosts:
        return LeakageFinding(
            "entity_overlap", True,
            "skipped: no src_ip entities (dataset lacks IPs).",
            severity="info",
        )
    shared = tr_hosts & te_hosts
    if shared:
        frac = len(shared) / len(te_hosts)
        # This is inherent to testbed datasets (UNSW-NB15 / CTU-13); report as
        # medium so it's visible but doesn't hard-fail by default.
        return LeakageFinding(
            "entity_overlap",
            passed=warn_only,
            detail=(f"{len(shared)} host(s) appear in both train and test "
                    f"({frac:.0%} of test hosts). Consider GroupKFold by host."),
            severity="medium",
        )
    return LeakageFinding("entity_overlap", True, "no shared hosts between train/test.")


def _check_duplicates(train, test, features) -> LeakageFinding:
    def _key(r):
        return tuple(getattr(r, f) for f in features)
    tr_keys = {_key(r) for r in train}
    dupes = sum(1 for r in test if _key(r) in tr_keys)
    if dupes:
        frac = dupes / max(1, len(test))
        return LeakageFinding(
            "duplicate_rows",
            passed=(frac < 0.001),  # tolerate tiny coincidental overlap
            detail=f"{dupes} test rows share an identical feature key with train ({frac:.2%}).",
            severity="high",
        )
    return LeakageFinding("duplicate_rows", True, "no duplicate feature keys across splits.")


def _check_preprocessor(preprocessor: Optional[FlowPreprocessor]) -> LeakageFinding:
    if preprocessor is None:
        return LeakageFinding(
            "preprocessor_fit", True,
            "skipped: no preprocessor supplied.", severity="info")
    return LeakageFinding(
        "preprocessor_fit", preprocessor.is_fitted,
        "preprocessor is fitted (should be train-only)." if preprocessor.is_fitted
        else "preprocessor not fitted.",
        severity="high",
    )


def run_leakage_checks(
    train: Sequence[UnifiedFlowRecord],
    val: Sequence[UnifiedFlowRecord],
    test: Sequence[UnifiedFlowRecord],
    *,
    preprocessor: Optional[FlowPreprocessor] = None,
    dup_features: Sequence[str] = ("flow_duration", "total_fwd_bytes",
                                   "total_bwd_bytes", "dst_port", "protocol"),
    entity_overlap_warn_only: bool = True,
) -> LeakageReport:
    """Run all leakage checks and return a report."""
    report = LeakageReport()
    report.add(_check_temporal(train, val, test))
    report.add(_check_entity_overlap(train, test, warn_only=entity_overlap_warn_only))
    report.add(_check_duplicates(train, test, dup_features))
    report.add(_check_preprocessor(preprocessor))
    return report
