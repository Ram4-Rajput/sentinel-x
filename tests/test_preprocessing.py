"""Preprocessing tests: train-only fit, imputation, scaling, categorical
encoding, and leakage guards. Expected values hand-derived."""

import pytest

from sentinelx.data.preprocessing import FlowPreprocessor
from sentinelx.data.schema import UnifiedFlowRecord


def _rec(**kw):
    return UnifiedFlowRecord(dataset="x", source_index=0, **kw)


def test_transform_before_fit_raises():
    pp = FlowPreprocessor()
    with pytest.raises(RuntimeError):
        pp.transform([_rec(flow_duration=1.0)])


def test_median_imputation_uses_train_only():
    # train medians: flow_duration median of [1,3] = 2.0
    train = [_rec(flow_duration=1.0), _rec(flow_duration=3.0)]
    pp = FlowPreprocessor(add_missing_indicators=True).fit(train)
    # a test record missing flow_duration should be imputed with TRAIN median 2.0
    # (before scaling). Verify via the missing indicator + that no crash occurs.
    names = pp.feature_names()
    fd_idx = names.index("flow_duration")
    miss_idx = names.index("flow_duration_was_missing")
    out = pp.transform_record(_rec(flow_duration=None))[0:]
    # imputed value 2.0 equals the train mean 2.0 -> standardized to 0.0
    assert out[fd_idx] == pytest.approx(0.0)
    assert out[miss_idx] == 1.0


def test_standardization_uses_train_stats():
    train = [_rec(flow_duration=0.0), _rec(flow_duration=10.0)]
    pp = FlowPreprocessor(add_missing_indicators=False).fit(train)
    names = pp.feature_names()
    fd_idx = names.index("flow_duration")
    # train mean=5, pstdev=5 -> value 10 => (10-5)/5 = 1.0
    out = pp.transform_record(_rec(flow_duration=10.0))
    assert out[fd_idx] == pytest.approx(1.0)


def test_zero_std_does_not_divide_by_zero():
    train = [_rec(flow_duration=5.0), _rec(flow_duration=5.0)]
    pp = FlowPreprocessor(add_missing_indicators=False).fit(train)
    names = pp.feature_names()
    fd_idx = names.index("flow_duration")
    out = pp.transform_record(_rec(flow_duration=5.0))
    assert out[fd_idx] == pytest.approx(0.0)  # (5-5)/1.0 guard


def test_categorical_unseen_maps_to_unk():
    train = [_rec(protocol="tcp"), _rec(protocol="udp")]
    pp = FlowPreprocessor(add_missing_indicators=False).fit(train)
    names = pp.feature_names()
    proto_idx = names.index("protocol_idx")
    vocab = pp.vocabulary("protocol")
    # unseen 'icmp' -> <unk> index 0
    out_unseen = pp.transform_record(_rec(protocol="icmp"))
    assert out_unseen[proto_idx] == 0.0
    # known 'tcp' -> its vocab index
    out_tcp = pp.transform_record(_rec(protocol="tcp"))
    assert out_tcp[proto_idx] == float(vocab["tcp"])


def test_feature_names_length_matches_vector():
    train = [_rec(flow_duration=1.0, protocol="tcp")]
    pp = FlowPreprocessor(add_missing_indicators=True).fit(train)
    vec = pp.transform_record(_rec(flow_duration=2.0, protocol="tcp"))
    assert len(vec) == len(pp.feature_names())


def test_fit_requires_records():
    with pytest.raises(ValueError):
        FlowPreprocessor().fit([])
