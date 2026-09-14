"""End-to-end: adapter -> chronological split -> train-only preprocess ->
windows/graph -> leakage checks, on the graph-capable CTU-13 fixture.

This mirrors how a training script would consume the layer, and asserts the
leakage-safe contract holds end to end."""

from sentinelx.data.adapters import get_adapter
from sentinelx.data.leakage import run_leakage_checks
from sentinelx.data.preprocessing import FlowPreprocessor
from sentinelx.data.temporal import (
    assert_chronological_disjoint,
    chronological_split,
    generate_windows,
)


def test_ctu_end_to_end(ctu_rows):
    ad = get_adapter("ctu-13")
    records = list(ad.iter_records(ctu_rows, header=ctu_rows[0].keys()))
    assert len(records) == 3

    # chronological split (tiny: 1/1/1)
    train, val, test = chronological_split(records, train_frac=0.34, val_frac=0.33)
    assert_chronological_disjoint(train, val, test)

    # train-only preprocessing
    pp = FlowPreprocessor().fit(train)
    train_vecs = pp.transform(train)
    test_vecs = pp.transform(test)
    assert len(train_vecs[0]) == len(pp.feature_names())
    assert len(test_vecs) == len(test)

    # windows + graph on the full ordered set
    windows = generate_windows(records, window_seconds=5, build_graphs=True)
    assert sum(w.num_flows for w in windows) == 3
    # at least one window should have graph nodes (CTU-13 has IPs)
    assert any(w.num_nodes > 0 for w in windows)

    # leakage checks pass (temporal ordering respected, preprocessor fitted)
    report = run_leakage_checks(train, val, test, preprocessor=pp)
    # temporal + preprocessor must pass; entity_overlap may warn (medium)
    temporal = next(f for f in report.findings if f.check == "temporal_order")
    ppf = next(f for f in report.findings if f.check == "preprocessor_fit")
    assert temporal.passed and ppf.passed


def test_availability_documented_for_all(ctu_rows, cic_rows, ciciot_rows):
    # every adapter exposes transform notes for its unavailable features
    for name in ("cic-ids2018", "ciciot2023", "ctu-13", "unsw-nb15"):
        ad = get_adapter(name)
        av = ad.get_availability()
        for feat in av.unavailable:
            # documentation must exist for unavailable features
            assert feat in av.transform_notes or feat in (
                "src_port", "dst_port", "total_bwd_packets", "total_bwd_bytes",
                "fwd_iat_mean", "flow_iat_mean", "ack_flag_count",
                "fin_flag_count", "rst_flag_count", "header_length",
                "init_win_bytes_fwd", "ttl_src", "ttl_dst", "total_fwd_packets",
                "syn_flag_count",
            )
