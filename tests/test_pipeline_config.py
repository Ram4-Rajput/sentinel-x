"""Phase-2 config tests: path resolution order, file filtering, PCAP optional."""

import os
from pathlib import Path

import pytest

from sentinelx.pipeline.config import resolve_dataset, PipelineConfig
from sentinelx.pipeline import pcap


def test_explicit_raw_path_wins(tmp_path):
    (tmp_path / "a_TrafficForML_CICFlowMeter.csv").write_text("x")
    dp = resolve_dataset("cic-ids2018", raw_path=str(tmp_path))
    assert dp.raw_path == tmp_path
    assert len(dp.list_files()) == 1


def test_env_var_resolution(tmp_path, monkeypatch):
    (tmp_path / "b_TrafficForML_CICFlowMeter.csv").write_text("x")
    monkeypatch.setenv("SENTINELX_CIC_IDS2018_RAW", str(tmp_path))
    dp = resolve_dataset("cic-ids2018")
    assert dp.raw_path == tmp_path


def test_missing_path_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("SENTINELX_CIC_IDS2018_RAW", raising=False)
    # point config file to an empty temp file so nothing resolves
    empty_cfg = tmp_path / "empty.yaml"
    empty_cfg.write_text("# nothing\n")
    with pytest.raises(FileNotFoundError):
        resolve_dataset("cic-ids2018", config_path=empty_cfg)


def test_ciciot_ignores_mislabeled_cic_files(tmp_path):
    # the CICIoT folder also contains the 10 mislabeled CIC day files -> ignored
    (tmp_path / "train.csv").write_text("x")
    (tmp_path / "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv").write_text("x")
    dp = resolve_dataset("ciciot2023", raw_path=str(tmp_path))
    names = [f.name for f in dp.list_files()]
    assert "train.csv" in names
    assert all("cicflowmeter" not in n.lower() for n in names)


def test_unsw_ignores_payload_file(tmp_path):
    (tmp_path / "UNSW_NB15_training-set.csv").write_text("x")
    (tmp_path / "Payload_data_CICIDS2017.csv").write_text("x")
    dp = resolve_dataset("unsw-nb15", raw_path=str(tmp_path))
    names = [f.name for f in dp.list_files()]
    assert "UNSW_NB15_training-set.csv" in names
    assert all("payload_data_cicids2017" not in n.lower() for n in names)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        resolve_dataset("kddcup99", raw_path=".")


def test_pipeline_config_cache_dir():
    cfg = PipelineConfig(dataset="ctu-13")
    assert cfg.cache_dir().name == "ctu-13"


def test_pcap_interface_optional():
    # pcap_available reflects whether scapy is installed; either way the API is
    # importable and does not block anything.
    avail = pcap.pcap_available()
    assert isinstance(avail, bool)
    if not avail:
        with pytest.raises(ImportError):
            list(pcap.iter_packet_features("nonexistent.pcap", dataset="ctu-13"))
