"""Phase-3 runner test: end-to-end run of all three baselines with the shared
protocol, producing baseline_results.csv / baseline_metrics.json /
baseline_report.md, and recording to a temp MLflow SQLite store."""

import json
from pathlib import Path

import numpy as np

from sentinelx.experiments.common import ExperimentConfig
from sentinelx.experiments.runner import run_all_baselines, write_outputs


def _write_synth_cache(root: Path, dataset: str, n_windows: int, attack_from: int):
    wdir = root / dataset / "windows"
    wdir.mkdir(parents=True, exist_ok=True)
    n_tr = int(n_windows * 0.7)
    n_va = int(n_windows * 0.15)
    splits = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va),
              "test": range(n_tr + n_va, n_windows)}
    gi = 0
    rng = np.random.RandomState(0)
    for split, idxs in splits.items():
        with open(wdir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for local_i, _ in enumerate(idxs):
                attack = 1 if gi >= attack_from else 0
                g = {
                    "window_index": local_i, "start": None, "end": None,
                    "num_nodes": 2, "num_edges": 1, "num_flows": 3 + 5 * attack,
                    "node_ids": ["a", "b"],
                    "node_features": [[1.0, 0.0, 100.0 + 50 * attack + rng.rand(), 0.0, 2.0],
                                      [0.0, 1.0, 0.0, 50.0, 1.0]],
                    "edge_index": [[0, 1]],
                    "edge_features": [[1.0, 150.0 + 200.0 * attack, float(attack)]],
                    "attack_ratio": 0.5 * attack, "label_any_attack": attack,
                }
                fh.write(json.dumps(g) + "\n")
                gi += 1


def test_run_all_and_write_outputs(tmp_path):
    _write_synth_cache(tmp_path, "ctu-13", n_windows=60, attack_from=30)
    cfg = ExperimentConfig(dataset="ctu-13", processed_root=tmp_path, seq_len=4, horizon=1)
    mlflow_uri = "sqlite:///" + str((tmp_path / "mlflow.db").resolve()).replace("\\", "/")
    artifacts = "file:///" + str((tmp_path / "art").resolve()).replace("\\", "/")

    res = run_all_baselines(cfg, temporal_type="gru", gnn_type="sage", epochs=5,
                            mlflow_uri=mlflow_uri, mlflow_artifacts=artifacts,
                            log=lambda m: None)
    assert not res.get("skipped")
    # all three baselines present
    assert "logistic_regression" in res["models"]
    assert "sequence_gru" in res["models"]
    assert "temporal_gnn_sage" in res["models"]

    out = write_outputs([res], tmp_path / "experiments")
    assert out["csv"].exists() and out["json"].exists() and out["md"].exists()

    # CSV has 3 model rows
    csv_text = out["csv"].read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_text) == 1 + 3  # header + 3 models

    # JSON reloads and has the models
    data = json.loads(out["json"].read_text(encoding="utf-8"))
    assert data[0]["dataset"] == "ctu-13"
    assert set(data[0]["models"]) == {"logistic_regression", "sequence_gru", "temporal_gnn_sage"}

    # report mentions the three baselines and the fairness section
    md = out["md"].read_text(encoding="utf-8")
    assert "logistic_regression" in md
    assert "Fairness & reproducibility" in md
    assert "Sentinel-X world model is intentionally NOT" in md


def test_runner_skips_unusable_dataset(tmp_path):
    _write_synth_cache(tmp_path, "tiny", n_windows=6, attack_from=3)
    cfg = ExperimentConfig(dataset="tiny", processed_root=tmp_path, min_windows=12)
    res = run_all_baselines(cfg, epochs=2, mlflow_uri=None, log=lambda m: None)
    assert res.get("skipped") is True
    # still writable without crashing
    out = write_outputs([res], tmp_path / "experiments")
    assert out["md"].exists()


def test_mlflow_records_created(tmp_path):
    _write_synth_cache(tmp_path, "ctu-13", n_windows=60, attack_from=30)
    cfg = ExperimentConfig(dataset="ctu-13", processed_root=tmp_path, seq_len=4, horizon=1)
    mlflow_uri = "sqlite:///" + str((tmp_path / "mlflow.db").resolve()).replace("\\", "/")
    artifacts = "file:///" + str((tmp_path / "art").resolve()).replace("\\", "/")
    run_all_baselines(cfg, epochs=3, mlflow_uri=mlflow_uri, mlflow_artifacts=artifacts,
                      log=lambda m: None)
    import mlflow
    mlflow.set_tracking_uri(mlflow_uri)
    exp = mlflow.get_experiment_by_name("sentinelx-phase3-baselines")
    assert exp is not None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 3  # one per baseline
