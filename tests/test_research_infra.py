"""Phase-9 research-infrastructure tests.

Covers the consolidation layer without any real model training:
  * reproducibility utilities (seeding, environment capture, config snapshot,
    run manifest);
  * the experiment registry surface (all named experiments present + dispatch);
  * the model-comparison assembler (only genuinely-measured fields populated;
    unmeasured cells stay blank — never fabricated).
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sentinelx.research import comparison, experiments
from sentinelx.research.reproducibility import (
    RunManifest,
    capture_environment,
    finalize_manifest,
    new_manifest,
    seed_everything,
    snapshot_config,
)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_seed_everything_reexport_is_deterministic():
    seed_everything(7)
    a = np.random.rand(4)
    seed_everything(7)
    b = np.random.rand(4)
    assert np.allclose(a, b)


def test_capture_environment_has_python_and_packages():
    env = capture_environment()
    assert "python" in env and env["python"]
    assert "packages" in env and isinstance(env["packages"], dict)
    # tracked packages appear as keys even if a version is None
    assert "numpy" in env["packages"]
    assert "torch" in env["packages"]


def test_snapshot_config_roundtrip(tmp_path):
    cfg = {"model": {"gnn_type": "gat"}, "seed": 42, "path": Path("x/y")}
    out = snapshot_config(cfg, tmp_path / "cfg.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["seed"] == 42
    assert loaded["model"]["gnn_type"] == "gat"
    assert loaded["path"] == str(Path("x/y"))  # Path serialized to string (OS-agnostic)


def test_run_manifest_lifecycle(tmp_path):
    m = new_manifest("run_experiment", "known_attacks", 42,
                     config={"dataset": "ctu-13"}, repo_root=tmp_path)
    assert m.status == "started"
    assert m.environment.get("python")
    assert m.started_utc
    finalize_manifest(m, status="completed",
                      outputs=[tmp_path / "a.csv"], notes="ok")
    assert m.status == "completed"
    assert m.finished_utc
    assert m.outputs == [str(tmp_path / "a.csv")]
    p = m.write(tmp_path / "manifest.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["experiment"] == "known_attacks"
    assert data["status"] == "completed"


# --------------------------------------------------------------------------- #
# experiment registry
# --------------------------------------------------------------------------- #
def test_all_named_experiments_registered():
    expected = {
        "known_attacks", "unseen_attacks", "kstep", "early_warning",
        "missing_telemetry", "uncertainty_error", "ood_detection",
        "forecast_stability", "cross_dataset", "baseline_comparison",
    }
    assert expected.issubset(set(experiments.ALL_EXPERIMENTS))
    for name in expected:
        assert callable(experiments.EXPERIMENT_REGISTRY[name])


def test_run_experiment_unknown_name_raises():
    with pytest.raises(KeyError):
        experiments.run_experiment("does_not_exist", dataset="ctu-13")


# --------------------------------------------------------------------------- #
# comparison assembler
# --------------------------------------------------------------------------- #
def _write_min_artifacts(exp_dir: Path):
    """Write minimal genuine-looking artifacts the assembler reads."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    # baseline_results.csv (two models)
    with open(exp_dir / "baseline_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "precision", "recall", "f1", "pr_auc",
                    "false_positive_rate"])
        w.writerow(["ctu-13", "logistic_regression", "0.86", "1.0", "0.92", "0.91", "0.27"])
        w.writerow(["ctu-13", "sequence_gru", "0.79", "1.0", "0.88", "0.95", "0.45"])
    # world_model_results.csv (best combo graphsage_lstm)
    with open(exp_dir / "world_model_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "combo", "best_val_state_loss", "test_precision",
                    "test_recall", "test_f1", "test_pr_auc", "test_fpr"])
        w.writerow(["ctu-13", "graphsage_lstm", "0.0025", "0.65", "1.0", "0.79",
                    "0.895", "0.909"])
    # k_step_results.csv (forecast_error @ horizon 1)
    with open(exp_dir / "k_step_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "K", "horizon", "state_cosine_distance"])
        w.writerow(["ctu-13", "1", "1", "0.002565"])
    # uncertainty_ood_metrics.json (ece + ood auroc)
    (exp_dir / "uncertainty_ood_metrics.json").write_text(json.dumps({
        "calibration": {"before": {"ece": 0.2057, "brier": 0.274}},
        "ood": {"metrics": {"auroc": 0.3014}},
    }), encoding="utf-8")
    # phase8_metrics.json (stability)
    (exp_dir / "phase8_metrics.json").write_text(json.dumps({
        "summary": {"mean_stability_score": 0.99999}
    }), encoding="utf-8")
    # models/sentinel_x/metadata.json for best_combo selection
    meta_dir = exp_dir.parent / "models" / "sentinel_x"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "metadata.json").write_text(json.dumps({
        "best_combo": "graphsage_lstm"
    }), encoding="utf-8")


def test_comparison_populates_only_measured_fields(tmp_path):
    exp_dir = tmp_path / "experiments"
    _write_min_artifacts(exp_dir)
    early = {"lead_time": {"mean_windows": 0.148}}
    rows = comparison.build_comparison_rows(
        exp_dir, repo_root=tmp_path, early_warning_result=early)
    by_model = {r["model"]: r for r in rows}

    # baselines: classification metrics present, world-model-only fields blank
    lr = by_model["logistic_regression"]
    assert lr["pr_auc"] == 0.91 and lr["recall"] == 1.0
    assert lr["forecast_error"] is None
    assert lr["lead_time"] is None
    assert lr["ece"] is None
    assert lr["ood_auroc"] is None
    assert lr["stability"] is None

    # world model: all genuinely measured fields populated
    wm = by_model["sentinel_x_world_model"]
    assert wm["pr_auc"] == 0.895
    assert wm["forecast_error"] == 0.002565
    assert wm["lead_time"] == 0.148
    assert wm["ece"] == 0.2057
    assert wm["ood_auroc"] == 0.3014
    assert wm["stability"] == 0.99999


def test_comparison_csv_written_with_blank_unmeasured_cells(tmp_path):
    exp_dir = tmp_path / "experiments"
    _write_min_artifacts(exp_dir)
    path = comparison.build_and_write(exp_dir, repo_root=tmp_path)
    assert path.exists()
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == comparison.COLUMNS
        rows = list(reader)
    lr = next(r for r in rows if r["model"] == "logistic_regression")
    # unmeasured baseline fields are blank strings, never fabricated numbers
    assert lr["forecast_error"] == ""
    assert lr["ece"] == ""
    assert lr["ood_auroc"] == ""
    assert lr["stability"] == ""


def test_comparison_handles_missing_world_model(tmp_path):
    """If no world_model_results.csv exists, only baseline rows appear."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "baseline_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "precision", "recall", "f1", "pr_auc",
                    "false_positive_rate"])
        w.writerow(["ctu-13", "logistic_regression", "0.86", "1.0", "0.92", "0.91", "0.27"])
    rows = comparison.build_comparison_rows(exp_dir, repo_root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["model"] == "logistic_regression"
