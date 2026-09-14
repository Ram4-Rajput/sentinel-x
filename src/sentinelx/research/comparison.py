"""Phase-9: assemble the machine-readable model comparison.

Produces ``experiments/model_comparison.csv`` with the columns required by the
Phase-9 spec:

    model dataset precision recall f1 pr_auc false_positive_rate
    forecast_error lead_time ece ood_auroc stability

RULE: only populate a field that was GENUINELY measured for that (model,
dataset). Anything not measured is left blank — never fabricated. The sources
are the existing experiment artifacts already produced by earlier phases plus
the Phase-9 experiment outputs:

  * precision/recall/f1/pr_auc/false_positive_rate
      - baselines  -> experiments/baseline_results.csv (per model x dataset)
      - world model-> experiments/world_model_results.csv (best combo test row)
  * forecast_error  (world model only) -> k_step_results.csv, state cosine
                     distance at horizon t+1 (the one-step forecast error the
                     model was trained on).
  * lead_time       (world model only) -> Phase-9 early_warning result
                     (mean early-warning lead time in windows), if measured.
  * ece             (world model only) -> uncertainty_ood_metrics.json
                     calibration.before.ece.
  * ood_auroc       (world model only) -> uncertainty_ood_metrics.json
                     ood.metrics.auroc.
  * stability       (world model only) -> phase8_metrics.json
                     summary.mean_stability_score.

Baselines legitimately have no forecast_error/lead_time/ece/ood_auroc/stability
(they are single-step classifiers without the world model's latent forecasting
/ novelty / uncertainty machinery), so those cells are intentionally blank.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

COLUMNS = [
    "model", "dataset", "precision", "recall", "f1", "pr_auc",
    "false_positive_rate", "forecast_error", "lead_time", "ece",
    "ood_auroc", "stability",
]


def _read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in ("n/a", "na", "none", "null"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    if isinstance(v, (int, float)):
        f = float(v)
        return None if f != f else f  # NaN guard
    return None


def _round(v: Optional[float], nd: int = 4) -> Optional[float]:
    return None if v is None else round(v, nd)


def _world_model_forecast_error(exp_dir: Path) -> Optional[float]:
    """One-step (t+1) state cosine distance from k_step_results.csv."""
    rows = _read_csv_rows(exp_dir / "k_step_results.csv")
    for r in rows:
        if r.get("horizon") == "1":
            return _num(r.get("state_cosine_distance"))
    return None


def _world_model_ece_ood(exp_dir: Path) -> Dict[str, Optional[float]]:
    data = _read_json(exp_dir / "uncertainty_ood_metrics.json") or {}
    ece = None
    cal = (data.get("calibration") or {}).get("before") or {}
    ece = _num(cal.get("ece"))
    ood_auroc = _num(((data.get("ood") or {}).get("metrics") or {}).get("auroc"))
    return {"ece": ece, "ood_auroc": ood_auroc}


def _world_model_stability(exp_dir: Path) -> Optional[float]:
    data = _read_json(exp_dir / "phase8_metrics.json") or {}
    return _num((data.get("summary") or {}).get("mean_stability_score"))


def _world_model_lead_time(early_warning_result: Optional[Dict]) -> Optional[float]:
    if not early_warning_result:
        return None
    lt = early_warning_result.get("lead_time") or {}
    return _num(lt.get("mean_windows"))


def _best_world_model_row(exp_dir: Path) -> Optional[Dict[str, str]]:
    """Pick the world-model row matching the saved best combo (from metadata)."""
    meta = _read_json(exp_dir.parent / "models" / "sentinel_x" / "metadata.json")
    best_combo = meta.get("best_combo") if meta else None
    rows = _read_csv_rows(exp_dir / "world_model_results.csv")
    if not rows:
        return None
    if best_combo:
        for r in rows:
            if r.get("combo") == best_combo:
                return r
    # fallback: lowest val state loss
    try:
        return min(rows, key=lambda r: _num(r.get("best_val_state_loss")) or 1e9)
    except Exception:
        return rows[0]


def build_comparison_rows(
    experiments_dir: Path,
    *,
    repo_root: Optional[Path] = None,
    early_warning_result: Optional[Dict] = None,
    world_model_name: str = "sentinel_x_world_model",
) -> List[Dict[str, Any]]:
    """Assemble comparison rows from genuinely measured artifacts only."""
    exp_dir = Path(experiments_dir)
    rows: List[Dict[str, Any]] = []

    # ---- Baselines (classification metrics only) ----
    for r in _read_csv_rows(exp_dir / "baseline_results.csv"):
        rows.append({
            "model": r.get("model"),
            "dataset": r.get("dataset"),
            "precision": _round(_num(r.get("precision"))),
            "recall": _round(_num(r.get("recall"))),
            "f1": _round(_num(r.get("f1"))),
            "pr_auc": _round(_num(r.get("pr_auc"))),
            "false_positive_rate": _round(_num(r.get("false_positive_rate"))),
            # baselines have no forecasting / novelty / uncertainty machinery
            "forecast_error": None,
            "lead_time": None,
            "ece": None,
            "ood_auroc": None,
            "stability": None,
        })

    # ---- Sentinel-X world model (best combo) ----
    wm = _best_world_model_row(exp_dir)
    if wm is not None:
        ece_ood = _world_model_ece_ood(exp_dir)
        rows.append({
            "model": world_model_name,
            "dataset": wm.get("dataset"),
            "precision": _round(_num(wm.get("test_precision"))),
            "recall": _round(_num(wm.get("test_recall"))),
            "f1": _round(_num(wm.get("test_f1"))),
            "pr_auc": _round(_num(wm.get("test_pr_auc"))),
            "false_positive_rate": _round(_num(wm.get("test_fpr"))),
            "forecast_error": _round(_world_model_forecast_error(exp_dir), 6),
            "lead_time": _round(_world_model_lead_time(early_warning_result), 4),
            "ece": _round(ece_ood["ece"]),
            "ood_auroc": _round(ece_ood["ood_auroc"]),
            "stability": _round(_world_model_stability(exp_dir), 6),
        })

    return rows


def write_comparison_csv(rows: List[Dict[str, Any]], path: Path) -> Path:
    """Write model_comparison.csv. Unmeasured fields are blank (never faked)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            # blank cells for None so "genuinely measured only" is visible
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in COLUMNS})
    return path


def build_and_write(
    experiments_dir: Path,
    *,
    repo_root: Optional[Path] = None,
    early_warning_result: Optional[Dict] = None,
    out_path: Optional[Path] = None,
) -> Path:
    rows = build_comparison_rows(
        experiments_dir, repo_root=repo_root,
        early_warning_result=early_warning_result)
    out_path = out_path or (Path(experiments_dir) / "model_comparison.csv")
    return write_comparison_csv(rows, out_path)
