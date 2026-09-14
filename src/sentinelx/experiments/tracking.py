"""MLflow experiment tracking (experiment-tracking skill).

Logs dataset, model, hyperparameters, metrics, training time, parameter count,
seed, split configuration, and feature configuration. MLflow is optional at
runtime: if unavailable the tracker degrades to a no-op so baselines still run
and the CSV/JSON/MD outputs are still produced.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, Optional


class MlflowTracker:
    def __init__(self, tracking_uri: Optional[str], experiment: str,
                 artifact_location: Optional[str] = None):
        self.enabled = False
        self._mlflow = None
        try:
            import mlflow
            self._mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            # Create experiment with an explicit artifact location (SQLite backend
            # needs one), then select it. Tolerate pre-existing experiments.
            try:
                if mlflow.get_experiment_by_name(experiment) is None:
                    mlflow.create_experiment(experiment, artifact_location=artifact_location)
            except Exception:
                pass
            mlflow.set_experiment(experiment)
            self.enabled = True
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[tracking] MLflow unavailable ({type(e).__name__}: {e}); continuing without it.")

    @contextlib.contextmanager
    def run(self, run_name: str):
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=run_name):
            yield self._mlflow

    def log(self, mlflow, *, params: Dict, metrics: Dict, tags: Optional[Dict] = None):
        if not self.enabled or mlflow is None:
            return
        # flatten/scrub None for MLflow
        clean_params = {k: v for k, v in params.items() if v is not None}
        clean_metrics = {k: float(v) for k, v in metrics.items()
                         if isinstance(v, (int, float)) and v == v}  # drop NaN
        mlflow.log_params(clean_params)
        mlflow.log_metrics(clean_metrics)
        if tags:
            mlflow.set_tags(tags)
