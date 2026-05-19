from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# MLflow import with graceful fallback
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowTracker:
    """
    Experiment tracker wrapping MLflow.

    Falls back to CSV logging if MLflow is not installed,
    maintaining full backward compatibility with existing code.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name. Created if it does not exist.
    tracking_uri : str
        Where to store MLflow runs. Default: ./mlruns
    csv_fallback_path : str
        Path for CSV fallback logging.
    """

    def __init__(
        self,
        experiment_name: str = "adni_cross_validation",
        tracking_uri: str = "mlruns/",
        csv_fallback_path: str = "output/results/master_results.csv",
    ):
        self.experiment_name   = experiment_name
        self.tracking_uri      = tracking_uri
        self.csv_fallback_path = Path(csv_fallback_path)
        self._active_run_id    = None
        self._fold_metrics: List[dict] = []
        self._run_context: dict = {}

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
        else:
            print(
                "[MLflowTracker] MLflow not installed — falling back to CSV logging.\n"
                "Install: pip install mlflow"
            )

    # Context manager
    @contextmanager
    def start_run(
        self,
        model_name: str,
        classifier_type: str,
        run_name: Optional[str] = None,
        tags: Optional[dict] = None,
    ):
        self._fold_metrics  = []
        self._run_context   = {
            "model_name":      model_name,
            "classifier_type": classifier_type,
        }

        run_name = run_name or "ADNI_{model_name}_{classifier_type}"

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=run_name, tags=tags or {}):
                mlflow.log_params({
                    "model_name":      model_name,
                    "classifier_type": classifier_type,
                    "disease":         "ADNI",
                })
                self._active_run_id = mlflow.active_run().info.run_id
                yield self
                self._active_run_id = None
        else:
            yield self

    # Per-fold logging
    def log_fold(
        self,
        fold: int,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        step = step if step is not None else fold
        fold_entry = {"fold": fold, **metrics}
        self._fold_metrics.append(fold_entry)

        if MLFLOW_AVAILABLE and mlflow.active_run():
            prefixed = {f"fold{fold}/{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(prefixed, step=step)

    # Final experiment logging
    def log_experiment(
        self,
        final_metrics: Dict[str, float],
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics(final_metrics)

            if config:
                # MLflow params must be strings ≤ 250 chars
                mlflow.log_params({
                    k: str(v)[:250] for k, v in config.items()
                })

            # Log fold summary as a JSON artifact
            if self._fold_metrics:
                fold_summary_path = "/tmp/fold_summary.json"
                with open(fold_summary_path, "w") as f:
                    json.dump(self._fold_metrics, f, indent=2)
                mlflow.log_artifact(fold_summary_path, artifact_path="fold_metrics")

            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="model")

        # Always also write to CSV (backward compat)
        self._append_csv(final_metrics, config)


    # CSV fallback
    def _append_csv(
        self,
        final_metrics: dict,
        config: Optional[dict],
    ):
        import pandas as pd

        row = {**self._run_context, **final_metrics}
        if config:
            row.update(config)

        self.csv_fallback_path.parent.mkdir(parents=True, exist_ok=True)
        df_new = pd.DataFrame([row])

        if self.csv_fallback_path.exists():
            df_existing = pd.read_csv(self.csv_fallback_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(self.csv_fallback_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Integration helper — drop-in for cross_validation.py
# ─────────────────────────────────────────────────────────────────────────────

def log_cross_validation_result(
    model_name: str,
    classifier_type: str,
    fold_metrics: List[dict],
    final_metrics: dict,
    config: dict,
    model_path: Optional[str] = None,
    tracking_uri: str = "mlruns/",
    experiment_name: Optional[str] = None,
):
    exp_name  = experiment_name or "ADNI_cross_validation"
    tracker   = MLflowTracker(
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
    )

    with tracker.start_run(model_name, classifier_type):
        for entry in fold_metrics:
            fold = entry.get("fold", 0)
            metrics = {k: v for k, v in entry.items() if k != "fold"}
            tracker.log_fold(fold=fold, metrics=metrics)

        tracker.log_experiment(
            final_metrics=final_metrics,
            model_path=model_path,
            config=config,
        )
