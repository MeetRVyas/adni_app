"""
mlflow_tracker.py
=================
Replaces the CSV append in cross_validation.py.

Wraps MLflow to provide the same interface as the existing
master_results.csv logging, but with full experiment tracking.

Usage
-----
    from data_pipeline.tracking.mlflow_tracker import MLflowTracker

    tracker = MLflowTracker(
        experiment_name="adni_cross_validation",
        tracking_uri="mlruns/",
    )

    with tracker.start_run(
        model_name="resnet18",
        classifier_type="progressive",
        disease="adni",
    ):
        for fold in range(5):
            tracker.log_fold(fold=fold, metrics={
                "val_recall": 0.97,
                "val_accuracy": 96.5,
                "val_f1": 0.965,
            })

        tracker.log_experiment(
            final_metrics={"test_recall": 0.982, "test_accuracy": 97.8},
            model_path="output/results/best_models/resnet18_best.pth",
            config={"epochs": 30, "batch_size": 32, "lr": 1e-4},
        )

Query results with:
    mlflow ui --backend-store-uri mlruns/
    # → opens at http://localhost:5000
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── MLflow import with graceful fallback ─────────────────────────────────────
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

    # ── Context manager ────────────────────────────────────────────────────────

    @contextmanager
    def start_run(
        self,
        model_name: str,
        classifier_type: str,
        disease: str,
        run_name: Optional[str] = None,
        tags: Optional[dict] = None,
    ):
        """
        Context manager for a single experiment run.

        Usage:
            with tracker.start_run("resnet18", "progressive", "adni"):
                tracker.log_fold(...)
                tracker.log_experiment(...)
        """
        self._fold_metrics  = []
        self._run_context   = {
            "model_name":      model_name,
            "classifier_type": classifier_type,
            "disease":         disease,
        }

        run_name = run_name or f"{disease}_{model_name}_{classifier_type}"

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=run_name, tags=tags or {}):
                mlflow.log_params({
                    "model_name":      model_name,
                    "classifier_type": classifier_type,
                    "disease":         disease,
                })
                self._active_run_id = mlflow.active_run().info.run_id
                yield self
                self._active_run_id = None
        else:
            yield self

    # ── Per-fold logging ──────────────────────────────────────────────────────

    def log_fold(
        self,
        fold: int,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """
        Log metrics for a single fold.

        Parameters
        ----------
        fold : int
            Fold index (0-based).
        metrics : dict
            Metric name → value, e.g. {"val_recall": 0.97}.
        step : int | None
            Global step (e.g., epoch number). Defaults to fold index.
        """
        step = step if step is not None else fold
        fold_entry = {"fold": fold, **metrics}
        self._fold_metrics.append(fold_entry)

        if MLFLOW_AVAILABLE and mlflow.active_run():
            prefixed = {f"fold{fold}/{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(prefixed, step=step)

    # ── Final experiment logging ───────────────────────────────────────────────

    def log_experiment(
        self,
        final_metrics: Dict[str, float],
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[str]] = None,
    ):
        """
        Log final test metrics, config, and model artifact.

        Parameters
        ----------
        final_metrics : dict
            Test set metrics, e.g. {"test_recall": 0.98, "test_accuracy": 97.5}.
        model_path : str | None
            Path to the best model .pth file.
        config : dict | None
            Training hyperparameters.
        artifacts : list | None
            Paths to additional artifact files (plots, reports).
        """
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

            if artifacts:
                for artifact in artifacts:
                    if os.path.exists(artifact):
                        mlflow.log_artifact(artifact)

        # Always also write to CSV (backward compat)
        self._append_csv(final_metrics, config)

    # ── Cross-disease query helpers ───────────────────────────────────────────

    @staticmethod
    def get_best_run(
        experiment_name: str,
        metric: str = "test_recall",
        tracking_uri: str = "mlruns/",
    ) -> Optional[dict]:
        """
        Return the run with the highest value of a given metric.

        Returns None if MLflow is not installed.
        """
        if not MLFLOW_AVAILABLE:
            return None

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return None

        run = runs[0]
        return {
            "run_id":   run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "params":   run.data.params,
            "metrics":  run.data.metrics,
        }

    @staticmethod
    def compare_classifiers(
        experiment_name: str,
        metric: str = "test_recall",
        tracking_uri: str = "mlruns/",
    ):
        """
        Print a sorted table of all runs ranked by the given metric.
        """
        if not MLFLOW_AVAILABLE:
            print("[MLflowTracker] MLflow not available — cannot compare runs")
            return

        try:
            import pandas as pd
        except ImportError:
            return

        mlflow.set_tracking_uri(tracking_uri)
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=[f"metrics.{metric} DESC"],
        )
        if runs.empty:
            print("No runs found.")
            return

        cols = ["tags.mlflow.runName", f"metrics.{metric}",
                "params.model_name", "params.classifier_type", "params.disease"]
        cols = [c for c in cols if c in runs.columns]
        print(runs[cols].to_string(index=False))

    # ── CSV fallback ──────────────────────────────────────────────────────────

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
    disease: str,
    fold_metrics: List[dict],
    final_metrics: dict,
    config: dict,
    model_path: Optional[str] = None,
    tracking_uri: str = "mlruns/",
    experiment_name: Optional[str] = None,
):
    """
    One-shot function: log a complete cross-validation result to MLflow.

    This is the simplest integration point — call it from
    Cross_Validator._run_classifier() at the end of each experiment.

    Parameters
    ----------
    fold_metrics : list of dicts
        One dict per fold with keys like val_recall, val_acc, etc.
    final_metrics : dict
        Test set metrics.
    config : dict
        Hyperparameters (epochs, batch_size, lr, etc.).
    """
    exp_name  = experiment_name or f"{disease}_cross_validation"
    tracker   = MLflowTracker(
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
    )

    with tracker.start_run(model_name, classifier_type, disease):
        for entry in fold_metrics:
            fold = entry.get("fold", 0)
            metrics = {k: v for k, v in entry.items() if k != "fold"}
            tracker.log_fold(fold=fold, metrics=metrics)

        tracker.log_experiment(
            final_metrics=final_metrics,
            model_path=model_path,
            config=config,
        )
