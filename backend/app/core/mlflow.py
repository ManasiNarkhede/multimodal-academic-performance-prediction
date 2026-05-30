"""MLflow experiment tracking helpers for the academic performance prediction system."""

import logging
import os
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global client instance (lazy-initialized)
_client: Optional[MlflowClient] = None


def get_tracking_uri() -> str:
    """Return the configured MLflow tracking URI."""
    return settings.MLFLOW_TRACKING_URI


def _get_client() -> MlflowClient:
    """Return (and cache) an MlflowClient instance."""
    global _client
    if _client is None:
        _client = MlflowClient(tracking_uri=get_tracking_uri())
    return _client


def _ensure_tracking_uri() -> None:
    """Set the global MLflow tracking URI if it differs from the current one."""
    current = mlflow.get_tracking_uri()
    target = get_tracking_uri()
    if current != target:
        mlflow.set_tracking_uri(target)
        logger.info("Set MLflow tracking URI to %s", target)


def start_run(experiment_name: str, run_name: Optional[str] = None) -> None:
    """Start an MLflow run under the given experiment.

    Creates the experiment if it does not already exist.
    """
    _ensure_tracking_uri()

    client = _get_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=settings.MLFLOW_ARTIFACT_ROOT,
        )
        logger.info("Created MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)
    else:
        experiment_id = experiment.experiment_id

    mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    logger.info("Started MLflow run: %s", run_name or "(no name)")


def log_params(params_dict: Dict[str, Any]) -> None:
    """Log a dictionary of parameters to the active MLflow run."""
    for key, value in params_dict.items():
        # MLflow params must be strings; coerce scalars safely
        if isinstance(value, (list, dict, tuple)):
            value = str(value)
        mlflow.log_param(key, value)
    logger.debug("Logged %d parameters", len(params_dict))


def log_metrics(metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
    """Log a dictionary of metrics to the active MLflow run.

    Args:
        metrics_dict: Mapping from metric name to float value.
        step: Optional integer step (e.g., epoch number).
    """
    for key, value in metrics_dict.items():
        try:
            mlflow.log_metric(key, float(value), step=step)
        except Exception:
            logger.warning("Failed to log metric %s=%s", key, value)
    logger.debug("Logged %d metrics (step=%s)", len(metrics_dict), step)


def log_model(model_path: str, artifact_path: str = "model") -> None:
    """Log a model artifact (file or directory) to the active MLflow run.

    Args:
        model_path: Local filesystem path to the model file/directory.
        artifact_path: Destination artifact path within the run (default: "model").
    """
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path, artifact_path=artifact_path)
        logger.info("Logged model artifact: %s -> %s", model_path, artifact_path)
    else:
        logger.warning("Model path does not exist, skipping artifact log: %s", model_path)


def end_run() -> None:
    """End the active MLflow run."""
    mlflow.end_run()
    logger.info("Ended MLflow run")
