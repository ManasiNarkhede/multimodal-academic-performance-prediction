# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingParameterType=false, reportExplicitAny=false, reportAny=false, reportUnusedCallResult=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportUnannotatedClassAttribute=false

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.core.mlflow import end_run, log_metrics, log_model, log_params, start_run
from app.preprocessing.tabular_processor import engineer_features, load_tabular_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from app.core.paths import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, EVIDENCE_DIR, DOCS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_FEATURES_PATH = PROCESSED_DATA_DIR / "tabular_features.npy"
DEFAULT_STUDENT_IDS_PATH = PROCESSED_DATA_DIR / "tabular_student_ids.npy"
DEFAULT_STUDENTS_CSV = RAW_DATA_DIR / "students.csv"
DEFAULT_MODEL_LATEST_PATH = MODELS_DIR / "tabular_model_latest.json"
DEFAULT_FEATURE_IMPORTANCE_PATH = DOCS_DIR / "tabular_feature_importance.png"
DEFAULT_EVIDENCE_PATH = EVIDENCE_DIR / "task-11-tabular-model-metrics.json"


class TabularModel:
    """Thin wrapper around XGBoost for tabular risk classification."""

    def __init__(self, scale_pos_weight: float | None = None, random_state: int = 42):
        kwargs: dict[str, Any] = {
            "max_depth": 4,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": 1,
        }
        if scale_pos_weight is not None:
            kwargs["scale_pos_weight"] = scale_pos_weight
        self.model = XGBClassifier(**kwargs)

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> None:
        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def save(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)

    def load(self, model_path: str | Path) -> None:
        self.model.load_model(Path(model_path))


class TabularPredictor:
    """Loads the latest trained tabular model and returns risk probabilities."""

    def __init__(self, model_path: str | Path | None = None):
        resolved_path = Path(model_path) if model_path is not None else self._resolve_latest_model()
        if resolved_path is None or not resolved_path.exists():
            raise FileNotFoundError("No trained tabular model found.")

        self.model_path = resolved_path
        self.model = XGBClassifier()
        self.model.load_model(self.model_path)

    def _resolve_latest_model(self) -> Path | None:
        if DEFAULT_MODEL_LATEST_PATH.exists() or DEFAULT_MODEL_LATEST_PATH.is_symlink():
            return DEFAULT_MODEL_LATEST_PATH

        candidates = sorted(MODELS_DIR.glob("tabular_model_*.json"))
        return candidates[-1] if candidates else None

    def predict(self, features_array: np.ndarray) -> float:
        features_array = np.asarray(features_array, dtype=np.float32)
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)

        start = time.perf_counter()
        probability = float(self.model.predict_proba(features_array)[0, 1])
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("Single tabular inference completed in %.3f ms", elapsed_ms)
        return probability

    def predict_batch(self, features_array: np.ndarray) -> np.ndarray:
        features_array = np.asarray(features_array, dtype=np.float32)
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        return self.model.predict_proba(features_array)[:, 1]


def _load_training_data(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    student_ids_path: str | Path = DEFAULT_STUDENT_IDS_PATH,
    students_csv: str | Path = DEFAULT_STUDENTS_CSV,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    features = np.load(features_path)
    student_ids = np.load(student_ids_path)

    students_df = load_tabular_data(students_csv)
    _, feature_names = engineer_features(students_df)
    label_map = dict(zip(students_df["student_id"], students_df["at_risk"]))
    labels = np.array([label_map[int(student_id)] for student_id in student_ids], dtype=np.int32)

    feature_frame = pd.DataFrame(features, columns=feature_names)
    return feature_frame, labels, feature_names


def _compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else 0.0,
    }


def _save_feature_importance_plot(
    trained_model: XGBClassifier,
    feature_names: list[str],
    output_path: str | Path = DEFAULT_FEATURE_IMPORTANCE_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importances = trained_model.feature_importances_
    importance_frame = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_frame["feature"], importance_frame["importance"], color="#3b82f6")
    plt.xlabel("Importance")
    plt.title("Tabular Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def _persist_latest_pointer(model_path: Path, latest_path: Path) -> Path:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()

    try:
        latest_path.symlink_to(model_path.name)
    except OSError:
        shutil.copy2(model_path, latest_path)
    return latest_path


def _write_evidence(payload: dict[str, Any], evidence_path: str | Path = DEFAULT_EVIDENCE_PATH) -> Path:
    evidence_path = Path(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return evidence_path


def train_tabular_model(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    student_ids_path: str | Path = DEFAULT_STUDENT_IDS_PATH,
    students_csv: str | Path = DEFAULT_STUDENTS_CSV,
    models_dir: str | Path = MODELS_DIR,
    feature_importance_path: str | Path = DEFAULT_FEATURE_IMPORTANCE_PATH,
    evidence_path: str | Path = DEFAULT_EVIDENCE_PATH,
    val_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    logger.info("Loading tabular features and labels...")
    features, labels, feature_names = _load_training_data(features_path, student_ids_path, students_csv)

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=val_size,
        random_state=random_state,
        stratify=labels,
    )

    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = negatives / positives if positives else 1.0

    start_run(experiment_name="tabular_model", run_name=f"tabular_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_params({
        "val_size": val_size,
        "random_state": random_state,
        "feature_count": len(feature_names),
        "train_size": int(len(x_train)),
        "validation_size": int(len(x_val)),
        "positive_rate": float(labels.mean()),
        "scale_pos_weight": float(scale_pos_weight),
    })

    logger.info("Training XGBoost classifier with scale_pos_weight=%.4f", scale_pos_weight)
    tabular_model = TabularModel(scale_pos_weight=scale_pos_weight, random_state=random_state)
    tabular_model.fit(x_train, y_train)

    val_probabilities = tabular_model.predict_proba(x_val)
    validation_metrics = _compute_metrics(y_val, val_probabilities)

    train_probabilities = tabular_model.predict_proba(x_train)
    train_metrics = _compute_metrics(y_train, train_probabilities)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"tabular_model_{timestamp}.json"
    tabular_model.save(model_path)
    latest_path = _persist_latest_pointer(model_path, models_dir / "tabular_model_latest.json")
    plot_path = _save_feature_importance_plot(tabular_model.model, feature_names, feature_importance_path)

    inference_probe = np.asarray(x_val.iloc[0].to_numpy(dtype=np.float32))
    predictor = TabularPredictor(model_path=latest_path)
    single_start = time.perf_counter()
    predictor.predict(inference_probe)
    single_inference_ms = (time.perf_counter() - single_start) * 1000
    training_time_seconds = time.perf_counter() - started_at

    log_metrics({
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in validation_metrics.items()},
        "training_time_seconds": float(training_time_seconds),
        "single_inference_ms": float(single_inference_ms),
    })
    log_model(str(model_path), artifact_path="model")
    end_run()

    result = {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_size": int(len(x_train)),
        "validation_size": int(len(x_val)),
        "positive_rate": float(labels.mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "training_time_seconds": float(training_time_seconds),
        "single_inference_ms": float(single_inference_ms),
        "model_path": str(model_path),
        "latest_model_path": str(latest_path),
        "feature_importance_path": str(plot_path),
        "evidence_path": str(_write_evidence(
            {
                "model_path": str(model_path),
                "latest_model_path": str(latest_path),
                "feature_importance_path": str(plot_path),
                "training_time_seconds": float(training_time_seconds),
                "single_inference_ms": float(single_inference_ms),
                "validation_metrics": validation_metrics,
                "train_metrics": train_metrics,
                "feature_names": feature_names,
                "scale_pos_weight": float(scale_pos_weight),
            },
            evidence_path,
        )),
    }
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    training_result = train_tabular_model()
    print("Training complete.")
    print(f"Validation F1: {training_result['validation_metrics']['f1']:.4f}")
    print(f"Training time (s): {training_result['training_time_seconds']:.3f}")
    print(f"Single inference (ms): {training_result['single_inference_ms']:.3f}")
    print(f"Model saved to: {training_result['model_path']}")
