import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.models.behavioral_model import BehavioralPredictor
from app.models.tabular_model import TabularPredictor
from app.models.text_model import TextPredictor

logger = logging.getLogger(__name__)

from app.core.paths import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, EVIDENCE_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_STUDENTS_CSV = RAW_DATA_DIR / "students.csv"
DEFAULT_TEXT_FEATURES_PATH = PROCESSED_DATA_DIR / "text_features.npy"
DEFAULT_TEXT_STUDENT_IDS_PATH = PROCESSED_DATA_DIR / "text_student_ids.npy"
DEFAULT_TABULAR_FEATURES_PATH = PROCESSED_DATA_DIR / "tabular_features.npy"
DEFAULT_TABULAR_STUDENT_IDS_PATH = PROCESSED_DATA_DIR / "tabular_student_ids.npy"
DEFAULT_BEHAVIORAL_SEQUENCES_PATH = PROCESSED_DATA_DIR / "behavioral_sequences.npy"
DEFAULT_BEHAVIORAL_STUDENT_IDS_PATH = PROCESSED_DATA_DIR / "behavioral_student_ids.npy"
DEFAULT_MODEL_LATEST_PATH = MODELS_DIR / "fusion_model_latest.json"
DEFAULT_EVIDENCE_PATH = EVIDENCE_DIR / "task-14-fusion-metrics.json"


class FusionEnsemble:
    """Late fusion ensemble combining text, tabular, and behavioral model predictions."""

    def __init__(
        self,
        meta_learner_type: str = "xgboost",
        random_state: int = 42,
    ):
        self.meta_learner_type = meta_learner_type
        self.random_state = random_state
        self.meta_learner: Optional[Any] = None
        self.population_means: Dict[str, float] = {}
        self.single_modality_metrics: Dict[str, Dict[str, float]] = {}
        self.validation_metrics: Dict[str, float] = {}
        self.train_metrics: Dict[str, float] = {}
        self.feature_importances: Optional[Dict[str, float]] = None

    def _create_meta_learner(self) -> Any:
        if self.meta_learner_type == "xgboost":
            return XGBClassifier(
                max_depth=3,
                n_estimators=50,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=1,
            )
        elif self.meta_learner_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
        else:
            raise ValueError(f"Unknown meta_learner_type: {self.meta_learner_type}")

    def _load_base_predictions(
        self,
        text_features_path: Path = DEFAULT_TEXT_FEATURES_PATH,
        text_student_ids_path: Path = DEFAULT_TEXT_STUDENT_IDS_PATH,
        tabular_features_path: Path = DEFAULT_TABULAR_FEATURES_PATH,
        tabular_student_ids_path: Path = DEFAULT_TABULAR_STUDENT_IDS_PATH,
        behavioral_sequences_path: Path = DEFAULT_BEHAVIORAL_SEQUENCES_PATH,
        behavioral_student_ids_path: Path = DEFAULT_BEHAVIORAL_STUDENT_IDS_PATH,
        students_csv: Path = DEFAULT_STUDENTS_CSV,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load base model predictions aligned by student_id.

        Returns:
            train_df, val_df, test_df, all_df: DataFrames with columns
            [student_id, text_prob, tabular_prob, behavioral_prob, label]
        """
        logger.info("Loading base model predictions...")

        text_model_path = MODELS_DIR / "text_model_latest.pt"
        tabular_model_path = MODELS_DIR / "tabular_model_latest.json"
        behavioral_model_path = MODELS_DIR / "behavioral_model_latest.pt"

        text_predictor = TextPredictor(model_path=str(text_model_path))
        tabular_predictor = TabularPredictor(model_path=str(tabular_model_path))
        behavioral_predictor = BehavioralPredictor(model_path=str(behavioral_model_path))

        text_features = np.load(text_features_path)
        text_ids = np.load(text_student_ids_path)
        tabular_features = np.load(tabular_features_path)
        tabular_ids = np.load(tabular_student_ids_path)
        behavioral_sequences = np.load(behavioral_sequences_path)
        behavioral_ids = np.load(behavioral_student_ids_path)

        students_df = pd.read_csv(students_csv)
        label_map = dict(zip(students_df["student_id"], students_df["at_risk"]))

        text_probs = text_predictor.predict_batch(text_features)
        tabular_probs = tabular_predictor.predict_batch(tabular_features)
        behavioral_probs = behavioral_predictor.predict(behavioral_sequences)

        text_df = pd.DataFrame({
            "student_id": text_ids,
            "text_prob": text_probs,
        })
        tabular_df = pd.DataFrame({
            "student_id": tabular_ids,
            "tabular_prob": tabular_probs,
        })
        behavioral_df = pd.DataFrame({
            "student_id": behavioral_ids,
            "behavioral_prob": behavioral_probs,
        })

        merged = text_df.merge(tabular_df, on="student_id", how="outer")
        merged = merged.merge(behavioral_df, on="student_id", how="outer")
        merged["label"] = merged["student_id"].map(label_map)

        all_ids = merged["student_id"].values
        train_ids, temp_ids = train_test_split(
            all_ids, test_size=0.3, random_state=self.random_state, stratify=merged["label"].values
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=0.5, random_state=self.random_state,
            stratify=merged[merged["student_id"].isin(temp_ids)]["label"].values
        )

        train_df = merged[merged["student_id"].isin(train_ids)].copy()
        val_df = merged[merged["student_id"].isin(val_ids)].copy()
        test_df = merged[merged["student_id"].isin(test_ids)].copy()

        return train_df, val_df, test_df, merged

    def _compute_single_modality_metrics(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each single modality on the given dataframe."""
        metrics = {}
        for col in ["text_prob", "tabular_prob", "behavioral_prob"]:
            probs = df[col].values
            labels = df["label"].values
            preds = (probs >= 0.5).astype(int)
            metrics[col.replace("_prob", "")] = {
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
                "f1": float(f1_score(labels, preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
            }
        return metrics

    def _compute_metrics(self, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        predictions = (probabilities >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else 0.0,
        }

    def train(
        self,
        text_features_path: Path = DEFAULT_TEXT_FEATURES_PATH,
        text_student_ids_path: Path = DEFAULT_TEXT_STUDENT_IDS_PATH,
        tabular_features_path: Path = DEFAULT_TABULAR_FEATURES_PATH,
        tabular_student_ids_path: Path = DEFAULT_TABULAR_STUDENT_IDS_PATH,
        behavioral_sequences_path: Path = DEFAULT_BEHAVIORAL_SEQUENCES_PATH,
        behavioral_student_ids_path: Path = DEFAULT_BEHAVIORAL_STUDENT_IDS_PATH,
        students_csv: Path = DEFAULT_STUDENTS_CSV,
    ) -> Dict[str, Any]:
        """Train the fusion ensemble meta-learner.

        Returns:
            Dictionary with training results and metrics.
        """
        started_at = time.perf_counter()

        train_df, val_df, test_df, all_df = self._load_base_predictions(
            text_features_path, text_student_ids_path,
            tabular_features_path, tabular_student_ids_path,
            behavioral_sequences_path, behavioral_student_ids_path,
            students_csv,
        )

        self.population_means = {
            "text": float(all_df["text_prob"].mean()),
            "tabular": float(all_df["tabular_prob"].mean()),
            "behavioral": float(all_df["behavioral_prob"].mean()),
        }

        feature_cols = ["text_prob", "tabular_prob", "behavioral_prob"]
        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        logger.info(
            f"Training fusion ensemble on {len(X_train)} samples, "
            f"validating on {len(X_val)}, testing on {len(X_test)}"
        )

        self.meta_learner = self._create_meta_learner()
        self.meta_learner.fit(X_train, y_train)

        train_probs = self.meta_learner.predict_proba(X_train)[:, 1]
        val_probs = self.meta_learner.predict_proba(X_val)[:, 1]
        test_probs = self.meta_learner.predict_proba(X_test)[:, 1]

        self.train_metrics = self._compute_metrics(y_train, train_probs)
        self.validation_metrics = self._compute_metrics(y_val, val_probs)
        test_metrics = self._compute_metrics(y_test, test_probs)

        self.single_modality_metrics = {
            "train": self._compute_single_modality_metrics(train_df),
            "validation": self._compute_single_modality_metrics(val_df),
            "test": self._compute_single_modality_metrics(test_df),
        }

        if hasattr(self.meta_learner, "feature_importances_"):
            self.feature_importances = dict(zip(feature_cols, self.meta_learner.feature_importances_.tolist()))
        elif hasattr(self.meta_learner, "coef_"):
            self.feature_importances = dict(zip(feature_cols, np.abs(self.meta_learner.coef_[0]).tolist()))

        training_time_seconds = time.perf_counter() - started_at

        best_single_f1 = max(
            self.single_modality_metrics["test"][m]["f1"]
            for m in ["text", "tabular", "behavioral"]
        )

        result = {
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": test_metrics,
            "single_modality_metrics": self.single_modality_metrics,
            "population_means": self.population_means,
            "feature_importances": self.feature_importances,
            "best_single_modality_f1": best_single_f1,
            "improvement_over_best_single": test_metrics["f1"] - best_single_f1,
            "training_time_seconds": float(training_time_seconds),
            "meta_learner_type": self.meta_learner_type,
        }

        logger.info(
            f"Fusion ensemble trained. Test F1={test_metrics['f1']:.4f}, "
            f"best single modality F1={best_single_f1:.4f}"
        )
        return result

    def predict(self, text_prob: float, tabular_prob: float, behavioral_prob: float) -> float:
        """Return fused probability from all three modalities."""
        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not trained. Call train() first.")

        X = np.array([[text_prob, tabular_prob, behavioral_prob]], dtype=np.float32)
        prob = float(self.meta_learner.predict_proba(X)[0, 1])
        return prob

    def predict_with_missing(
        self,
        text_prob: Optional[float] = None,
        tabular_prob: Optional[float] = None,
        behavioral_prob: Optional[float] = None,
    ) -> float:
        """Return fused probability, handling missing modalities with population means."""
        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not trained. Call train() first.")

        text = text_prob if text_prob is not None else self.population_means.get("text", 0.5)
        tabular = tabular_prob if tabular_prob is not None else self.population_means.get("tabular", 0.5)
        behavioral = behavioral_prob if behavioral_prob is not None else self.population_means.get("behavioral", 0.5)

        return self.predict(text, tabular, behavioral)

    def predict_batch(self, probabilities: np.ndarray) -> np.ndarray:
        """Return fused probabilities for a batch of [text_prob, tabular_prob, behavioral_prob]."""
        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not trained. Call train() first.")

        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        return self.meta_learner.predict_proba(probabilities)[:, 1]

    def save(self, model_path: str | Path) -> None:
        """Save the fusion ensemble meta-learner and metadata."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        meta_path = model_path.with_suffix(".json")
        learner_path = model_path.parent / (model_path.stem + "_learner.json")

        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not trained. Call train() first.")

        if hasattr(self.meta_learner, "save_model"):
            self.meta_learner.save_model(learner_path)
        else:
            import joblib
            joblib.dump(self.meta_learner, learner_path)

        payload = {
            "meta_learner_type": self.meta_learner_type,
            "population_means": self.population_means,
            "single_modality_metrics": self.single_modality_metrics,
            "validation_metrics": self.validation_metrics,
            "train_metrics": self.train_metrics,
            "feature_importances": self.feature_importances,
            "learner_path": str(learner_path.name),
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"Saved fusion ensemble metadata to {meta_path}")
        logger.info(f"Saved fusion ensemble learner to {learner_path}")

    def load(self, model_path: str | Path) -> None:
        """Load the fusion ensemble meta-learner and metadata."""
        model_path = Path(model_path)
        meta_path = model_path.with_suffix(".json")

        if not meta_path.exists():
            raise FileNotFoundError(f"Fusion ensemble metadata not found at {meta_path}")

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        self.meta_learner_type = payload["meta_learner_type"]
        self.population_means = payload["population_means"]
        self.single_modality_metrics = payload.get("single_modality_metrics", {})
        self.validation_metrics = payload.get("validation_metrics", {})
        self.train_metrics = payload.get("train_metrics", {})
        self.feature_importances = payload.get("feature_importances")

        learner_filename = payload.get("learner_path", f"{model_path.stem}_learner.json")
        learner_path = model_path.parent / learner_filename

        self.meta_learner = self._create_meta_learner()
        if hasattr(self.meta_learner, "load_model"):
            self.meta_learner.load_model(learner_path)
        else:
            import joblib
            self.meta_learner = joblib.load(learner_path)

        logger.info(f"Loaded fusion ensemble from {model_path}")


class FusionPredictor:
    """Convenience wrapper for loading and using a trained FusionEnsemble."""

    def __init__(self, model_path: Optional[str | Path] = None):
        self.ensemble = FusionEnsemble()

        if model_path is None:
            model_path = self._resolve_latest_model()

        if model_path is None:
            raise FileNotFoundError("No trained fusion model found.")

        self.ensemble.load(model_path)
        self.model_path = Path(model_path)

    def _resolve_latest_model(self) -> Optional[Path]:
        if DEFAULT_MODEL_LATEST_PATH.exists() or DEFAULT_MODEL_LATEST_PATH.is_symlink():
            return DEFAULT_MODEL_LATEST_PATH

        candidates = sorted(MODELS_DIR.glob("fusion_model_*.json"))
        return candidates[-1] if candidates else None

    def predict(self, text_prob: float, tabular_prob: float, behavioral_prob: float) -> float:
        return self.ensemble.predict(text_prob, tabular_prob, behavioral_prob)

    def predict_with_missing(
        self,
        text_prob: Optional[float] = None,
        tabular_prob: Optional[float] = None,
        behavioral_prob: Optional[float] = None,
    ) -> float:
        return self.ensemble.predict_with_missing(text_prob, tabular_prob, behavioral_prob)

    def predict_batch(self, probabilities: np.ndarray) -> np.ndarray:
        return self.ensemble.predict_batch(probabilities)


def _persist_latest_pointer(model_path: Path, latest_path: Path) -> Path:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()

    try:
        latest_path.symlink_to(model_path.name)
    except OSError:
        shutil.copy2(model_path, latest_path)
    return latest_path


def _write_evidence(payload: Dict[str, Any], evidence_path: Path) -> Path:
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return evidence_path


def train_fusion_ensemble(
    text_features_path: Path = DEFAULT_TEXT_FEATURES_PATH,
    text_student_ids_path: Path = DEFAULT_TEXT_STUDENT_IDS_PATH,
    tabular_features_path: Path = DEFAULT_TABULAR_FEATURES_PATH,
    tabular_student_ids_path: Path = DEFAULT_TABULAR_STUDENT_IDS_PATH,
    behavioral_sequences_path: Path = DEFAULT_BEHAVIORAL_SEQUENCES_PATH,
    behavioral_student_ids_path: Path = DEFAULT_BEHAVIORAL_STUDENT_IDS_PATH,
    students_csv: Path = DEFAULT_STUDENTS_CSV,
    models_dir: Path = MODELS_DIR,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    meta_learner_type: str = "xgboost",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and save a fusion ensemble model.

    Returns:
        Dictionary with training results, metrics, and file paths.
    """
    ensemble = FusionEnsemble(meta_learner_type=meta_learner_type, random_state=random_state)

    result = ensemble.train(
        text_features_path=text_features_path,
        text_student_ids_path=text_student_ids_path,
        tabular_features_path=tabular_features_path,
        tabular_student_ids_path=tabular_student_ids_path,
        behavioral_sequences_path=behavioral_sequences_path,
        behavioral_student_ids_path=behavioral_student_ids_path,
        students_csv=students_csv,
    )

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"fusion_model_{timestamp}.json"
    ensemble.save(model_path)

    latest_path = _persist_latest_pointer(model_path, models_dir / "fusion_model_latest.json")

    single_inference_ms = _measure_single_inference(ensemble)

    result.update({
        "model_path": str(model_path),
        "latest_model_path": str(latest_path),
        "single_inference_ms": single_inference_ms,
    })

    evidence_payload = {
        "model_path": str(model_path),
        "latest_model_path": str(latest_path),
        "training_time_seconds": result["training_time_seconds"],
        "single_inference_ms": single_inference_ms,
        "validation_metrics": result["validation_metrics"],
        "test_metrics": result["test_metrics"],
        "single_modality_metrics": result["single_modality_metrics"],
        "population_means": result["population_means"],
        "feature_importances": result["feature_importances"],
        "meta_learner_type": meta_learner_type,
    }
    result["evidence_path"] = str(_write_evidence(evidence_payload, evidence_path))

    logger.info(f"Fusion ensemble saved to {model_path}")
    logger.info(f"Test F1: {result['test_metrics']['f1']:.4f}")
    return result


def _measure_single_inference(ensemble: FusionEnsemble) -> float:
    start = time.perf_counter()
    ensemble.predict(0.5, 0.5, 0.5)
    return (time.perf_counter() - start) * 1000


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = train_fusion_ensemble()
    print("\nTraining complete.")
    print(f"Test F1: {result['test_metrics']['f1']:.4f}")
    print(f"Validation F1: {result['validation_metrics']['f1']:.4f}")
    print(f"Training time (s): {result['training_time_seconds']:.3f}")
    print(f"Single inference (ms): {result['single_inference_ms']:.3f}")
    print(f"Model saved to: {result['model_path']}")
