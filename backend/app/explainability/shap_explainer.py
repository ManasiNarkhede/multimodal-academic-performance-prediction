"""SHAP explainer for tabular XGBoost model."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import shap

from app.models.tabular_model import TabularPredictor

logger = logging.getLogger(__name__)

from app.core.paths import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "tabular_model_latest.json"

FEATURE_NAMES = [
    "attendance_rate",
    "avg_assignment_score",
    "assignment_score_std",
    "internal_exam_score",
    "study_hours_per_week",
    "extracurricular_count",
    "prior_gpa",
    "gender_F",
    "gender_M",
    "gender_NB",
    "socioeconomic_status_high",
    "socioeconomic_status_low",
    "socioeconomic_status_medium",
]


class TabularShapExplainer:
    """Generates SHAP explanations for the tabular XGBoost model."""

    def __init__(self, model_path: str | Path | None = None):
        resolved_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model not found at {resolved_path}")

        self.predictor = TabularPredictor(model_path=resolved_path)
        self.explainer = shap.TreeExplainer(self.predictor.model)
        self.feature_names = FEATURE_NAMES

    def explain(self, features: np.ndarray) -> dict[str, Any]:
        """Generate SHAP values and extract top positive/negative drivers.

        Args:
            features: Array of shape (n_features,) or (1, n_features).

        Returns:
            Dictionary with shap_values, base_value, probability, top_positive, top_negative.
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {features.shape[1]}"
            )

        shap_values = self.explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (risk) SHAP values

        shap_values = np.asarray(shap_values).flatten()
        base_value = float(self.explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])

        probability = self.predictor.predict(features[0])

        feature_importance = list(zip(self.feature_names, shap_values))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        top_positive = [
            {"feature": name, "shap_value": float(value)}
            for name, value in feature_importance[:3]
            if value > 0
        ]
        top_negative = [
            {"feature": name, "shap_value": float(value)}
            for name, value in feature_importance[-3:]
            if value < 0
        ]
        top_negative.reverse()  # Most negative first

        return {
            "shap_values": shap_values.tolist(),
            "base_value": base_value,
            "probability": probability,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }
