"""Explainability service that orchestrates SHAP and NLG components."""

from pathlib import Path
from typing import Any

import numpy as np

from app.explainability.nlg_summarizer import summarize
from app.explainability.shap_explainer import FEATURE_NAMES, TabularShapExplainer

from app.core.paths import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "tabular_model_latest.json"


class ExplainabilityService:
    """High-level service for generating student risk explanations."""

    def __init__(self, model_path: str | Path | None = None):
        resolved_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.shap_explainer = TabularShapExplainer(model_path=resolved_path)

    def explain(
        self,
        student_features: np.ndarray | list[float],
        feature_values: dict[str, float] | None = None,
        text_prob: float | None = None,
        tabular_prob: float | None = None,
        behavioral_prob: float | None = None,
    ) -> dict[str, Any]:
        """Generate a full explanation for a student's risk prediction.

        Args:
            student_features: Array of tabular features (must match FEATURE_NAMES order).
            feature_values: Optional mapping of feature names to raw/unscaled values for NLG.
            text_prob: Optional text model probability for modality contributions.
            tabular_prob: Optional tabular model probability for modality contributions.
            behavioral_prob: Optional behavioral model probability for modality contributions.

        Returns:
            Dictionary with risk_level, probability, top_factors, modality_contributions, narrative_summary.
        """
        features = np.asarray(student_features, dtype=np.float32)
        shap_result = self.shap_explainer.explain(features)

        if feature_values is None:
            feature_values = {
                name: float(val) for name, val in zip(FEATURE_NAMES, features.flatten())
            }

        return summarize(
            shap_result=shap_result,
            feature_values=feature_values,
            text_prob=text_prob,
            tabular_prob=tabular_prob,
            behavioral_prob=behavioral_prob,
        )
