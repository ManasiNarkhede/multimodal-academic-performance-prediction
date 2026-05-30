from pathlib import Path

import numpy as np
import pytest

from app.explainability.nlg_summarizer import RISK_THRESHOLDS, _get_risk_level, summarize
from app.explainability.service import ExplainabilityService
from app.explainability.shap_explainer import FEATURE_NAMES, TabularShapExplainer
from app.models.tabular_model import train_tabular_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "tabular_features.npy"


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    result = train_tabular_model(
        models_dir=tmp_path / "models",
        feature_importance_path=tmp_path / "plot.png",
        evidence_path=tmp_path / "metrics.json",
    )
    return Path(result["latest_model_path"])


@pytest.fixture
def sample_features() -> np.ndarray:
    features = np.load(FEATURES_PATH)
    return features[0]


def test_shap_explainer_initialization(trained_model_path: Path):
    explainer = TabularShapExplainer(model_path=trained_model_path)
    assert explainer.feature_names == FEATURE_NAMES
    assert explainer.explainer is not None


def test_shap_explainer_generates_values(trained_model_path: Path, sample_features: np.ndarray):
    explainer = TabularShapExplainer(model_path=trained_model_path)
    result = explainer.explain(sample_features)

    assert "shap_values" in result
    assert "base_value" in result
    assert "probability" in result
    assert "top_positive" in result
    assert "top_negative" in result

    assert len(result["shap_values"]) == len(FEATURE_NAMES)
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["base_value"], float)


def test_shap_explainer_top_factors(trained_model_path: Path, sample_features: np.ndarray):
    explainer = TabularShapExplainer(model_path=trained_model_path)
    result = explainer.explain(sample_features)

    for factor in result["top_positive"]:
        assert "feature" in factor
        assert "shap_value" in factor
        assert factor["shap_value"] > 0
        assert factor["feature"] in FEATURE_NAMES

    for factor in result["top_negative"]:
        assert "feature" in factor
        assert "shap_value" in factor
        assert factor["shap_value"] < 0
        assert factor["feature"] in FEATURE_NAMES


def test_shap_explainer_batch_shape(trained_model_path: Path, sample_features: np.ndarray):
    explainer = TabularShapExplainer(model_path=trained_model_path)
    result = explainer.explain(sample_features)
    assert len(result["shap_values"]) == len(FEATURE_NAMES)


def test_shap_explainer_invalid_feature_count(trained_model_path: Path):
    explainer = TabularShapExplainer(model_path=trained_model_path)
    with pytest.raises(ValueError, match="Expected"):
        explainer.explain(np.array([1.0, 2.0, 3.0]))


def test_nlg_risk_levels():
    assert _get_risk_level(0.75) == "high"
    assert _get_risk_level(0.70) == "high"
    assert _get_risk_level(0.50) == "medium"
    assert _get_risk_level(0.40) == "medium"
    assert _get_risk_level(0.35) == "low"
    assert _get_risk_level(0.0) == "low"


def test_nlg_summarize_structure(trained_model_path: Path, sample_features: np.ndarray):
    explainer = TabularShapExplainer(model_path=trained_model_path)
    shap_result = explainer.explain(sample_features)

    summary = summarize(shap_result)

    assert "risk_level" in summary
    assert "probability" in summary
    assert "top_factors" in summary
    assert "modality_contributions" in summary
    assert "narrative_summary" in summary

    assert summary["risk_level"] in ("high", "medium", "low")
    assert 0.0 <= summary["probability"] <= 1.0
    assert isinstance(summary["narrative_summary"], str)
    assert len(summary["narrative_summary"]) <= 200


def test_nlg_summarize_with_modality_probs(trained_model_path: Path, sample_features: np.ndarray):
    explainer = TabularShapExplainer(model_path=trained_model_path)
    shap_result = explainer.explain(sample_features)

    summary = summarize(
        shap_result,
        text_prob=0.8,
        tabular_prob=0.6,
        behavioral_prob=0.3,
    )

    assert "Text analysis suggests negative sentiment." in summary["modality_contributions"]
    assert "Tabular data shows poor academic indicators." in summary["modality_contributions"]
    assert "Behavioral data indicates healthy engagement." in summary["modality_contributions"]


def test_nlg_summarize_missing_modality_probs(trained_model_path: Path, sample_features: np.ndarray):
    explainer = TabularShapExplainer(model_path=trained_model_path)
    shap_result = explainer.explain(sample_features)

    summary = summarize(shap_result, text_prob=None, tabular_prob=None, behavioral_prob=None)
    assert summary["modality_contributions"] == "No modality-specific contributions available."


def test_nlg_summarize_with_feature_values(trained_model_path: Path, sample_features: np.ndarray):
    explainer = TabularShapExplainer(model_path=trained_model_path)
    shap_result = explainer.explain(sample_features)

    feature_values = {name: float(val) for name, val in zip(FEATURE_NAMES, sample_features)}
    summary = summarize(shap_result, feature_values=feature_values)

    assert len(summary["top_factors"]) > 0
    for factor in summary["top_factors"]:
        assert "description" in factor
        assert isinstance(factor["description"], str)


def test_service_integration(trained_model_path: Path, sample_features: np.ndarray):
    service = ExplainabilityService(model_path=trained_model_path)
    result = service.explain(sample_features)

    assert "risk_level" in result
    assert "probability" in result
    assert "top_factors" in result
    assert "modality_contributions" in result
    assert "narrative_summary" in result

    assert result["risk_level"] in ("high", "medium", "low")
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["narrative_summary"], str)
    assert len(result["narrative_summary"]) <= 200


def test_service_with_modality_probs(trained_model_path: Path, sample_features: np.ndarray):
    service = ExplainabilityService(model_path=trained_model_path)
    result = service.explain(
        sample_features,
        text_prob=0.8,
        tabular_prob=0.6,
        behavioral_prob=0.3,
    )

    assert "Text analysis suggests negative sentiment." in result["modality_contributions"]


def test_service_generates_feature_values_fallback(trained_model_path: Path, sample_features: np.ndarray):
    service = ExplainabilityService(model_path=trained_model_path)
    result = service.explain(sample_features)

    for factor in result["top_factors"]:
        assert "description" in factor
        assert isinstance(factor["description"], str)


def test_service_explanation_under_200_words(trained_model_path: Path, sample_features: np.ndarray):
    service = ExplainabilityService(model_path=trained_model_path)
    result = service.explain(sample_features)
    word_count = len(result["narrative_summary"].split())
    assert word_count <= 200
