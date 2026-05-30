from pathlib import Path

import numpy as np
import pytest

from app.models.fusion_ensemble import FusionEnsemble, FusionPredictor, train_fusion_ensemble

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_ensemble_training_runs_without_errors(tmp_path: Path):
    models_dir = tmp_path / "models"
    evidence_path = tmp_path / "task-14-fusion-metrics.json"

    result = train_fusion_ensemble(
        models_dir=models_dir,
        evidence_path=evidence_path,
    )

    assert result["test_metrics"]["f1"] > 0.70
    assert result["training_time_seconds"] < 120
    assert Path(result["model_path"]).exists()
    assert Path(result["latest_model_path"]).exists()
    assert Path(result["evidence_path"]).exists()


def test_ensemble_training_with_logistic_regression(tmp_path: Path):
    models_dir = tmp_path / "models"
    evidence_path = tmp_path / "task-14-fusion-metrics.json"

    result = train_fusion_ensemble(
        models_dir=models_dir,
        evidence_path=evidence_path,
        meta_learner_type="logistic",
    )

    assert result["test_metrics"]["f1"] > 0.70
    assert result["training_time_seconds"] < 120


def test_prediction_returns_probability(tmp_path: Path):
    result = train_fusion_ensemble(
        models_dir=tmp_path / "models",
        evidence_path=tmp_path / "metrics.json",
    )

    predictor = FusionPredictor(model_path=result["latest_model_path"])
    probability = predictor.predict(0.5, 0.5, 0.5)

    assert 0.0 <= probability <= 1.0
    assert result["single_inference_ms"] < 100


def test_predict_with_missing_modalities(tmp_path: Path):
    result = train_fusion_ensemble(
        models_dir=tmp_path / "models",
        evidence_path=tmp_path / "metrics.json",
    )

    predictor = FusionPredictor(model_path=result["latest_model_path"])

    prob_all = predictor.predict_with_missing(0.5, 0.5, 0.5)
    prob_missing_text = predictor.predict_with_missing(None, 0.5, 0.5)
    prob_missing_tabular = predictor.predict_with_missing(0.5, None, 0.5)
    prob_missing_behavioral = predictor.predict_with_missing(0.5, 0.5, None)
    prob_missing_two = predictor.predict_with_missing(None, None, 0.5)
    prob_all_missing = predictor.predict_with_missing(None, None, None)

    assert 0.0 <= prob_all <= 1.0
    assert 0.0 <= prob_missing_text <= 1.0
    assert 0.0 <= prob_missing_tabular <= 1.0
    assert 0.0 <= prob_missing_behavioral <= 1.0
    assert 0.0 <= prob_missing_two <= 1.0
    assert 0.0 <= prob_all_missing <= 1.0


def test_batch_prediction(tmp_path: Path):
    result = train_fusion_ensemble(
        models_dir=tmp_path / "models",
        evidence_path=tmp_path / "metrics.json",
    )

    predictor = FusionPredictor(model_path=result["latest_model_path"])

    probs = np.array([
        [0.5, 0.5, 0.5],
        [0.2, 0.3, 0.4],
        [0.8, 0.9, 0.7],
    ], dtype=np.float32)

    probabilities = predictor.predict_batch(probs)

    assert probabilities.shape == (3,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_ensemble_outperforms_or_matches_single_modalities(tmp_path: Path):
    result = train_fusion_ensemble(
        models_dir=tmp_path / "models",
        evidence_path=tmp_path / "metrics.json",
    )

    ensemble_f1 = result["test_metrics"]["f1"]
    best_single_f1 = result["best_single_modality_f1"]

    assert ensemble_f1 >= best_single_f1 - 0.05


def test_model_save_and_load(tmp_path: Path):
    models_dir = tmp_path / "models"
    evidence_path = tmp_path / "metrics.json"

    result = train_fusion_ensemble(
        models_dir=models_dir,
        evidence_path=evidence_path,
    )

    original_predictor = FusionPredictor(model_path=result["latest_model_path"])
    original_prob = original_predictor.predict(0.3, 0.6, 0.4)

    loaded_predictor = FusionPredictor(model_path=result["model_path"])
    loaded_prob = loaded_predictor.predict(0.3, 0.6, 0.4)

    assert pytest.approx(original_prob, rel=1e-5) == loaded_prob


def test_ensemble_population_means_set_correctly(tmp_path: Path):
    result = train_fusion_ensemble(
        models_dir=tmp_path / "models",
        evidence_path=tmp_path / "metrics.json",
    )

    predictor = FusionPredictor(model_path=result["latest_model_path"])

    assert "text" in predictor.ensemble.population_means
    assert "tabular" in predictor.ensemble.population_means
    assert "behavioral" in predictor.ensemble.population_means

    for key, value in predictor.ensemble.population_means.items():
        assert 0.0 <= value <= 1.0


def test_feature_importances_present(tmp_path: Path):
    result = train_fusion_ensemble(
        models_dir=tmp_path / "models",
        evidence_path=tmp_path / "metrics.json",
    )

    predictor = FusionPredictor(model_path=result["latest_model_path"])

    assert predictor.ensemble.feature_importances is not None
    assert "text_prob" in predictor.ensemble.feature_importances
    assert "tabular_prob" in predictor.ensemble.feature_importances
    assert "behavioral_prob" in predictor.ensemble.feature_importances
