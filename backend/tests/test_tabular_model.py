# pyright: reportAny=false, reportUnknownArgumentType=false

from pathlib import Path

import numpy as np

from app.models.tabular_model import TabularPredictor, train_tabular_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "tabular_features.npy"


def test_model_training_runs_without_errors(tmp_path: Path):
    models_dir = tmp_path / "models"
    plot_path = tmp_path / "tabular_feature_importance.png"
    evidence_path = tmp_path / "task-11-tabular-model-metrics.json"

    result = train_tabular_model(
        models_dir=models_dir,
        feature_importance_path=plot_path,
        evidence_path=evidence_path,
    )

    assert result["validation_metrics"]["f1"] > 0.65
    assert result["training_time_seconds"] < 300
    assert Path(result["model_path"]).exists()
    assert Path(result["latest_model_path"]).exists()
    assert Path(result["feature_importance_path"]).exists()
    assert Path(result["evidence_path"]).exists()


def test_predictor_returns_probability(tmp_path: Path):
    result = train_tabular_model(
        models_dir=tmp_path / "models",
        feature_importance_path=tmp_path / "plot.png",
        evidence_path=tmp_path / "metrics.json",
    )
    predictor = TabularPredictor(model_path=result["latest_model_path"])

    features = np.load(FEATURES_PATH)
    probability = predictor.predict(features[0])

    assert 0.0 <= probability <= 1.0
    assert result["single_inference_ms"] < 50


def test_predictor_loads_from_symlink(tmp_path: Path):
    result = train_tabular_model(
        models_dir=tmp_path / "models",
        feature_importance_path=tmp_path / "plot.png",
        evidence_path=tmp_path / "metrics.json",
    )

    predictor = TabularPredictor(model_path=result["latest_model_path"])
    features = np.load(FEATURES_PATH)

    probability = predictor.predict(features[1])

    assert 0.0 <= probability <= 1.0


def test_batch_prediction(tmp_path: Path):
    result = train_tabular_model(
        models_dir=tmp_path / "models",
        feature_importance_path=tmp_path / "plot.png",
        evidence_path=tmp_path / "metrics.json",
    )
    predictor = TabularPredictor(model_path=result["latest_model_path"])

    features = np.load(FEATURES_PATH)[:5]
    probabilities = predictor.predict_batch(features)

    assert probabilities.shape == (5,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
