# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportAttributeAccessIssue=false

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from app.preprocessing.tabular_processor import (
    engineer_features,
    load_tabular_data,
    preprocess_features,
    process_all_tabular,
)


def build_students_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": [101, 102, 103],
            "name": ["Ada", "Ben", "Cy"],
            "age": [20, 21, 22],
            "gender": ["F", "M", "NB"],
            "socioeconomic_status": ["high", "low", "medium"],
            "prior_gpa": [3.6, np.nan, 2.7],
            "cohort_id": [1, 1, 2],
            "attendance": [90.0, 75.0, 88.0],
            "assignment_scores": ["80|90|100", "60|70", np.nan],
            "n_assignments": [3, 2, 0],
            "internal_exam_score": [85.0, 70.0, np.nan],
            "study_hours_per_week": [14.0, 9.0, 11.0],
            "extracurricular_count": [2, 1, np.nan],
            "at_risk": [0, 1, 0],
            "risk_score": [0.1, 0.8, 0.2],
            "split": ["train", "train", "val"],
        }
    )


def test_load_tabular_data_reads_csv(tmp_path: Path):
    csv_path = tmp_path / "students.csv"
    build_students_df().to_csv(csv_path, index=False)

    loaded_df = load_tabular_data(csv_path)

    assert loaded_df.shape == (3, 16)
    assert loaded_df.loc[0, "student_id"] == 101


def test_engineer_features_produces_expected_columns_and_count():
    features, feature_names = engineer_features(build_students_df())

    assert features.shape == (3, len(feature_names))
    assert len(feature_names) >= 8
    assert "attendance_rate" in feature_names
    assert "avg_assignment_score" in feature_names
    assert "assignment_score_std" in feature_names
    assert "gender_F" in feature_names
    assert "socioeconomic_status_medium" in feature_names


def test_preprocess_features_has_no_nan_and_persists_scaler(tmp_path: Path):
    features, _ = engineer_features(build_students_df())
    scaler_path = tmp_path / "tabular_scaler.pkl"

    scaled_features, scaler = preprocess_features(features, fit_scaler=True, scaler_path=scaler_path)
    reloaded_features, loaded_scaler = preprocess_features(features, fit_scaler=False, scaler_path=scaler_path)

    assert not np.isnan(scaled_features).any()
    np.testing.assert_allclose(scaled_features, reloaded_features)
    assert scaler_path.exists()
    assert scaler.mean_.shape[0] == features.shape[1]
    assert loaded_scaler.scale_.shape[0] == features.shape[1]
    with scaler_path.open("rb") as scaler_file:
        persisted_scaler = pickle.load(scaler_file)
    assert persisted_scaler.scale_.shape[0] == features.shape[1]


def test_process_all_tabular_saves_outputs_and_matches_student_count(tmp_path: Path):
    csv_path = tmp_path / "students.csv"
    build_students_df().to_csv(csv_path, index=False)

    features_path = tmp_path / "tabular_features.npy"
    student_ids_path = tmp_path / "tabular_student_ids.npy"
    scaler_path = tmp_path / "tabular_scaler.pkl"

    scaled_features, student_ids, feature_names = process_all_tabular(
        csv_path=csv_path,
        features_path=features_path,
        student_ids_path=student_ids_path,
        scaler_path=scaler_path,
    )

    assert scaled_features.shape == (3, len(feature_names))
    assert scaled_features.shape[0] == len(student_ids)
    assert len(feature_names) >= 8
    assert not np.isnan(scaled_features).any()
    np.testing.assert_array_equal(student_ids, np.array([101, 102, 103]))
    assert features_path.exists()
    assert student_ids_path.exists()
    assert scaler_path.exists()
    assert np.load(features_path).shape == scaled_features.shape
    np.testing.assert_array_equal(np.load(student_ids_path), student_ids)
