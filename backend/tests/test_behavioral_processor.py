# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from app.preprocessing.behavioral_processor import (
    FEATURE_COLUMNS,
    aggregate_daily,
    create_sequences,
    load_behavioral_logs,
    process_all_behavioral,
    standardize_sequences,
)


def build_logs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": [1, 1, 1, 1, 2, 2, 2],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-01 09:00:00",
                    "2024-01-01 10:00:00",
                    "2024-01-02 11:00:00",
                    "2024-01-01 12:00:00",
                    "2024-01-03 13:00:00",
                    "2024-01-04 14:00:00",
                ]
            ),
            "event_type": [
                "login",
                "page_view",
                "assignment_open",
                "forum_post",
                "assignment_submit",
                "login",
                "page_view",
            ],
            "duration_seconds": [30, 20, 40, 50, 60, 70, 80],
        }
    )


def test_load_behavioral_logs_parses_timestamps(tmp_path: Path):
    csv_path = tmp_path / "behavioral_logs.csv"
    build_logs_df().to_csv(csv_path, index=False)

    logs_df = load_behavioral_logs(csv_path)

    assert pd.api.types.is_datetime64_any_dtype(logs_df["timestamp"])


def test_daily_aggregation_produces_expected_features():
    daily_agg = aggregate_daily(build_logs_df())

    day_one = daily_agg[(daily_agg["student_id"] == 1) & (daily_agg["date"] == pd.Timestamp("2024-01-01"))].iloc[0]
    assert list(daily_agg.columns) == ["student_id", "date", *FEATURE_COLUMNS]
    assert day_one["daily_login_count"] == 1
    assert day_one["daily_total_duration"] == 90
    assert day_one["daily_assignment_opens"] == 1
    assert day_one["daily_assignment_submissions"] == 0
    assert day_one["daily_forum_posts"] == 0
    assert day_one["daily_unique_pages"] == 1


def test_create_sequences_returns_fixed_shape_and_padding():
    daily_agg = aggregate_daily(build_logs_df())

    sequences, student_ids = create_sequences(daily_agg, seq_length=4)

    assert sequences.shape == (2, 4, 6)
    np.testing.assert_array_equal(student_ids, np.array([1, 2]))
    np.testing.assert_array_equal(sequences[0, 0], np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(sequences[0, -2], np.array([1, 90, 1, 0, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(sequences[0, -1], np.array([0, 50, 0, 0, 1, 0], dtype=np.float32))


def test_create_sequences_truncates_to_last_n_days():
    daily_agg = pd.DataFrame(
        {
            "student_id": [1, 1, 1],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "daily_login_count": [1, 2, 3],
            "daily_total_duration": [10, 20, 30],
            "daily_assignment_opens": [0, 0, 1],
            "daily_assignment_submissions": [0, 1, 1],
            "daily_forum_posts": [0, 0, 1],
            "daily_unique_pages": [1, 2, 3],
        }
    )

    sequences, _ = create_sequences(daily_agg, seq_length=2)

    np.testing.assert_array_equal(
        sequences[0],
        np.array(
            [
                [2, 20, 0, 1, 0, 2],
                [3, 30, 1, 1, 1, 3],
            ],
            dtype=np.float32,
        ),
    )


def test_standardize_sequences_has_no_nan_and_persists_scaler(tmp_path: Path):
    sequences = np.array(
        [
            [[0, 0, 0, 0, 0, 0], [1, 10, 0, 0, 0, 1]],
            [[0, 0, 0, 0, 0, 0], [2, 20, 1, 1, 1, 2]],
        ],
        dtype=np.float32,
    )
    scaler_path = tmp_path / "behavioral_scaler.pkl"

    standardized, scaler = standardize_sequences(sequences, fit_scaler=True, scaler_path=scaler_path)
    reloaded, loaded_scaler = standardize_sequences(sequences, fit_scaler=False, scaler_path=scaler_path)

    assert not np.isnan(standardized).any()
    np.testing.assert_array_equal(standardized[:, 0, :], np.zeros((2, 6), dtype=np.float32))
    np.testing.assert_allclose(standardized, reloaded)
    assert scaler_path.exists()
    assert scaler.mean_.shape == (6,)
    assert loaded_scaler.mean_.shape == (6,)
    with scaler_path.open("rb") as scaler_file:
        persisted_scaler = pickle.load(scaler_file)
    assert persisted_scaler.mean_.shape == (6,)


def test_process_all_behavioral_saves_outputs(tmp_path: Path):
    logs_csv = tmp_path / "behavioral_logs.csv"
    build_logs_df().to_csv(logs_csv, index=False)

    sequences_path = tmp_path / "behavioral_sequences.npy"
    student_ids_path = tmp_path / "behavioral_student_ids.npy"
    scaler_path = tmp_path / "behavioral_scaler.pkl"

    sequences, student_ids = process_all_behavioral(
        logs_csv=logs_csv,
        sequences_path=sequences_path,
        student_ids_path=student_ids_path,
        scaler_path=scaler_path,
        seq_length=30,
    )

    assert sequences.shape == (2, 30, 6)
    np.testing.assert_array_equal(student_ids, np.array([1, 2]))
    assert not np.isnan(sequences).any()
    assert sequences_path.exists()
    assert student_ids_path.exists()
    assert scaler_path.exists()
    np.testing.assert_array_equal(np.load(student_ids_path), student_ids)
    assert np.load(sequences_path).shape == (2, 30, 6)
