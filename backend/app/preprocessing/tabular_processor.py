import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_STUDENTS_CSV = RAW_DATA_DIR / "students.csv"
DEFAULT_FEATURES_PATH = PROCESSED_DATA_DIR / "tabular_features.npy"
DEFAULT_STUDENT_IDS_PATH = PROCESSED_DATA_DIR / "tabular_student_ids.npy"
DEFAULT_SCALER_PATH = PROCESSED_DATA_DIR / "tabular_scaler.pkl"

GENDER_CATEGORIES = ["F", "M", "NB"]
SOCIOECONOMIC_CATEGORIES = ["high", "low", "medium"]
NUMERIC_FEATURE_COLUMNS = [
    "attendance_rate",
    "avg_assignment_score",
    "assignment_score_std",
    "internal_exam_score",
    "study_hours_per_week",
    "extracurricular_count",
    "prior_gpa",
]


def load_tabular_data(csv_path: str | Path = DEFAULT_STUDENTS_CSV) -> pd.DataFrame:
    """Load student tabular data from CSV."""
    return pd.read_csv(csv_path)


def _parse_assignment_scores(raw_scores: object) -> list[float]:
    if pd.isna(raw_scores):
        return []

    score_values: list[float] = []
    for value in str(raw_scores).split("|"):
        stripped = value.strip()
        if not stripped:
            continue
        score_values.append(float(stripped))
    return score_values


def engineer_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Engineer tabular features without using label leakage columns."""
    working_df = df.copy()

    assignment_lists = working_df["assignment_scores"].apply(_parse_assignment_scores)
    working_df["attendance_rate"] = pd.to_numeric(
        working_df.get("attendance_rate", working_df.get("attendance")),
        errors="coerce",
    )
    working_df["avg_assignment_score"] = assignment_lists.apply(
        lambda scores: float(np.mean(scores)) if scores else np.nan
    )
    working_df["assignment_score_std"] = assignment_lists.apply(
        lambda scores: float(np.std(scores)) if scores else np.nan
    )
    working_df["internal_exam_score"] = pd.to_numeric(working_df["internal_exam_score"], errors="coerce")
    working_df["study_hours_per_week"] = pd.to_numeric(working_df["study_hours_per_week"], errors="coerce")
    working_df["extracurricular_count"] = pd.to_numeric(working_df["extracurricular_count"], errors="coerce")
    working_df["prior_gpa"] = pd.to_numeric(working_df["prior_gpa"], errors="coerce")

    gender_encoded = pd.get_dummies(
        working_df["gender"].fillna("unknown"),
        prefix="gender",
    )
    gender_encoded = gender_encoded.reindex(
        columns=[f"gender_{category}" for category in GENDER_CATEGORIES],
        fill_value=0,
    )

    socioeconomic_encoded = pd.get_dummies(
        working_df["socioeconomic_status"].fillna("unknown"),
        prefix="socioeconomic_status",
    )
    socioeconomic_encoded = socioeconomic_encoded.reindex(
        columns=[f"socioeconomic_status_{category}" for category in SOCIOECONOMIC_CATEGORIES],
        fill_value=0,
    )

    feature_frame = pd.concat(
        [working_df[NUMERIC_FEATURE_COLUMNS], gender_encoded, socioeconomic_encoded],
        axis=1,
    )
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")

    return feature_frame.to_numpy(dtype=np.float32), feature_frame.columns.tolist()


def preprocess_features(
    features: np.ndarray,
    fit_scaler: bool = True,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> tuple[np.ndarray, StandardScaler]:
    """Fill missing values, standardize features, and persist/load scaler."""
    scaler_path = Path(scaler_path)
    feature_frame = pd.DataFrame(features)
    medians = feature_frame.median(numeric_only=True)
    feature_frame = feature_frame.fillna(medians)
    feature_frame = feature_frame.fillna(0.0)
    dense_features = feature_frame.to_numpy(dtype=np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(dense_features)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with scaler_path.open("wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
    else:
        with scaler_path.open("rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        scaled_features = scaler.transform(dense_features)

    return scaled_features.astype(np.float32), scaler


def process_all_tabular(
    csv_path: str | Path = DEFAULT_STUDENTS_CSV,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    student_ids_path: str | Path = DEFAULT_STUDENT_IDS_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run full tabular preprocessing pipeline and save artifacts."""
    logger.info("Loading tabular student data...")
    students_df = load_tabular_data(csv_path)

    logger.info("Engineering tabular features...")
    features, feature_names = engineer_features(students_df)

    logger.info("Preprocessing tabular features...")
    scaled_features, _ = preprocess_features(features, fit_scaler=True, scaler_path=scaler_path)
    student_ids = students_df["student_id"].to_numpy(dtype=np.int64)

    features_path = Path(features_path)
    student_ids_path = Path(student_ids_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(features_path, scaled_features)
    np.save(student_ids_path, student_ids)

    logger.info("Saved tabular features to %s with shape %s", features_path, scaled_features.shape)
    return scaled_features, student_ids, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    feature_matrix, student_ids, feature_names = process_all_tabular()
    print(f"Tabular features shape: {feature_matrix.shape}")
    print(f"Tabular student IDs shape: {student_ids.shape}")
    print(f"Tabular feature count: {len(feature_names)}")
