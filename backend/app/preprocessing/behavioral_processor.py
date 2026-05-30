# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportAny=false, reportUnusedCallResult=false, reportDeprecated=false

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

DEFAULT_BEHAVIORAL_LOGS_CSV = RAW_DATA_DIR / "behavioral_logs.csv"
DEFAULT_SEQUENCES_PATH = PROCESSED_DATA_DIR / "behavioral_sequences.npy"
DEFAULT_STUDENT_IDS_PATH = PROCESSED_DATA_DIR / "behavioral_student_ids.npy"
DEFAULT_SCALER_PATH = PROCESSED_DATA_DIR / "behavioral_scaler.pkl"

FEATURE_COLUMNS = [
    "daily_login_count",
    "daily_total_duration",
    "daily_assignment_opens",
    "daily_assignment_submissions",
    "daily_forum_posts",
    "daily_unique_pages",
]


def load_behavioral_logs(csv_path: str | Path = DEFAULT_BEHAVIORAL_LOGS_CSV) -> pd.DataFrame:
    """Load behavioral logs CSV with parsed timestamps."""
    logs_df = pd.read_csv(csv_path)
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
    return logs_df


def aggregate_daily(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate behavioral logs into daily student features."""
    working_df = logs_df.copy()
    working_df["date"] = working_df["timestamp"].dt.normalize()

    daily_agg = (
        working_df.groupby(["student_id", "date"])
        .agg(
            daily_login_count=("event_type", lambda s: int((s == "login").sum())),
            daily_total_duration=("duration_seconds", "sum"),
            daily_assignment_opens=("event_type", lambda s: int((s == "assignment_open").sum())),
            daily_assignment_submissions=("event_type", lambda s: int((s == "assignment_submit").sum())),
            daily_forum_posts=("event_type", lambda s: int((s == "forum_post").sum())),
            daily_unique_pages=("event_type", lambda s: int((s == "page_view").sum())),
        )
        .reset_index()
        .sort_values(["student_id", "date"])
    )

    daily_agg[FEATURE_COLUMNS] = daily_agg[FEATURE_COLUMNS].astype(np.float32)
    return daily_agg


def create_sequences(
    daily_agg: pd.DataFrame,
    seq_length: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length per-student sequences from daily aggregates."""
    student_ids = np.asarray(daily_agg["student_id"].unique(), dtype=np.int64)
    student_ids.sort()
    sequences = np.zeros((len(student_ids), seq_length, len(FEATURE_COLUMNS)), dtype=np.float32)

    for idx, student_id in enumerate(student_ids):
        student_rows = daily_agg.loc[daily_agg["student_id"] == student_id, FEATURE_COLUMNS]
        student_values = student_rows.to_numpy(dtype=np.float32)
        if len(student_values) > seq_length:
            student_values = student_values[-seq_length:]

        sequences[idx, -len(student_values) :, :] = student_values

    return sequences, student_ids.astype(np.int64)


def standardize_sequences(
    sequences: np.ndarray,
    fit_scaler: bool = True,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> tuple[np.ndarray, StandardScaler]:
    """Standardize features across all non-padded timesteps and persist scaler."""
    scaler_path = Path(scaler_path)
    flattened = sequences.reshape(-1, sequences.shape[-1])
    non_padded_mask = np.any(flattened != 0, axis=1)

    if fit_scaler:
        scaler = StandardScaler()
        fit_data = flattened[non_padded_mask] if non_padded_mask.any() else flattened
        scaler.fit(fit_data)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with scaler_path.open("wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
    else:
        with scaler_path.open("rb") as scaler_file:
            scaler = pickle.load(scaler_file)

    standardized = flattened.copy()
    if non_padded_mask.any():
        standardized[non_padded_mask] = scaler.transform(flattened[non_padded_mask])
    standardized_sequences = standardized.reshape(sequences.shape).astype(np.float32)
    return standardized_sequences, scaler


def process_all_behavioral(
    logs_csv: str | Path = DEFAULT_BEHAVIORAL_LOGS_CSV,
    sequences_path: str | Path = DEFAULT_SEQUENCES_PATH,
    student_ids_path: str | Path = DEFAULT_STUDENT_IDS_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    seq_length: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Run full behavioral preprocessing pipeline and save artifacts."""
    logger.info("Loading behavioral logs...")
    logs_df = load_behavioral_logs(logs_csv)

    logger.info("Aggregating daily behavioral features...")
    daily_agg = aggregate_daily(logs_df)

    logger.info("Creating fixed-length sequences...")
    sequences, student_ids = create_sequences(daily_agg, seq_length=seq_length)

    logger.info("Standardizing behavioral sequences...")
    standardized_sequences, _ = standardize_sequences(
        sequences,
        fit_scaler=True,
        scaler_path=scaler_path,
    )

    sequences_path = Path(sequences_path)
    student_ids_path = Path(student_ids_path)
    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sequences_path, standardized_sequences)
    np.save(student_ids_path, student_ids)

    logger.info("Saved behavioral sequences to %s with shape %s", sequences_path, standardized_sequences.shape)
    return standardized_sequences, student_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sequences, student_ids = process_all_behavioral()
    print(f"Behavioral sequences shape: {sequences.shape}")
    print(f"Behavioral student IDs shape: {student_ids.shape}")
