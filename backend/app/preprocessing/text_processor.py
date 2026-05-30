import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.core.model_config import get_text_encoder, get_model_metadata, get_text_model_name

logger = logging.getLogger(__name__)

DEFAULT_TEXTS_CSV = os.path.join("data", "raw", "texts.csv")
DEFAULT_STUDENTS_CSV = os.path.join("data", "raw", "students.csv")
DEFAULT_CACHE_DIR = os.path.join("data", "processed")
DEFAULT_EMBEDDINGS_PATH = os.path.join(DEFAULT_CACHE_DIR, "text_features.npy")
DEFAULT_STUDENT_IDS_PATH = os.path.join(DEFAULT_CACHE_DIR, "text_student_ids.npy")


def load_texts(csv_path: str = DEFAULT_TEXTS_CSV) -> Dict[int, str]:
    """Load texts from CSV and group by student_id.

    Args:
        csv_path: Path to the texts CSV file.

    Returns:
        Dict mapping student_id to concatenated text string.
    """
    df = pd.read_csv(csv_path)
    grouped = df.groupby("student_id")["text"].apply(lambda texts: " ".join(texts))
    return grouped.to_dict()


def load_student_labels(csv_path: str = DEFAULT_STUDENTS_CSV) -> Dict[int, int]:
    """Load student at_risk labels from CSV.

    Args:
        csv_path: Path to the students CSV file.

    Returns:
        Dict mapping student_id to at_risk label (0 or 1).
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df["student_id"], df["at_risk"]))


class TextEmbeddingDataset(Dataset):
    """PyTorch Dataset for text embeddings."""

    def __init__(
        self,
        texts: Dict[int, str],
        labels: Optional[Dict[int, int]] = None,
    ):
        self.student_ids = list(texts.keys())
        self.texts = [texts[sid] for sid in self.student_ids]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.student_ids)

    def __getitem__(self, idx: int) -> Tuple[int, str, Optional[int]]:
        sid = self.student_ids[idx]
        text = self.texts[idx]
        label = self.labels.get(sid, None) if self.labels else None
        return sid, text, label


def generate_embeddings(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    """Generate pooled embeddings for a list of texts.

    Args:
        texts: List of text strings.
        model: Pre-trained transformer model (frozen).
        tokenizer: Corresponding tokenizer.
        batch_size: Number of texts per batch.
        max_length: Maximum token sequence length.
        device: Device to run on ('cpu', 'cuda', or None for auto).

    Returns:
        Numpy array of shape (len(texts), hidden_size).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    torch.set_num_threads(2)

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            pooled = last_hidden.mean(dim=1)
            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def cache_embeddings(
    embeddings: np.ndarray,
    student_ids: np.ndarray,
    cache_path: str = DEFAULT_EMBEDDINGS_PATH,
    ids_path: str = DEFAULT_STUDENT_IDS_PATH,
) -> None:
    """Save embeddings and student IDs to disk.

    Args:
        embeddings: Array of shape (n_students, hidden_size).
        student_ids: Array of student IDs.
        cache_path: Path to save embeddings.
        ids_path: Path to save student IDs.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    np.save(ids_path, student_ids)
    logger.info(f"Cached embeddings to {cache_path} (shape: {embeddings.shape})")


def load_cached_embeddings(
    cache_path: str = DEFAULT_EMBEDDINGS_PATH,
    ids_path: str = DEFAULT_STUDENT_IDS_PATH,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load embeddings and student IDs from disk if they exist.

    Args:
        cache_path: Path to embeddings file.
        ids_path: Path to student IDs file.

    Returns:
        Tuple of (embeddings, student_ids) or None if files don't exist.
    """
    if os.path.exists(cache_path) and os.path.exists(ids_path):
        embeddings = np.load(cache_path)
        student_ids = np.load(ids_path)
        logger.info(f"Loaded cached embeddings from {cache_path} (shape: {embeddings.shape})")
        return embeddings, student_ids
    return None


def process_all_texts(
    texts_csv: str = DEFAULT_TEXTS_CSV,
    students_csv: str = DEFAULT_STUDENTS_CSV,
    cache_path: str = DEFAULT_EMBEDDINGS_PATH,
    ids_path: str = DEFAULT_STUDENT_IDS_PATH,
    batch_size: int = 32,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Main pipeline: load texts, generate embeddings, cache results.

    Args:
        texts_csv: Path to texts CSV.
        students_csv: Path to students CSV.
        cache_path: Path to save embeddings.
        ids_path: Path to save student IDs.
        batch_size: Batch size for embedding generation.
        force_recompute: If True, recompute even if cache exists.

    Returns:
        Tuple of (embeddings array, student_ids array).
    """
    if not force_recompute:
        cached = load_cached_embeddings(cache_path, ids_path)
        if cached is not None:
            return cached

    logger.info("Loading texts and student labels...")
    texts = load_texts(texts_csv)
    labels = load_student_labels(students_csv)

    dataset = TextEmbeddingDataset(texts, labels)
    student_ids = np.array(dataset.student_ids)
    text_list = dataset.texts

    logger.info(f"Loading text encoder for {len(text_list)} students...")
    model, tokenizer = get_text_encoder()
    model_name = get_text_model_name()
    metadata = get_model_metadata(model_name)
    hidden_size = metadata["hidden_size"]

    for param in model.parameters():
        param.requires_grad = False

    logger.info(f"Generating embeddings with {model_name} ({hidden_size}-dim)...")
    embeddings = generate_embeddings(text_list, model, tokenizer, batch_size=batch_size)

    cache_embeddings(embeddings, student_ids, cache_path, ids_path)
    return embeddings, student_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embeddings, student_ids = process_all_texts()
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Student IDs shape: {student_ids.shape}")
