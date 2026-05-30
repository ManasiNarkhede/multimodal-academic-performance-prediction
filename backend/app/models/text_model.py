import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from app.core.mlflow import end_run, log_metrics, log_model, log_params, start_run

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDINGS_PATH = os.path.join("..", "data", "processed", "text_features.npy")
DEFAULT_STUDENT_IDS_PATH = os.path.join("..", "data", "processed", "text_student_ids.npy")
DEFAULT_STUDENTS_CSV = os.path.join("..", "data", "raw", "students.csv")
DEFAULT_MODELS_DIR = os.path.join("models")


class TextClassifier(nn.Module):
    """Lightweight classifier head on top of frozen DistilBERT embeddings."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for precomputed embeddings and labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class TextPredictor:
    """Wrapper for loading a trained TextClassifier and running inference."""

    def __init__(self, model_path: Optional[str] = None, input_dim: int = 768):
        self.input_dim = input_dim
        self.device = "cpu"
        self.model = TextClassifier(input_dim=input_dim)

        if model_path is None:
            model_path = self._resolve_latest_model()

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            logger.warning(f"No model found at {model_path}; predictor will use random weights.")

    def _resolve_latest_model(self) -> Optional[str]:
        symlink_path = os.path.join(DEFAULT_MODELS_DIR, "text_model_latest.pt")
        if os.path.islink(symlink_path) or os.path.exists(symlink_path):
            return symlink_path

        if os.path.isdir(DEFAULT_MODELS_DIR):
            candidates = [
                f for f in os.listdir(DEFAULT_MODELS_DIR)
                if f.startswith("text_model_") and f.endswith(".pt")
            ]
            if candidates:
                candidates.sort()
                return os.path.join(DEFAULT_MODELS_DIR, candidates[-1])
        return None

    def load(self, model_path: str) -> None:
        """Load model weights from disk."""
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")

    def predict(self, embedding: np.ndarray) -> float:
        """Return predicted probability for a single embedding vector."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).cpu().item()
        return prob

    def predict_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for a batch of embeddings."""
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs


def _load_data(
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    student_ids_path: str = DEFAULT_STUDENT_IDS_PATH,
    students_csv: str = DEFAULT_STUDENTS_CSV,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and align with labels from students CSV."""
    embeddings = np.load(embeddings_path)
    student_ids = np.load(student_ids_path)

    df = pd.read_csv(students_csv)
    label_map = dict(zip(df["student_id"], df["at_risk"]))

    labels = np.array([label_map[sid] for sid in student_ids], dtype=np.float32)
    return embeddings, labels


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_batch.numpy().flatten().tolist())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
    }


def train_text_model(
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    student_ids_path: str = DEFAULT_STUDENT_IDS_PATH,
    students_csv: str = DEFAULT_STUDENTS_CSV,
    models_dir: str = DEFAULT_MODELS_DIR,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    val_size: float = 0.2,
    random_state: int = 777,
) -> Dict[str, any]:
    """Train a lightweight classifier on frozen DistilBERT embeddings.

    Returns:
        Dictionary with training history and best metrics.
    """
    torch.set_num_threads(2)
    device = "cpu"

    logger.info("Loading embeddings and labels...")
    embeddings, labels = _load_data(embeddings_path, student_ids_path, students_csv)
    logger.info(f"Data shape: {embeddings.shape}, positive ratio: {labels.mean():.3f}")

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=val_size, random_state=random_state, stratify=labels
    )

    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TextClassifier(input_dim=embeddings.shape[1])
    model.to(device)

    pos_weight = torch.tensor([(1 - labels.mean()) / labels.mean()], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = -1.0
    best_state = None
    history = []

    start_run(experiment_name="text_model", run_name=f"text_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_size": val_size,
        "random_state": random_state,
        "input_dim": embeddings.shape[1],
        "train_size": len(X_train),
        "val_size": len(X_val),
        "positive_ratio": float(labels.mean()),
    })

    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_metrics = _evaluate(model, train_loader, device)
        val_metrics = _evaluate(model, val_loader, device)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_summary)

        logger.info(
            f"Epoch {epoch:02d} | loss={epoch_summary['train_loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }

    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"text_model_{timestamp}.pt"
    model_path = os.path.join(models_dir, model_filename)

    if best_state is not None:
        torch.save(best_state, model_path)
        logger.info(f"Saved best model to {model_path} (val_f1={best_f1:.4f})")

        symlink_path = os.path.join(models_dir, "text_model_latest.pt")
        try:
            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                os.remove(symlink_path)
            os.symlink(model_filename, symlink_path)
            logger.info(f"Created symlink: {symlink_path} -> {model_filename}")
        except OSError:
            import shutil
            shutil.copy2(model_path, symlink_path)
            logger.info(f"Copied model to {symlink_path} (symlink not supported)")

    if best_state is not None:
        log_metrics({
            "best_val_f1": best_f1,
            "best_val_accuracy": best_state["val_metrics"]["accuracy"],
            "best_val_precision": best_state["val_metrics"]["precision"],
            "best_val_recall": best_state["val_metrics"]["recall"],
            "best_val_roc_auc": best_state["val_metrics"]["roc_auc"],
            "best_epoch": best_state["epoch"],
        })
        log_model(model_path, artifact_path="model")

    end_run()

    return {
        "best_val_f1": best_f1,
        "best_epoch": best_state["epoch"] if best_state else None,
        "best_model_path": model_path if best_state else None,
        "history": history,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = train_text_model()
    print("\nTraining complete.")
    print(f"Best validation F1: {result['best_val_f1']:.4f} (epoch {result['best_epoch']})")
    print(f"Model saved to: {result['best_model_path']}")
