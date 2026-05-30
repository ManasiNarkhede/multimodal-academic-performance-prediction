import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import settings
from app.db.session import async_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

from app.core.paths import MODELS_DIR, EVIDENCE_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILES = {
    "text_model": MODELS_DIR / "text_model_latest.pt",
    "tabular_model": MODELS_DIR / "tabular_model_latest.json",
    "behavioral_model": MODELS_DIR / "behavioral_model_latest.pt",
    "fusion_model": MODELS_DIR / "fusion_model_latest.json",
}

EVIDENCE_FILES = {
    "text_model": EVIDENCE_DIR / "task-9-text-model-metrics.json",
    "tabular_model": EVIDENCE_DIR / "task-11-tabular-model-metrics.json",
    "behavioral_model": EVIDENCE_DIR / "task-13-behavioral-model-metrics.json",
    "fusion_model": EVIDENCE_DIR / "task-14-fusion-metrics.json",
}

APP_VERSION = "0.1.0"


async def _check_dbConnectivity(engine: AsyncEngine) -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        return False


def _check_model_files() -> dict[str, bool]:
    return {name: path.exists() or path.is_symlink() for name, path in MODEL_FILES.items()}


def _get_memory_usage_mb() -> int:
    process = psutil.Process()
    return int(process.memory_info().rss / (1024 * 1024))


def _get_system_metrics() -> dict[str, float]:
    return {
        "cpu_percent": round(psutil.cpu_percent(interval=0.1), 2),
        "memory_percent": round(psutil.virtual_memory().percent, 2),
        "disk_percent": round(psutil.disk_usage("/").percent, 2),
    }


def _read_evidence(model_name: str) -> dict[str, Any]:
    path = EVIDENCE_FILES.get(model_name)
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read evidence for %s: %s", model_name, exc)
        return {}


def _get_model_version(model_name: str) -> str:
    evidence = _read_evidence(model_name)
    if model_name == "text_model" and "best_epoch" in evidence:
        return f"v1.0.{evidence['best_epoch']}"
    if model_name == "behavioral_model" and "best_epoch" in evidence:
        return f"v1.0.{evidence['best_epoch']}"
    if model_name == "tabular_model" and "feature_names" in evidence:
        return "v1.0.0"
    if model_name == "fusion_model" and "meta_learner_type" in evidence:
        return f"v1.0.{evidence.get('meta_learner_type', 'ensemble')}"
    return "v1.0.0"


def _get_last_trained(model_name: str) -> str | None:
    model_path = MODEL_FILES.get(model_name)
    if model_path is None or not (model_path.exists() or model_path.is_symlink()):
        return None
    try:
        mtime = model_path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _get_model_metrics(model_name: str) -> dict[str, Any]:
    evidence = _read_evidence(model_name)
    metrics: dict[str, Any] = {}

    if model_name == "fusion_model":
        test_metrics = evidence.get("test_metrics", {})
        metrics = {
            "accuracy": test_metrics.get("accuracy"),
            "f1": test_metrics.get("f1"),
            "precision": test_metrics.get("precision"),
            "recall": test_metrics.get("recall"),
            "roc_auc": test_metrics.get("roc_auc"),
        }
    elif model_name in ("text_model", "behavioral_model"):
        metrics = {
            "best_val_f1": evidence.get("best_val_f1"),
            "best_epoch": evidence.get("best_epoch"),
            "training_time_seconds": evidence.get("training_time_seconds"),
        }
    elif model_name == "tabular_model":
        val_metrics = evidence.get("validation_metrics", {})
        metrics = {
            "accuracy": val_metrics.get("accuracy"),
            "f1": val_metrics.get("f1"),
            "precision": val_metrics.get("precision"),
            "recall": val_metrics.get("recall"),
            "roc_auc": val_metrics.get("roc_auc"),
        }

    return {k: v for k, v in metrics.items() if v is not None}


@router.get("/health")
async def health_check() -> dict[str, Any]:
    db_ok = await _check_dbConnectivity(async_engine)
    model_status = _check_model_files()
    models_ok = all(model_status.values())
    memory_mb = _get_memory_usage_mb()
    system_metrics = _get_system_metrics()

    return {
        "status": "ok" if db_ok and models_ok else "degraded",
        "db": "ok" if db_ok else "error",
        "models": "ok" if models_ok else "error",
        "memory_mb": memory_mb,
        **system_metrics,
    }


@router.get("/info")
async def info() -> dict[str, Any]:
    models_metadata = []
    for model_name in MODEL_FILES:
        last_trained = _get_last_trained(model_name)
        metrics = _get_model_metrics(model_name)
        models_metadata.append({
            "name": model_name,
            "version": _get_model_version(model_name),
            "last_trained": last_trained,
            "metrics": metrics,
        })

    return {
        "version": APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "models": models_metadata,
    }


@router.get("/models")
async def list_models() -> dict[str, Any]:
    models = []
    for model_name in MODEL_FILES:
        model_path = MODEL_FILES[model_name]
        exists = model_path.exists() or model_path.is_symlink()
        last_trained = _get_last_trained(model_name)
        models.append({
            "name": model_name,
            "version": _get_model_version(model_name),
            "last_trained": last_trained,
            "status": "active" if exists else "deprecated",
        })

    return {"models": models}
