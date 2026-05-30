import gc
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.cache import get_embedding_cache
from app.db.session import get_db
from app.explainability.service import ExplainabilityService
from app.models.behavioral_model import BehavioralPredictor
from app.models.fusion_ensemble import FusionPredictor
from app.models.prediction import Prediction
from app.models.student import Student
from app.models.tabular_model import TabularPredictor
from app.models.text_model import TextPredictor
from app.models.user import User
from app.schemas.prediction import PredictRequest, PredictResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])
limiter = Limiter(key_func=get_remote_address)

from app.core.paths import MODELS_DIR, PROCESSED_DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_feature_cache: dict[str, np.ndarray] = {}


def _get_cached_array(path: Path) -> np.ndarray:
    path_str = str(path)
    if path_str not in _feature_cache:
        _feature_cache[path_str] = np.load(path_str)
    return _feature_cache[path_str]


def _get_feature_by_student_id(student_id: int, features_path: Path, student_ids_path: Path) -> Optional[np.ndarray]:
    cache = get_embedding_cache()
    feature_type = features_path.stem
    feature_index = hash(feature_type) % 100000
    composite_key = student_id * 100000 + feature_index

    cached = cache.get(composite_key)
    if cached is not None:
        return cached

    features = _get_cached_array(features_path)
    student_ids = _get_cached_array(student_ids_path)
    idx = np.where(student_ids == student_id)[0]
    if len(idx) == 0:
        return None
    result = features[idx[0]]
    cache.set(composite_key, result)
    return result


class PredictionService:
    def __init__(self) -> None:
        self._text_predictor: Optional[TextPredictor] = None
        self._tabular_predictor: Optional[TabularPredictor] = None
        self._behavioral_predictor: Optional[BehavioralPredictor] = None
        self._fusion_predictor: Optional[FusionPredictor] = None
        self._explainability_service: Optional[ExplainabilityService] = None

    def _load_models(self) -> None:
        if self._text_predictor is None:
            self._text_predictor = TextPredictor(
                model_path=str(MODELS_DIR / "text_model_latest.pt")
            )
        if self._tabular_predictor is None:
            self._tabular_predictor = TabularPredictor(
                model_path=str(MODELS_DIR / "tabular_model_latest.json")
            )
        if self._behavioral_predictor is None:
            self._behavioral_predictor = BehavioralPredictor(
                model_path=str(MODELS_DIR / "behavioral_model_latest.pt")
            )
        if self._fusion_predictor is None:
            self._fusion_predictor = FusionPredictor(
                model_path=str(MODELS_DIR / "fusion_model_latest.json")
            )
        if self._explainability_service is None:
            self._explainability_service = ExplainabilityService()

    async def predict(self, student_id: int, user_id: Optional[int], db: AsyncSession) -> dict[str, Any]:
        self._load_models()
        assert self._text_predictor is not None
        assert self._tabular_predictor is not None
        assert self._behavioral_predictor is not None
        assert self._fusion_predictor is not None
        assert self._explainability_service is not None

        result = await db.execute(select(Student).where(Student.id == student_id))
        student = result.scalar_one_or_none()
        if student is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found",
            )

        text_features = _get_feature_by_student_id(
            student_id,
            PROCESSED_DATA_DIR / "text_features.npy",
            PROCESSED_DATA_DIR / "text_student_ids.npy",
        )
        tabular_features = _get_feature_by_student_id(
            student_id,
            PROCESSED_DATA_DIR / "tabular_features.npy",
            PROCESSED_DATA_DIR / "tabular_student_ids.npy",
        )
        behavioral_features = _get_feature_by_student_id(
            student_id,
            PROCESSED_DATA_DIR / "behavioral_sequences.npy",
            PROCESSED_DATA_DIR / "behavioral_student_ids.npy",
        )

        text_prob: Optional[float] = None
        tabular_prob: Optional[float] = None
        behavioral_prob: Optional[float] = None

        if text_features is not None:
            text_prob = float(self._text_predictor.predict(text_features))
        if tabular_features is not None:
            tabular_prob = float(self._tabular_predictor.predict(tabular_features))
        if behavioral_features is not None:
            behavioral_prob = float(self._behavioral_predictor.predict(behavioral_features))

        fused_prob = float(
            self._fusion_predictor.predict_with_missing(
                text_prob=text_prob,
                tabular_prob=tabular_prob,
                behavioral_prob=behavioral_prob,
            )
        )

        explanation = self._explainability_service.explain(
            student_features=tabular_features if tabular_features is not None else np.zeros(13),
            feature_values=None,
            text_prob=text_prob,
            tabular_prob=tabular_prob,
            behavioral_prob=behavioral_prob,
        )

        _ = gc.collect()

        if fused_prob >= 0.7:
            risk_level = "high"
        elif fused_prob >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        prediction = Prediction(
            student_id=student_id,
            user_id=user_id,
            at_risk_probability=fused_prob,
            risk_level=risk_level,
            explanation_text=explanation["narrative_summary"],
            text_score=text_prob,
            tabular_score=tabular_prob,
            behavioral_score=behavioral_prob,
        )
        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)

        model_version = "fusion-ensemble-v1"
        if hasattr(self._fusion_predictor.ensemble, "meta_learner_type"):
            model_version = f"fusion-{self._fusion_predictor.ensemble.meta_learner_type}-v1"

        return {
            "student_id": student_id,
            "at_risk_probability": round(fused_prob, 4),
            "risk_level": risk_level,
            "explanation": explanation,
            "model_version": model_version,
            "prediction_id": f"pred-{prediction.id}-{uuid.uuid4().hex[:8]}",
        }


_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


@router.post("/predict", response_model=PredictResponse)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    service = get_prediction_service()
    user_id = int(current_user.id)  # type: ignore[arg-type]
    return await service.predict(
        student_id=payload.student_id,
        user_id=user_id,
        db=db,
    )
