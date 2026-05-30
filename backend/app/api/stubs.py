"""
TODO: These are temporary stub endpoints for frontend development only.
They return static mock data and should be removed once real endpoints are implemented.
DO NOT use these in production.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/stub", tags=["stubs"])


# TODO: Remove this stub once real cohort endpoint is implemented
@router.get("/cohort/{cohort_id}")
async def get_cohort_summary(cohort_id: int):
    """Return mock cohort summary data."""
    return {
        "cohort_id": cohort_id,
        "cohort_name": f"Computer Science {2023 + cohort_id}",
        "total_students": 150,
        "at_risk_count": 23,
        "risk_distribution": {
            "low": 89,
            "medium": 38,
            "high": 23
        },
        "students": [
            {
                "id": 1,
                "name": "Alice Johnson",
                "risk_probability": 0.12,
                "risk_level": "low"
            },
            {
                "id": 2,
                "name": "Bob Smith",
                "risk_probability": 0.45,
                "risk_level": "medium"
            },
            {
                "id": 3,
                "name": "Charlie Brown",
                "risk_probability": 0.87,
                "risk_level": "high"
            },
            {
                "id": 4,
                "name": "Diana Prince",
                "risk_probability": 0.05,
                "risk_level": "low"
            },
            {
                "id": 5,
                "name": "Evan Wright",
                "risk_probability": 0.72,
                "risk_level": "high"
            }
        ]
    }


class PredictRequest(BaseModel):
    student_id: int
    features: Optional[dict] = None


# TODO: Remove this stub once real prediction endpoint is implemented
@router.post("/predict")
async def predict_student_risk(request: PredictRequest):
    """Return mock prediction with explanation and SHAP values."""
    risk_prob = 0.72 if request.student_id % 3 == 0 else 0.35
    risk_level = "high" if risk_prob > 0.6 else "medium" if risk_prob > 0.3 else "low"

    explanations = {
        "high": "This student is at high risk because: attendance is low (65%), assignment completion rate is below average (45%), and midterm scores are in the bottom quartile.",
        "medium": "This student shows moderate risk due to inconsistent assignment submission and average attendance. Intervention recommended.",
        "low": "This student is performing well with consistent attendance and strong assignment scores. Continue current support."
    }

    return {
        "student_id": request.student_id,
        "at_risk_probability": round(risk_prob, 2),
        "risk_level": risk_level,
        "explanation": explanations[risk_level],
        "shap_values": [
            {"feature": "attendance_rate", "value": -0.18, "description": "Attendance below 70%"},
            {"feature": "assignment_completion", "value": -0.14, "description": "Completion rate at 45%"},
            {"feature": "midterm_score", "value": -0.12, "description": "Score in bottom quartile"},
            {"feature": "study_hours", "value": 0.08, "description": "Above average study time"},
            {"feature": "participation", "value": 0.05, "description": "Active in discussions"}
        ],
        "model_version": "v1.2.0",
        "prediction_id": f"pred-{request.student_id}-stub"
    }


# TODO: Remove this stub once real models endpoint is implemented
@router.get("/models")
async def get_model_metadata():
    """Return mock model metadata."""
    return {
        "models": [
            {
                "name": "ensemble_v1",
                "version": "1.2.0",
                "accuracy": 0.87,
                "f1_score": 0.84,
                "last_trained": "2024-03-15T10:30:00Z",
                "status": "active"
            },
            {
                "name": "tabular_only",
                "version": "1.0.5",
                "accuracy": 0.82,
                "f1_score": 0.79,
                "last_trained": "2024-02-20T14:00:00Z",
                "status": "deprecated"
            },
            {
                "name": "multimodal_v2",
                "version": "2.0.0-beta",
                "accuracy": 0.91,
                "f1_score": 0.89,
                "last_trained": "2024-04-01T09:00:00Z",
                "status": "experimental"
            }
        ]
    }
