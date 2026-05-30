import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import select

from app.models.prediction import Prediction
from app.models.student import Student
from app.api.predictions import get_prediction_service, PredictionService


@pytest_asyncio.fixture
async def test_student(db_session):
    student = Student(name="Test Student", demographics={"age": 20})
    db_session.add(student)
    await db_session.commit()
    await db_session.refresh(student)
    return student


@pytest_asyncio.fixture
async def authenticated_client(client, test_user):
    login_response = await client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "testpassword"},
    )
    assert login_response.status_code == 200
    cookies = login_response.cookies
    client.cookies = cookies
    return client


@pytest.mark.asyncio
async def test_predict_happy_path(authenticated_client, test_student, db_session):
    mock_result = {
        "student_id": test_student.id,
        "at_risk_probability": 0.72,
        "risk_level": "high",
        "explanation": {
            "risk_level": "high",
            "probability": 0.72,
            "top_factors": [
                {
                    "feature": "attendance_rate",
                    "shap_value": -0.18,
                    "description": "Attendance is only 65% (strong negative factor)",
                }
            ],
            "modality_contributions": "Text analysis suggests negative sentiment. Tabular data shows poor academic indicators.",
            "narrative_summary": "This student is at high risk (probability: 72.0%). Key factors: (1) Attendance is only 65% (strong negative factor)",
        },
        "model_version": "fusion-xgboost-v1",
        "prediction_id": "pred-123-abc12345",
    }

    with patch("app.api.predictions.get_prediction_service") as mock_get_service:
        mock_service = AsyncMock(spec=PredictionService)
        mock_service.predict.return_value = mock_result
        mock_get_service.return_value = mock_service

        response = await authenticated_client.post(
            "/predictions/predict",
            json={"student_id": test_student.id},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == test_student.id
    assert data["at_risk_probability"] == 0.72
    assert data["risk_level"] == "high"
    assert "explanation" in data
    assert data["explanation"]["risk_level"] == "high"
    assert data["model_version"] == "fusion-xgboost-v1"
    assert "prediction_id" in data


@pytest.mark.asyncio
async def test_predict_invalid_student_id(authenticated_client):
    with patch("app.api.predictions.get_prediction_service") as mock_get_service:
        mock_service = AsyncMock(spec=PredictionService)
        from fastapi import HTTPException
        mock_service.predict.side_effect = HTTPException(status_code=404, detail="Student not found")
        mock_get_service.return_value = mock_service

        response = await authenticated_client.post(
            "/predictions/predict",
            json={"student_id": 99999},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Student not found"


@pytest.mark.asyncio
async def test_predict_without_auth(client):
    response = await client.post(
        "/predictions/predict",
        json={"student_id": 1},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_predict_response_schema_validation(authenticated_client, test_student):
    mock_result = {
        "student_id": test_student.id,
        "at_risk_probability": 0.35,
        "risk_level": "medium",
        "explanation": {
            "risk_level": "medium",
            "probability": 0.35,
            "top_factors": [],
            "modality_contributions": "No modality-specific contributions available.",
            "narrative_summary": "This student is at medium risk (probability: 35.0%). Key factors: ",
        },
        "model_version": "fusion-xgboost-v1",
        "prediction_id": "pred-456-def67890",
    }

    with patch("app.api.predictions.get_prediction_service") as mock_get_service:
        mock_service = AsyncMock(spec=PredictionService)
        mock_service.predict.return_value = mock_result
        mock_get_service.return_value = mock_service

        response = await authenticated_client.post(
            "/predictions/predict",
            json={"student_id": test_student.id},
        )

    assert response.status_code == 200
    data = response.json()
    required_fields = {
        "student_id",
        "at_risk_probability",
        "risk_level",
        "explanation",
        "model_version",
        "prediction_id",
    }
    assert required_fields.issubset(data.keys())

    explanation_fields = {
        "risk_level",
        "probability",
        "top_factors",
        "modality_contributions",
        "narrative_summary",
    }
    assert explanation_fields.issubset(data["explanation"].keys())


@pytest.mark.asyncio
async def test_predict_database_persistence(authenticated_client, test_student, db_session):
    def mock_load_models(self):
        self._text_predictor = MagicMock()
        self._text_predictor.predict.return_value = 0.6
        self._tabular_predictor = MagicMock()
        self._tabular_predictor.predict.return_value = 0.7
        self._behavioral_predictor = MagicMock()
        self._behavioral_predictor.predict.return_value = 0.5
        self._fusion_predictor = MagicMock()
        self._fusion_predictor.predict_with_missing.return_value = 0.75
        self._explainability_service = MagicMock()
        self._explainability_service.explain.return_value = {
            "risk_level": "high",
            "probability": 0.75,
            "top_factors": [],
            "modality_contributions": "",
            "narrative_summary": "High risk",
        }

    with patch.object(PredictionService, "_load_models", mock_load_models):
        response = await authenticated_client.post(
            "/predictions/predict",
            json={"student_id": test_student.id},
        )

    assert response.status_code == 200

    result = await db_session.execute(
        select(Prediction).where(Prediction.student_id == test_student.id)
    )
    prediction = result.scalar_one_or_none()
    assert prediction is not None
    assert prediction.at_risk_probability == 0.75
    assert prediction.risk_level == "high"


@pytest.mark.asyncio
async def test_predict_rate_limiting(authenticated_client, test_student):
    mock_result = {
        "student_id": test_student.id,
        "at_risk_probability": 0.5,
        "risk_level": "medium",
        "explanation": {
            "risk_level": "medium",
            "probability": 0.5,
            "top_factors": [],
            "modality_contributions": "",
            "narrative_summary": "Medium risk",
        },
        "model_version": "fusion-xgboost-v1",
        "prediction_id": "pred-000-zzz00000",
    }

    with patch("app.api.predictions.get_prediction_service") as mock_get_service:
        mock_service = AsyncMock(spec=PredictionService)
        mock_service.predict.return_value = mock_result
        mock_get_service.return_value = mock_service

        for i in range(11):
            response = await authenticated_client.post(
                "/predictions/predict",
                json={"student_id": test_student.id},
            )

    assert response.status_code == 429
