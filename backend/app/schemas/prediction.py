"""Pydantic schemas for the prediction endpoint."""

from pydantic import BaseModel
from typing import List, Optional


class TopFactor(BaseModel):
    feature: str
    shap_value: float
    description: str


class ExplanationResponse(BaseModel):
    risk_level: str
    probability: float
    top_factors: List[TopFactor]
    modality_contributions: str
    narrative_summary: str


class PredictRequest(BaseModel):
    student_id: int


class PredictResponse(BaseModel):
    student_id: int
    at_risk_probability: float
    risk_level: str
    explanation: ExplanationResponse
    model_version: str
    prediction_id: str
