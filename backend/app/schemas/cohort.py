from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class CohortStudentResponse(BaseModel):
    id: int
    name: str
    risk_probability: Optional[float]
    risk_level: Optional[str]
    last_prediction_date: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class RiskDistribution(BaseModel):
    low: int
    medium: int
    high: int


class AverageModalityScores(BaseModel):
    text: Optional[float]
    tabular: Optional[float]
    behavioral: Optional[float]


class CohortPagination(BaseModel):
    page: int
    limit: int
    total_pages: int
    total_students: int


class CohortResponse(BaseModel):
    cohort_id: int
    cohort_name: str
    total_students: int
    at_risk_count: int
    at_risk_percentage: float
    risk_distribution: RiskDistribution
    average_modality_scores: AverageModalityScores
    students: List[CohortStudentResponse]
    pagination: CohortPagination
