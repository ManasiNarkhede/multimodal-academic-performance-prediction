from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.db.session import get_db
from app.models.cohort import Cohort
from app.models.student import Student
from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.cohort import (
    CohortResponse,
    CohortStudentResponse,
    RiskDistribution,
    AverageModalityScores,
    CohortPagination,
)

router = APIRouter(prefix="/cohorts", tags=["cohorts"])


@router.get("/{cohort_id}", response_model=CohortResponse)
async def get_cohort(
    cohort_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    sort_by: str = Query("risk", pattern="^(risk|name)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    risk_level: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CohortResponse:
    result = await db.execute(select(Cohort).where(Cohort.id == cohort_id))
    cohort = result.scalar_one_or_none()
    if cohort is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cohort not found",
        )

    students_result = await db.execute(
        select(Student).where(Student.cohort_id == cohort_id)
    )
    students = students_result.scalars().all()

    student_ids = [s.id for s in students]
    pred_map = {}
    if student_ids:
        latest_pred_subq = (
            select(
                Prediction.student_id,
                func.max(Prediction.created_at).label("max_created_at"),
            )
            .where(Prediction.student_id.in_(student_ids))
            .group_by(Prediction.student_id)
            .subquery()
        )

        preds_result = await db.execute(
            select(Prediction).join(
                latest_pred_subq,
                and_(
                    Prediction.student_id == latest_pred_subq.c.student_id,
                    Prediction.created_at == latest_pred_subq.c.max_created_at,
                ),
            )
        )
        predictions = preds_result.scalars().all()
        pred_map = {p.student_id: p for p in predictions}

    total_students = len(students)
    at_risk_count = 0
    risk_dist = {"low": 0, "medium": 0, "high": 0}
    text_scores = []
    tabular_scores = []
    behavioral_scores = []

    students_data = []
    for student in students:
        pred = pred_map.get(student.id)
        risk_prob = pred.at_risk_probability if pred else None
        risk_lvl = pred.risk_level if pred else None
        pred_date = pred.created_at.isoformat() if pred and pred.created_at else None

        if pred:
            if risk_lvl in risk_dist:
                risk_dist[risk_lvl] += 1
            if risk_lvl in ("medium", "high"):
                at_risk_count += 1
            if pred.text_score is not None:
                text_scores.append(pred.text_score)
            if pred.tabular_score is not None:
                tabular_scores.append(pred.tabular_score)
            if pred.behavioral_score is not None:
                behavioral_scores.append(pred.behavioral_score)

        students_data.append(
            {
                "id": student.id,
                "name": student.name,
                "risk_probability": risk_prob,
                "risk_level": risk_lvl,
                "last_prediction_date": pred_date,
                "prediction": pred,
            }
        )

    filtered = students_data
    if risk_level:
        filtered = [s for s in filtered if s["risk_level"] == risk_level]

    if from_date:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
        filtered = [
            s
            for s in filtered
            if s["prediction"]
            and s["prediction"].created_at
            and s["prediction"].created_at.date() >= from_dt
        ]

    if to_date:
        to_dt = datetime.strptime(to_date, "%Y-%m-%d").date()
        filtered = [
            s
            for s in filtered
            if s["prediction"]
            and s["prediction"].created_at
            and s["prediction"].created_at.date() <= to_dt
        ]

    if sort_by == "risk":
        if order == "asc":
            filtered.sort(
                key=lambda s: (s["risk_probability"] is None, s["risk_probability"] or 0),
            )
        else:
            filtered.sort(
                key=lambda s: (s["risk_probability"] is None, -(s["risk_probability"] or 0)),
            )
    elif sort_by == "name":
        filtered.sort(
            key=lambda s: s["name"].lower(),
            reverse=(order == "desc"),
        )

    total_filtered = len(filtered)
    total_pages = (total_filtered + limit - 1) // limit if total_filtered > 0 else 1
    start = (page - 1) * limit
    end = start + limit
    paginated = filtered[start:end]

    at_risk_percentage = (
        (at_risk_count / total_students * 100) if total_students > 0 else 0.0
    )

    return CohortResponse(
        cohort_id=cohort.id,
        cohort_name=cohort.name,
        total_students=total_students,
        at_risk_count=at_risk_count,
        at_risk_percentage=round(at_risk_percentage, 2),
        risk_distribution=RiskDistribution(**risk_dist),
        average_modality_scores=AverageModalityScores(
            text=round(sum(text_scores) / len(text_scores), 4) if text_scores else None,
            tabular=round(sum(tabular_scores) / len(tabular_scores), 4)
            if tabular_scores
            else None,
            behavioral=round(sum(behavioral_scores) / len(behavioral_scores), 4)
            if behavioral_scores
            else None,
        ),
        students=[
            CohortStudentResponse(
                id=s["id"],
                name=s["name"],
                risk_probability=s["risk_probability"],
                risk_level=s["risk_level"],
                last_prediction_date=s["last_prediction_date"],
            )
            for s in paginated
        ],
        pagination=CohortPagination(
            page=page,
            limit=limit,
            total_pages=total_pages,
            total_students=total_filtered,
        ),
    )
