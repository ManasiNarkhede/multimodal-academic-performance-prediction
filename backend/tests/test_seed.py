import pytest
from sqlalchemy import select, func

from app.models.cohort import Cohort
from app.models.student import Student
from scripts.seed_db import seed_db, _seed_cohorts, _seed_students


@pytest.mark.asyncio
async def test_seed_script_runs_without_errors(db_session):
    cohorts_seeded, students_seeded = await seed_db(limit=10, session=db_session)
    assert cohorts_seeded >= 0
    assert students_seeded >= 0


@pytest.mark.asyncio
async def test_seed_creates_cohorts_and_students(db_session):
    await seed_db(limit=10, session=db_session)

    cohort_count = await db_session.execute(select(func.count()).select_from(Cohort))
    assert cohort_count.scalar() > 0

    student_count = await db_session.execute(select(func.count()).select_from(Student))
    assert student_count.scalar() > 0


@pytest.mark.asyncio
async def test_seed_is_idempotent(db_session):
    await seed_db(limit=10, session=db_session)

    cohort_count_first = await db_session.execute(select(func.count()).select_from(Cohort))
    student_count_first = await db_session.execute(select(func.count()).select_from(Student))

    first_cohorts = cohort_count_first.scalar()
    first_students = student_count_first.scalar()

    await seed_db(limit=10, session=db_session)

    cohort_count_second = await db_session.execute(select(func.count()).select_from(Cohort))
    student_count_second = await db_session.execute(select(func.count()).select_from(Student))

    assert cohort_count_second.scalar() == first_cohorts
    assert student_count_second.scalar() == first_students


@pytest.mark.asyncio
async def test_seed_respects_limit(db_session):
    from sqlalchemy import delete
    await db_session.execute(delete(Student))
    await db_session.execute(delete(Cohort))
    await db_session.commit()

    _, students_seeded = await seed_db(limit=5, session=db_session)
    assert students_seeded == 5

    student_count = await db_session.execute(select(func.count()).select_from(Student))
    assert student_count.scalar() == 5
