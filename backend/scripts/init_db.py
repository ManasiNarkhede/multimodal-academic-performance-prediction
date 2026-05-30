import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_engine, AsyncSessionLocal
from app.db.base import Base
from app.models.cohort import Cohort
from app.models.student import Student
from app.models.user import User


from app.auth.security import get_password_hash


async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        await seed_data(session)
        await session.commit()

    print("Database initialized successfully.")


async def seed_data(session: AsyncSession):
    existing_cohorts = await session.execute(select(Cohort).limit(1))
    if existing_cohorts.scalar_one_or_none() is not None:
        print("Data already seeded. Skipping.")
        return

    cohorts = [
        Cohort(name="Fall 2024"),
        Cohort(name="Spring 2025"),
    ]
    session.add_all(cohorts)
    await session.flush()

    students = [
        Student(name="Alice Johnson", cohort_id=cohorts[0].id, demographics={"age": 20, "gender": "female"}),
        Student(name="Bob Smith", cohort_id=cohorts[0].id, demographics={"age": 21, "gender": "male"}),
        Student(name="Charlie Brown", cohort_id=cohorts[0].id, demographics={"age": 19, "gender": "male"}),
        Student(name="Diana Prince", cohort_id=cohorts[0].id, demographics={"age": 22, "gender": "female"}),
        Student(name="Evan Wright", cohort_id=cohorts[0].id, demographics={"age": 20, "gender": "male"}),
        Student(name="Fiona Gallagher", cohort_id=cohorts[1].id, demographics={"age": 21, "gender": "female"}),
        Student(name="George Miller", cohort_id=cohorts[1].id, demographics={"age": 19, "gender": "male"}),
        Student(name="Hannah Lee", cohort_id=cohorts[1].id, demographics={"age": 20, "gender": "female"}),
        Student(name="Ian Clark", cohort_id=cohorts[1].id, demographics={"age": 22, "gender": "male"}),
        Student(name="Julia Roberts", cohort_id=cohorts[1].id, demographics={"age": 21, "gender": "female"}),
    ]
    session.add_all(students)

    admin_user = User(
        email="admin@example.com",
        hashed_password=get_password_hash("admin"),
        role="admin",
    )
    session.add(admin_user)

    print("Seeded 2 cohorts, 10 students, and 1 admin user.")


if __name__ == "__main__":
    asyncio.run(init_db())
