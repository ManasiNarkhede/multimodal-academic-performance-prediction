import asyncio
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.models.cohort import Cohort
from app.models.student import Student
from app.core.config import settings


CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "students.csv"
DEFAULT_LIMIT = 100


async def seed_db(
    limit: int = DEFAULT_LIMIT,
    seed_all: bool = False,
    session: AsyncSession | None = None,
) -> tuple[int, int]:
    """Seed cohorts and students from synthetic CSV data.

    Args:
        limit: Maximum number of students to seed (default 100).
        seed_all: If True, ignore limit and seed all records.
        session: Optional async session for testing. If not provided, creates one.

    Returns:
        Tuple of (cohorts_seeded, students_seeded).
    """
    if settings.ENVIRONMENT == "production":
        print("ERROR: Refusing to seed production database.")
        return 0, 0

    if not CSV_PATH.exists():
        print(f"WARNING: CSV file not found at {CSV_PATH}. Skipping seed.")
        return 0, 0

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not seed_all and limit > 0:
        rows = rows[:limit]

    cohort_ids = sorted({int(row["cohort_id"]) for row in rows})
    cohort_map: dict[int, Cohort] = {}

    if session is not None:
        cohorts_seeded = await _seed_cohorts(session, cohort_ids)
        students_seeded = await _seed_students(session, rows, cohort_map)
        await session.commit()
    else:
        async with AsyncSessionLocal() as session:
            cohorts_seeded = await _seed_cohorts(session, cohort_ids)
            students_seeded = await _seed_students(session, rows, cohort_map)
            await session.commit()

    print(f"Seeded {cohorts_seeded} cohorts, {students_seeded} students")
    return cohorts_seeded, students_seeded


async def _seed_cohorts(session: AsyncSession, cohort_ids: list[int]) -> int:
    """Insert cohorts idempotently. Returns number of new cohorts inserted."""
    count = 0
    for cid in cohort_ids:
        name = f"Cohort {cid}"
        result = await session.execute(select(Cohort).where(Cohort.name == name))
        existing = result.scalar_one_or_none()
        if existing is None:
            cohort = Cohort(name=name)
            session.add(cohort)
            await session.flush()
            count += 1
    return count


async def _seed_students(
    session: AsyncSession,
    rows: list[dict],
    cohort_map: dict[int, Cohort],
) -> int:
    """Insert students idempotently. Returns number of new students inserted."""
    cohort_names = {cid: f"Cohort {cid}" for cid in {int(r["cohort_id"]) for r in rows}}
    for cid, name in cohort_names.items():
        result = await session.execute(select(Cohort).where(Cohort.name == name))
        cohort = result.scalar_one_or_none()
        if cohort:
            cohort_map[cid] = cohort

    count = 0
    for row in rows:
        name = row["name"]
        cid = int(row["cohort_id"])
        cohort = cohort_map.get(cid)
        cohort_db_id = cohort.id if cohort else None

        result = await session.execute(
            select(Student).where(
                Student.name == name,
                Student.cohort_id == cohort_db_id,
            )
        )
        existing = result.scalar_one_or_none()
        if existing is None:
            demographics = {
                "age": int(row["age"]),
                "gender": row["gender"],
                "socioeconomic_status": row["socioeconomic_status"],
                "prior_gpa": float(row["prior_gpa"]),
            }
            student = Student(
                name=name,
                cohort_id=cohort_db_id,
                demographics=demographics,
            )
            session.add(student)
            count += 1

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed database with synthetic student data.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Seed all records from CSV (default: first 100).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of students to seed (default: {DEFAULT_LIMIT}).",
    )
    args = parser.parse_args()

    limit = 0 if args.all else args.limit
    asyncio.run(seed_db(limit=limit, seed_all=args.all))
