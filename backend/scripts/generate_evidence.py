import asyncio
import time
import json
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.main import app
from app.db.session import get_db
from app.db.base import Base
from app.models.user import User
from app.models.student import Student
from app.auth.security import get_password_hash

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
engine = create_async_engine(TEST_DB_URL)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with AsyncSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        user = User(
            email="evidence@test.com",
            hashed_password=get_password_hash("testpass"),
            role="user",
        )
        session.add(user)
        student = Student(name="Evidence Student", demographics={"age": 21})
        session.add(student)
        await session.commit()
        await session.refresh(student)
        student_id = student.id
        await session.refresh(user)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        login_resp = await client.post(
            "/auth/login",
            data={"username": "evidence@test.com", "password": "testpass"},
        )
        assert login_resp.status_code == 200
        cookies = login_resp.cookies

        start = time.perf_counter()
        response = await client.post(
            "/predictions/predict",
            json={"student_id": student_id},
            cookies=cookies,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        print(f"Response status: {response.status_code}")
        print(f"Latency: {latency_ms:.2f} ms")

        if response.status_code == 200:
            data = response.json()

            evidence_dir = Path(".sisyphus/evidence")
            evidence_dir.mkdir(parents=True, exist_ok=True)

            with open(evidence_dir / "task-17-predict-response.json", "w") as f:
                json.dump(data, f, indent=2)

            with open(evidence_dir / "task-17-latency.txt", "w") as f:
                f.write(f"Inference latency: {latency_ms:.2f} ms\n")
                status = "PASS" if latency_ms < 2000 else "FAIL"
                f.write(f"Status: {status} (threshold: 2000 ms)\n")

            rate_limit_results = []
            for i in range(12):
                resp = await client.post(
                    "/predictions/predict",
                    json={"student_id": student_id},
                    cookies=cookies,
                )
                rate_limit_results.append({"request": i + 1, "status": resp.status_code})

            with open(evidence_dir / "task-17-rate-limit.json", "w") as f:
                json.dump(rate_limit_results, f, indent=2)

            print("Evidence files saved.")
        else:
            print(f"Error: {response.text}")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
