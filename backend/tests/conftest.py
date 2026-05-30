import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import delete

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.models.user import User
from app.models.cohort import Cohort
from app.models.student import Student
from app.models.prediction import Prediction
from app.auth.security import get_password_hash

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


test_engine = create_async_engine(
    TEST_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

TestAsyncSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


async def override_get_db():
    async with TestAsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield test_engine
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    async with TestAsyncSessionLocal() as session:
        yield session
        await session.execute(delete(Prediction))
        await session.execute(delete(Student))
        await session.execute(delete(Cohort))
        await session.execute(delete(User))
        await session.commit()


@pytest_asyncio.fixture
async def client(db_engine):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def test_user(db_session):
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword"),
        role="user",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user
