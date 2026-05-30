import pytest
import pytest_asyncio

from app.main import app
from app.api.cohorts import router as cohorts_router
from app.models.cohort import Cohort
from app.models.student import Student
from app.models.prediction import Prediction

app.include_router(cohorts_router)


@pytest_asyncio.fixture
async def test_cohort(db_session):
    cohort = Cohort(name="Test Cohort")
    db_session.add(cohort)
    await db_session.commit()
    await db_session.refresh(cohort)
    return cohort


@pytest_asyncio.fixture
async def empty_cohort(db_session):
    cohort = Cohort(name="Empty Cohort")
    db_session.add(cohort)
    await db_session.commit()
    await db_session.refresh(cohort)
    return cohort


@pytest_asyncio.fixture
async def test_students(db_session, test_cohort):
    students = []
    for i in range(5):
        student = Student(name=f"Student {i}", cohort_id=test_cohort.id)
        db_session.add(student)
        students.append(student)
    await db_session.commit()
    for student in students:
        await db_session.refresh(student)
    return students


@pytest_asyncio.fixture
async def test_predictions(db_session, test_students):
    predictions = []
    predictions.append(
        Prediction(
            student_id=test_students[0].id,
            at_risk_probability=0.85,
            risk_level="high",
            text_score=0.8,
            tabular_score=0.7,
            behavioral_score=0.9,
        )
    )
    predictions.append(
        Prediction(
            student_id=test_students[1].id,
            at_risk_probability=0.55,
            risk_level="medium",
            text_score=0.5,
            tabular_score=0.6,
            behavioral_score=0.5,
        )
    )
    predictions.append(
        Prediction(
            student_id=test_students[2].id,
            at_risk_probability=0.2,
            risk_level="low",
            text_score=0.3,
            tabular_score=0.2,
            behavioral_score=0.1,
        )
    )
    predictions.append(
        Prediction(
            student_id=test_students[4].id,
            at_risk_probability=0.75,
            risk_level="high",
            text_score=0.7,
            tabular_score=0.8,
            behavioral_score=0.6,
        )
    )
    for p in predictions:
        db_session.add(p)
    await db_session.commit()
    for p in predictions:
        await db_session.refresh(p)
    return predictions


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
async def test_get_cohort_happy_path(
    authenticated_client, test_cohort, test_students, test_predictions
):
    response = await authenticated_client.get(f"/cohorts/{test_cohort.id}")
    assert response.status_code == 200
    data = response.json()

    assert data["cohort_id"] == test_cohort.id
    assert data["cohort_name"] == test_cohort.name
    assert data["total_students"] == 5
    assert data["at_risk_count"] == 3
    assert data["at_risk_percentage"] == 60.0
    assert data["risk_distribution"]["low"] == 1
    assert data["risk_distribution"]["medium"] == 1
    assert data["risk_distribution"]["high"] == 2
    assert data["average_modality_scores"]["text"] is not None
    assert data["average_modality_scores"]["tabular"] is not None
    assert data["average_modality_scores"]["behavioral"] is not None
    assert len(data["students"]) == 5
    assert data["pagination"]["page"] == 1
    assert data["pagination"]["limit"] == 50
    assert data["pagination"]["total_pages"] == 1
    assert data["pagination"]["total_students"] == 5


@pytest.mark.asyncio
async def test_get_cohort_empty(authenticated_client, empty_cohort):
    response = await authenticated_client.get(f"/cohorts/{empty_cohort.id}")
    assert response.status_code == 200
    data = response.json()

    assert data["cohort_id"] == empty_cohort.id
    assert data["cohort_name"] == empty_cohort.name
    assert data["total_students"] == 0
    assert data["at_risk_count"] == 0
    assert data["at_risk_percentage"] == 0.0
    assert data["risk_distribution"]["low"] == 0
    assert data["risk_distribution"]["medium"] == 0
    assert data["risk_distribution"]["high"] == 0
    assert data["average_modality_scores"]["text"] is None
    assert data["average_modality_scores"]["tabular"] is None
    assert data["average_modality_scores"]["behavioral"] is None
    assert len(data["students"]) == 0
    assert data["pagination"]["total_pages"] == 1
    assert data["pagination"]["total_students"] == 0


@pytest.mark.asyncio
async def test_get_cohort_pagination(
    authenticated_client, test_cohort, test_students, test_predictions
):
    response = await authenticated_client.get(
        f"/cohorts/{test_cohort.id}?limit=2&page=1"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["students"]) == 2
    assert data["pagination"]["limit"] == 2
    assert data["pagination"]["total_pages"] == 3
    assert data["pagination"]["total_students"] == 5

    response = await authenticated_client.get(
        f"/cohorts/{test_cohort.id}?limit=2&page=2"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["students"]) == 2

    response = await authenticated_client.get(
        f"/cohorts/{test_cohort.id}?limit=2&page=3"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["students"]) == 1


@pytest.mark.asyncio
async def test_get_cohort_sorting_by_name(
    authenticated_client, test_cohort, test_students, test_predictions
):
    response = await authenticated_client.get(
        f"/cohorts/{test_cohort.id}?sort_by=name&order=asc"
    )
    assert response.status_code == 200
    data = response.json()
    names = [s["name"] for s in data["students"]]
    assert names == sorted(names)

    response = await authenticated_client.get(
        f"/cohorts/{test_cohort.id}?sort_by=name&order=desc"
    )
    assert response.status_code == 200
    data = response.json()
    names = [s["name"] for s in data["students"]]
    assert names == sorted(names, reverse=True)


@pytest.mark.asyncio
async def test_get_cohort_sorting_by_risk(
    authenticated_client, test_cohort, test_students, test_predictions
):
    response = await authenticated_client.get(
        f"/cohorts/{test_cohort.id}?sort_by=risk&order=desc"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["students"][0]["risk_level"] == "high"
    assert data["students"][1]["risk_level"] == "high"
    assert data["students"][2]["risk_level"] == "medium"
    assert data["students"][3]["risk_level"] == "low"
    assert data["students"][4]["risk_level"] is None

    response = await authenticated_client.get(
        f"/cohorts/{test_cohort.id}?sort_by=risk&order=asc"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["students"][0]["risk_level"] == "low"
    assert data["students"][1]["risk_level"] == "medium"
    assert data["students"][2]["risk_level"] == "high"
    assert data["students"][3]["risk_level"] == "high"
    assert data["students"][4]["risk_level"] is None


@pytest.mark.asyncio
async def test_get_cohort_filter_by_risk_level(
    authenticated_client, test_cohort, test_students, test_predictions
):
    response = await authenticated_client.get(
        f"/cohorts/{test_cohort.id}?risk_level=high"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["students"]) == 2
    for s in data["students"]:
        assert s["risk_level"] == "high"
    assert data["pagination"]["total_students"] == 2

    response = await authenticated_client.get(
        f"/cohorts/{test_cohort.id}?risk_level=low"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["students"]) == 1
    assert data["students"][0]["risk_level"] == "low"


@pytest.mark.asyncio
async def test_get_cohort_without_auth(client, test_cohort):
    response = await client.get(f"/cohorts/{test_cohort.id}")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_cohort_not_found(authenticated_client):
    response = await authenticated_client.get("/cohorts/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Cohort not found"
