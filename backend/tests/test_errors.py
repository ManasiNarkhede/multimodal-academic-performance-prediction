import pytest
import pytest_asyncio
from fastapi import APIRouter
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient

from app.main import app
from app.core.errors import AuthenticationError, ModelNotFoundError, PredictionError
from app.core.config import settings


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


def _add_temp_route(path: str, handler):
    router = APIRouter()
    router.add_api_route(path, handler)
    app.include_router(router)
    return router


@pytest.mark.asyncio
async def test_invalid_json_returns_400(client):
    response = await client.post(
        "/predictions/predict",
        data="not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Validation error"
    assert "errors" in data


@pytest.mark.asyncio
async def test_nonexistent_cohort_id_returns_404(authenticated_client):
    response = await authenticated_client.get("/cohorts/99999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_malformed_auth_token_returns_401(client):
    response = await client.get(
        "/protected",
        cookies={"access_token": "invalid-token"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_model_not_found_error_handler():
    async def raise_model_not_found():
        raise ModelNotFoundError("Student not found")

    router = _add_temp_route("/test-model-not-found", raise_model_not_found)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/test-model-not-found")
    assert response.status_code == 404
    assert response.json()["detail"] == "Student not found"


@pytest.mark.asyncio
async def test_prediction_error_handler():
    async def raise_prediction_error():
        raise PredictionError("Model failed")

    router = _add_temp_route("/test-prediction-error", raise_prediction_error)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/test-prediction-error")
    assert response.status_code == 500
    assert "prediction" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_authentication_error_handler():
    async def raise_auth_error():
        raise AuthenticationError("Invalid credentials")

    router = _add_temp_route("/test-auth-error", raise_auth_error)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/test-auth-error")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"
    assert response.headers.get("www-authenticate") == "Bearer"


def test_generic_exception_in_local_shows_error():
    async def raise_generic():
        raise RuntimeError("Something broke")

    router = _add_temp_route("/test-generic-local", raise_generic)
    original_env = settings.ENVIRONMENT
    settings.ENVIRONMENT = "local"
    try:
        with TestClient(app, raise_server_exceptions=False) as tc:
            response = tc.get("/test-generic-local")
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error"
        assert "error" in data
        assert "Something broke" in data["error"]
    finally:
        settings.ENVIRONMENT = original_env


def test_generic_exception_in_production_hides_stack_trace():
    async def raise_generic():
        raise RuntimeError("Something broke")

    router = _add_temp_route("/test-generic-prod", raise_generic)
    original_env = settings.ENVIRONMENT
    settings.ENVIRONMENT = "production"
    try:
        with TestClient(app, raise_server_exceptions=False) as tc:
            response = tc.get("/test-generic-prod")
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error"
        assert "error" not in data
    finally:
        settings.ENVIRONMENT = original_env


@pytest.mark.asyncio
async def test_cors_preflight_passes_from_frontend_origin(client):
    response = await client.options(
        "/auth/login",
        headers={
            "Origin": settings.FRONTEND_URL,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert settings.FRONTEND_URL in (response.headers.get("access-control-allow-origin") or "")


@pytest.mark.asyncio
async def test_cors_rejects_unauthorized_origin(client):
    response = await client.options(
        "/auth/login",
        headers={
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 400
