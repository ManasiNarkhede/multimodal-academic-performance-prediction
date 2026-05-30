import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from app.api.health import router as health_router


test_app = FastAPI()
test_app.include_router(health_router)


@pytest_asyncio.fixture
async def health_client():
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check_returns_200_with_required_fields(health_client):
    response = await health_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "db" in data
    assert "models" in data
    assert "memory_mb" in data
    assert isinstance(data["memory_mb"], int)
    assert data["memory_mb"] > 0
    assert "cpu_percent" in data
    assert "memory_percent" in data
    assert "disk_percent" in data
    assert isinstance(data["cpu_percent"], float)
    assert isinstance(data["memory_percent"], float)
    assert isinstance(data["disk_percent"], float)


@pytest.mark.asyncio
async def test_info_returns_model_metadata(health_client):
    response = await health_client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "environment" in data
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    for model in data["models"]:
        assert "name" in model
        assert "version" in model
        assert "last_trained" in model
        assert "metrics" in model


@pytest.mark.asyncio
async def test_models_list_returns_all_four_model_types(health_client):
    response = await health_client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    model_names = {m["name"] for m in data["models"]}
    expected = {"text_model", "tabular_model", "behavioral_model", "fusion_model"}
    assert expected.issubset(model_names)
    for model in data["models"]:
        assert "name" in model
        assert "version" in model
        assert "last_trained" in model
        assert "status" in model
        assert model["status"] in ("active", "deprecated")


@pytest.mark.asyncio
async def test_health_check_detects_missing_model_files(monkeypatch, health_client):
    import app.api.health as health_module

    original_model_files = health_module.MODEL_FILES.copy()
    monkeypatch.setattr(
        health_module,
        "MODEL_FILES",
        {
            "text_model": health_module.PROJECT_ROOT / "nonexistent" / "text_model_latest.pt",
            "tabular_model": health_module.PROJECT_ROOT / "nonexistent" / "tabular_model_latest.json",
            "behavioral_model": health_module.PROJECT_ROOT / "nonexistent" / "behavioral_model_latest.pt",
            "fusion_model": health_module.PROJECT_ROOT / "nonexistent" / "fusion_model_latest.json",
        },
    )

    response = await health_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["models"] == "error"
    assert data["status"] == "degraded"

    monkeypatch.setattr(health_module, "MODEL_FILES", original_model_files)
