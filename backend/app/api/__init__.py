from fastapi import APIRouter

from app.core.config import settings
from app.api.stubs import router as stubs_router
from app.api.predictions import router as predictions_router


def register_stubs(parent_router: APIRouter) -> None:
    """Register stub routes only in non-production environments."""
    if settings.ENVIRONMENT != "production":
        parent_router.include_router(stubs_router)
