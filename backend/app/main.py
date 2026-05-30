from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.auth import router as auth_router
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.api import register_stubs
from app.api.predictions import router as predictions_router, limiter
from app.api.cohorts import router as cohorts_router
from app.api.health import router as health_router
from app.core.config import settings
from app.core.errors import AuthenticationError, ModelNotFoundError, PredictionError
from app.middleware.error_handler import (
    validation_exception_handler,
    model_not_found_handler,
    prediction_error_handler,
    authentication_error_handler,
    generic_exception_handler,
)
from app.core.logging_config import setup_logging
from app.middleware.logging_middleware import LoggingMiddleware

setup_logging()

app = FastAPI(
    title="Academic Performance Prediction System",
    description="API for predicting academic performance using ML models",
    version="0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(ModelNotFoundError, model_not_found_handler)  # type: ignore[arg-type]
app.add_exception_handler(PredictionError, prediction_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(AuthenticationError, authentication_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, generic_exception_handler)  # type: ignore[arg-type]

app.add_middleware(LoggingMiddleware)

_cors_origins = [
    o.strip()
    for o in settings.FRONTEND_URL.split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(predictions_router)
app.include_router(cohorts_router)
app.include_router(health_router)

# Stubs are registered only in non-production environments for frontend development
register_stubs(app)


@app.get("/protected")
async def protected_endpoint(current_user: User = Depends(get_current_user)):
    return {"message": "Access granted", "user_id": current_user.id, "email": current_user.email}
