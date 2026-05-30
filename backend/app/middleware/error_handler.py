from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.errors import AuthenticationError, ModelNotFoundError, PredictionError
from app.core.config import settings


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg", ""),
            "type": error.get("type", ""),
        })
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": "Validation error",
            "errors": errors,
        },
    )


async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": exc.message},
    )


async def prediction_error_handler(request: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An error occurred while processing the prediction. Please try again later."},
    )


async def authentication_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": exc.message},
        headers={"WWW-Authenticate": "Bearer"},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if settings.ENVIRONMENT == "local":
        content = {
            "detail": "Internal server error",
            "error": str(exc),
        }
    else:
        content = {"detail": "Internal server error"}
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content,
    )
