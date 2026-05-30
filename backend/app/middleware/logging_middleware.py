import logging
import time
import uuid
from typing import Any

from jose import jwt

from app.core.config import settings

logger = logging.getLogger("app.middleware")
SLOW_RESPONSE_THRESHOLD = 1.0


def _extract_user_id_from_headers(headers: list[list[bytes]]) -> int | None:
    cookie_header = None
    for name, value in headers:
        if name.lower() == b"cookie":
            cookie_header = value.decode("utf-8")
            break

    if not cookie_header:
        return None

    access_token = None
    for cookie in cookie_header.split(";"):
        cookie = cookie.strip()
        if cookie.startswith("access_token="):
            access_token = cookie[len("access_token="):]
            break

    if not access_token:
        return None

    try:
        payload = jwt.decode(
            access_token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        user_id = payload.get("sub")
        return int(user_id) if user_id is not None else None
    except Exception:
        return None


class LoggingMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if path == "/health":
            await self.app(scope, receive, send)
            return

        method: str = scope.get("method", "")
        request_id = str(uuid.uuid4())
        scope["request_id"] = request_id

        headers = scope.get("headers", [])
        user_id = _extract_user_id_from_headers(headers)

        start_time = time.time()
        status_code = 200
        exception_occurred = False

        async def wrapped_send(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        except Exception:
            exception_occurred = True
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time
            log_data = {
                "request_id": request_id,
                "endpoint": f"{method} {path}",
                "response_time": round(duration, 4),
                "user_id": user_id,
                "status_code": status_code,
            }

            if exception_occurred:
                logger.error("Request failed", extra=log_data)
            elif duration > SLOW_RESPONSE_THRESHOLD:
                logger.warning("Slow response", extra=log_data)
            else:
                logger.info("Request completed", extra=log_data)
