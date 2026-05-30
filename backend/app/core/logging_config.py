import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger

from app.core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that adds standard fields to all log records."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        if "timestamp" not in log_record:
            log_record["timestamp"] = self.formatTime(record)
        if "level" not in log_record:
            log_record["level"] = record.levelname

        log_record["environment"] = settings.ENVIRONMENT
        log_record["logger"] = record.name

        sensitive_keys = {"password", "token", "jwt", "secret", "authorization"}
        for key in list(log_record.keys()):
            if any(s in key.lower() for s in sensitive_keys):
                log_record[key] = "[REDACTED]"


def setup_logging(log_level: str | None = None) -> None:
    """Configure structured JSON logging for the application.

    In production, logs are formatted as JSON for CloudWatch ingestion.
    In local development, plain text is used for readability.
    """
    if log_level is None:
        log_level = "DEBUG" if settings.ENVIRONMENT == "local" else "INFO"

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))

    if settings.ENVIRONMENT in ("local", "dev", "development"):
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s "
            "%(request_id)s %(endpoint)s %(response_time)s %(user_id)s "
            "%(environment)s %(logger)s",
            rename_fields={
                "levelname": "level",
                "asctime": "timestamp",
            },
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
