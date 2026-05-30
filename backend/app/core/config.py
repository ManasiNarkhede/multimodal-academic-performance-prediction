from pydantic_settings import BaseSettings
from pydantic import Field, validator
import sys


class Settings(BaseSettings):
    DATABASE_URL: str = Field(default="postgresql+asyncpg://user:password@localhost:5432/academic_db")
    JWT_SECRET_KEY: str = Field(default="your-secret-key-here")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    MLFLOW_TRACKING_URI: str = Field(default="sqlite:///backend/mlflow.db")
    MLFLOW_ARTIFACT_ROOT: str = Field(default="backend/mlruns")
    AWS_S3_BUCKET: str = Field(default="")
    ENVIRONMENT: str = Field(default="local")
    FRONTEND_URL: str = Field(default="http://localhost:3000")
    TEXT_MODEL: str = Field(default="distilbert-base-uncased")
    USE_MINILM_FALLBACK: bool = Field(default=False)

    @validator("JWT_SECRET_KEY")
    def validate_jwt_secret(cls, v):
        if v == "your-secret-key-here" or len(v) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters long and not a default value. "
                "Run: python scripts/generate_secrets.py"
            )
        return v

    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if "user:password@" in v or "changeme" in v:
            raise ValueError(
                "DATABASE_URL contains placeholder credentials. "
                "Update with real database credentials."
            )
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
