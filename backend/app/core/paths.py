"""Centralized path resolution for local dev and Docker environments."""

from pathlib import Path

# In Docker: /app/app/core/paths.py → parents[2] = /app
# In local dev: backend/app/core/paths.py → parents[2] = backend/
APP_ROOT = Path(__file__).resolve().parents[2]

# In local dev: backend/ → parents[3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Models directory
if (APP_ROOT / "models").exists():
    MODELS_DIR = APP_ROOT / "models"
else:
    MODELS_DIR = PROJECT_ROOT / "backend" / "models"

# Processed data directory
if Path("/data/processed").exists():
    PROCESSED_DATA_DIR = Path("/data/processed")
else:
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Raw data directory
if Path("/data/raw").exists():
    RAW_DATA_DIR = Path("/data/raw")
else:
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Evidence and docs (only used locally, not critical in Docker)
EVIDENCE_DIR = PROJECT_ROOT / ".sisyphus" / "evidence"
DOCS_DIR = PROJECT_ROOT / "docs"
