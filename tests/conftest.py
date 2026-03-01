"""
tests/conftest.py
-----------------
Shared pytest fixtures for the entire test suite.

Key patterns:
- DATABASE_URL env var is patched to a writable /tmp file BEFORE any app
  code is imported. SQLAlchemy creates the engine at module import time, so
  this ordering is critical — the env var must be set first.
- `client`: a FastAPI TestClient backed by the temp DB.
- Tests never touch the real database.

Why a /tmp file instead of :memory:?
- The lifespan context in main.py calls create_tables() against the engine
  defined in app/database.py (not via the get_db dependency). By patching
  DATABASE_URL before import, both the engine AND the get_db sessions point
  to the same writable temp file automatically.
- :memory: doesn't survive across threads/sessions the same way.
"""

import os
import sys
import tempfile
import pytest

# ── MUST happen before any app imports ────────────────────────────────────────
_db_fd, _db_path = tempfile.mkstemp(suffix=".db", dir="/tmp")
os.close(_db_fd)
os.environ["DATABASE_URL"]   = f"sqlite:///{_db_path}"
os.environ["MODEL_PATH"]     = "ml/artifacts/model.pkl"
os.environ["METADATA_PATH"]  = "ml/artifacts/model_metadata.json"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient   # noqa: E402
from sqlalchemy import create_engine        # noqa: E402
from sqlalchemy.orm import sessionmaker     # noqa: E402

from app.main import app                    # noqa: E402
from app.database import SessionLocal       # noqa: E402

API_KEY = "dev-key-123"
HEADERS = {"X-API-Key": API_KEY}


def TestingSessionLocal():
    """Return a plain DB session for direct assertions in tests."""
    return SessionLocal()


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient — shared across the entire test session."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    # Clean up temp DB after all tests
    try:
        os.unlink(_db_path)
    except OSError:
        pass


@pytest.fixture
def sample_payload() -> dict:
    """A canonical valid /predict request body (high-risk customer)."""
    return {
        "tenure": 3,
        "monthly_charges": 95.5,
        "total_charges": 286.5,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "senior_citizen": False,
        "partner": False,
        "dependents": False,
        "phone_service": True,
        "paperless_billing": True,
    }


@pytest.fixture
def low_risk_payload() -> dict:
    """A canonical valid /predict request body (low-risk customer)."""
    return {
        "tenure": 60,
        "monthly_charges": 40.0,
        "total_charges": 2400.0,
        "contract": "Two year",
        "internet_service": "DSL",
        "payment_method": "Bank transfer",
        "senior_citizen": False,
        "partner": True,
        "dependents": True,
        "phone_service": True,
        "paperless_billing": False,
    }
