"""
tests/test_health.py
--------------------
Tests for GET /health.

Key things to verify:
- Returns 200 with no auth required
- Response shape matches HealthResponse schema
- Model is loaded (model_version is not "unknown")
- DB is connected
"""

import pytest
from tests.conftest import HEADERS


class TestHealth:
    def test_health_no_auth_required(self, client):
        """Health check must work without an API key."""
        resp = client.get("/health")
        assert resp.status_code == 200, "Expected 200, got: " + str(resp.text)

    def test_health_response_shape(self, client):
        """Response must contain all required fields."""
        resp = client.get("/health")
        data = resp.json()
        required = {"status", "app_version", "model_version",
                    "db_connected", "rag_index_ready", "uptime_seconds", "total_requests"}
        assert required.issubset(data.keys()), f"Missing fields: {required - data.keys()}"

    def test_health_status_ok(self, client):
        """Status should be 'ok' when DB and model are both up."""
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["db_connected"] is True
        assert data["model_version"] != "unknown"

    def test_health_uptime_positive(self, client):
        """Uptime must be a positive number."""
        resp = client.get("/health")
        assert resp.json()["uptime_seconds"] > 0
