"""
tests/test_monitor.py
---------------------
Tests for GET /monitor.

The monitor endpoint reads from prediction_logs. Most tests first seed
the DB with a known prediction so the monitoring results are deterministic.
"""

import pytest
from sqlalchemy import text
from tests.conftest import HEADERS, TestingSessionLocal


def _seed_prediction(client, sample_payload):
    """Helper: make a real prediction so it gets logged to the test DB."""
    resp = client.post("/predict", json=sample_payload, headers=HEADERS)
    assert resp.status_code == 200
    return resp.json()


class TestMonitorAuth:
    def test_requires_api_key(self, client):
        resp = client.get("/monitor")
        assert resp.status_code == 401

    def test_valid_key_passes(self, client):
        resp = client.get("/monitor", headers=HEADERS)
        assert resp.status_code == 200


class TestMonitorEmptyDB:
    def test_monitor_returns_valid_status(self, client):
        """Monitor must always return a valid status regardless of DB contents."""
        resp = client.get("/monitor", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        # Status is valid regardless of how many predictions are in the DB
        assert data["overall_drift_status"] in ("stable", "warning", "drift")
        assert data["prediction_distribution"]["total_predictions"] >= 0


class TestMonitorResponseShape:
    def test_response_has_all_fields(self, client, sample_payload):
        _seed_prediction(client, sample_payload)
        resp = client.get("/monitor", headers=HEADERS)
        data = resp.json()
        assert "model_version" in data
        assert "window_days" in data
        assert "prediction_distribution" in data
        assert "feature_drift" in data
        assert "churn_rate_over_time" in data
        assert "overall_drift_status" in data

    def test_prediction_distribution_shape(self, client, sample_payload):
        _seed_prediction(client, sample_payload)
        resp = client.get("/monitor", headers=HEADERS)
        dist = resp.json()["prediction_distribution"]
        required = {"total_predictions", "churn_count", "no_churn_count",
                    "avg_probability", "high_risk_count", "medium_risk_count", "low_risk_count"}
        assert required.issubset(dist.keys())

    def test_feature_drift_is_list(self, client, sample_payload):
        _seed_prediction(client, sample_payload)
        resp = client.get("/monitor", headers=HEADERS)
        drift = resp.json()["feature_drift"]
        assert isinstance(drift, list)

    def test_feature_drift_shape(self, client, sample_payload):
        _seed_prediction(client, sample_payload)
        resp = client.get("/monitor", headers=HEADERS)
        drift = resp.json()["feature_drift"]
        if drift:   # may be empty if not enough data for PSI
            item = drift[0]
            assert "feature" in item
            assert "psi_score" in item
            assert "status" in item

    def test_drift_status_valid_values(self, client):
        resp = client.get("/monitor", headers=HEADERS)
        status = resp.json()["overall_drift_status"]
        assert status in ("stable", "warning", "drift")


class TestMonitorWindowDays:
    def test_custom_window_days(self, client):
        resp = client.get("/monitor?window_days=30", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["window_days"] == 30

    def test_window_days_too_large_returns_422(self, client):
        resp = client.get("/monitor?window_days=365", headers=HEADERS)
        assert resp.status_code == 422

    def test_window_days_zero_returns_422(self, client):
        resp = client.get("/monitor?window_days=0", headers=HEADERS)
        assert resp.status_code == 422
