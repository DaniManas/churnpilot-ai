"""
tests/test_predict.py
---------------------
Tests for POST /predict.

Coverage:
- Happy path: valid high-risk and low-risk inputs
- Auth: missing key, wrong key
- Validation: bad field types, out-of-range values, invalid Literal values
- Response contract: all required fields, correct types, sensible values
- DB logging: prediction is persisted after a successful call
"""

import pytest
from sqlalchemy import text
from tests.conftest import HEADERS, TestingSessionLocal


class TestPredictAuth:
    def test_missing_api_key_returns_401(self, client, sample_payload):
        resp = client.post("/predict", json=sample_payload)
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, client, sample_payload):
        resp = client.post("/predict", json=sample_payload, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_valid_key_passes(self, client, sample_payload):
        resp = client.post("/predict", json=sample_payload, headers=HEADERS)
        assert resp.status_code == 200


class TestPredictHappyPath:
    def test_high_risk_prediction(self, client, sample_payload):
        """Short tenure + month-to-month + fiber + e-check → high churn probability."""
        resp = client.post("/predict", json=sample_payload, headers=HEADERS)
        data = resp.json()
        assert data["prediction"] == 1
        assert data["probability"] > 0.5
        assert data["risk_label"] == "High"

    def test_low_risk_prediction(self, client, low_risk_payload):
        """Long tenure + two-year contract + DSL → low churn probability."""
        resp = client.post("/predict", json=low_risk_payload, headers=HEADERS)
        data = resp.json()
        assert data["prediction"] == 0
        assert data["probability"] < 0.5
        assert data["risk_label"] == "Low"

    def test_response_has_all_fields(self, client, sample_payload):
        resp = client.post("/predict", json=sample_payload, headers=HEADERS)
        data = resp.json()
        required = {"request_id", "prediction", "probability", "risk_label",
                    "model_version", "latency_ms"}
        assert required.issubset(data.keys())

    def test_request_id_is_uuid(self, client, sample_payload):
        import uuid
        resp = client.post("/predict", json=sample_payload, headers=HEADERS)
        uid = resp.json()["request_id"]
        uuid.UUID(uid)   # raises ValueError if not a valid UUID

    def test_probability_between_0_and_1(self, client, sample_payload):
        resp = client.post("/predict", json=sample_payload, headers=HEADERS)
        prob = resp.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_latency_is_positive(self, client, sample_payload):
        resp = client.post("/predict", json=sample_payload, headers=HEADERS)
        assert resp.json()["latency_ms"] > 0

    def test_model_version_returned(self, client, sample_payload):
        resp = client.post("/predict", json=sample_payload, headers=HEADERS)
        assert resp.json()["model_version"] == "1.0.0"


class TestPredictValidation:
    def test_invalid_contract_returns_422(self, client, sample_payload):
        payload = {**sample_payload, "contract": "weekly"}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_invalid_internet_service_returns_422(self, client, sample_payload):
        payload = {**sample_payload, "internet_service": "5G"}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_tenure_out_of_range_returns_422(self, client, sample_payload):
        payload = {**sample_payload, "tenure": 999}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_negative_monthly_charges_returns_422(self, client, sample_payload):
        payload = {**sample_payload, "monthly_charges": -10.0}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_missing_required_field_returns_422(self, client, sample_payload):
        payload = {k: v for k, v in sample_payload.items() if k != "tenure"}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/predict", json={}, headers=HEADERS)
        assert resp.status_code == 422


class TestPredictDBLogging:
    def test_prediction_is_logged_to_db(self, client, sample_payload):
        """Every successful prediction must write a row to prediction_logs."""
        # Make a prediction
        resp = client.post("/predict", json=sample_payload, headers=HEADERS)
        request_id = resp.json()["request_id"]

        # Verify it's in the DB
        db = TestingSessionLocal()
        result = db.execute(
            text("SELECT * FROM prediction_logs WHERE request_id = :rid"),
            {"rid": request_id},
        ).fetchone()
        db.close()

        assert result is not None, "Prediction was not logged to the database"
        assert result.endpoint == "predict"
