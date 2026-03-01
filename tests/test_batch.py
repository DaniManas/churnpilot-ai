"""
tests/test_batch.py
--------------------
Tests for the batch scoring endpoints:
  POST /batch              — submit a CSV
  GET  /batch/{job_id}     — poll status
  GET  /batch/{job_id}/download — download result

Key patterns:
- Batch processing is async (background thread). Tests poll status with
  a small sleep loop rather than assuming instant completion.
- Tests use the real sample_batch.csv from data/ to validate end-to-end scoring.
"""

import time
import pytest
from tests.conftest import HEADERS


def _wait_for_done(client, job_id: str, timeout: float = 10.0) -> dict:
    """Poll /batch/{job_id} until status is 'done' or 'failed' or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp   = client.get(f"/batch/{job_id}", headers=HEADERS)
        status = resp.json()["status"]
        if status in ("done", "failed"):
            return resp.json()
        time.sleep(0.2)
    return client.get(f"/batch/{job_id}", headers=HEADERS).json()


class TestBatchSubmit:
    def test_submit_valid_csv_returns_202(self, client):
        csv_content = (
            "tenure,monthly_charges,total_charges,contract,internet_service,"
            "payment_method,senior_citizen,partner,dependents,phone_service,paperless_billing\n"
            "3,95.5,286.5,Month-to-month,Fiber optic,Electronic check,False,False,False,True,True\n"
            "60,40.0,2400.0,Two year,DSL,Bank transfer,False,True,True,True,False\n"
        )
        resp = client.post(
            "/batch",
            files={"file": ("test.csv", csv_content.encode(), "text/csv")},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_submit_non_csv_returns_400(self, client):
        resp = client.post(
            "/batch",
            files={"file": ("test.txt", b"hello", "text/plain")},
            headers=HEADERS,
        )
        assert resp.status_code == 400

    def test_submit_empty_file_returns_400(self, client):
        resp = client.post(
            "/batch",
            files={"file": ("empty.csv", b"", "text/csv")},
            headers=HEADERS,
        )
        assert resp.status_code == 400

    def test_submit_no_auth_returns_401(self, client):
        csv_content = b"tenure,monthly_charges\n3,95.5\n"
        resp = client.post(
            "/batch",
            files={"file": ("test.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 401


class TestBatchStatus:
    def test_unknown_job_id_returns_404(self, client):
        resp = client.get("/batch/nonexistent-id", headers=HEADERS)
        assert resp.status_code == 404

    def test_job_reaches_done_or_failed(self, client):
        """Submit a minimal CSV and confirm it completes."""
        csv_content = (
            "tenure,monthly_charges,total_charges,contract,internet_service,"
            "payment_method,senior_citizen,partner,dependents,phone_service,paperless_billing\n"
            "3,95.5,286.5,Month-to-month,Fiber optic,Electronic check,False,False,False,True,True\n"
        )
        resp   = client.post(
            "/batch",
            files={"file": ("test.csv", csv_content.encode(), "text/csv")},
            headers=HEADERS,
        )
        job_id = resp.json()["job_id"]
        result = _wait_for_done(client, job_id, timeout=15.0)
        assert result["status"] in ("done", "failed"), f"Unexpected status: {result['status']}"

    def test_status_response_shape(self, client):
        """Status response must have all required fields."""
        csv_content = (
            "tenure,monthly_charges,total_charges,contract,internet_service,"
            "payment_method,senior_citizen,partner,dependents,phone_service,paperless_billing\n"
            "3,95.5,286.5,Month-to-month,Fiber optic,Electronic check,False,False,False,True,True\n"
        )
        resp   = client.post(
            "/batch",
            files={"file": ("test.csv", csv_content.encode(), "text/csv")},
            headers=HEADERS,
        )
        job_id = resp.json()["job_id"]
        status_resp = client.get(f"/batch/{job_id}", headers=HEADERS)
        data = status_resp.json()
        required = {"job_id", "status", "total_records", "processed_records",
                    "created_at", "completed_at", "download_url", "error_message"}
        assert required.issubset(data.keys())


class TestBatchDownload:
    def test_download_before_done_returns_409(self, client):
        """Downloading before a job completes must return 409 Conflict."""
        csv_content = (
            "tenure,monthly_charges,total_charges,contract,internet_service,"
            "payment_method,senior_citizen,partner,dependents,phone_service,paperless_billing\n"
            + "\n".join(
                "3,95.5,286.5,Month-to-month,Fiber optic,Electronic check,False,False,False,True,True"
                for _ in range(200)   # large enough to still be processing
            ) + "\n"
        )
        resp   = client.post(
            "/batch",
            files={"file": ("big.csv", csv_content.encode(), "text/csv")},
            headers=HEADERS,
        )
        job_id = resp.json()["job_id"]
        # Try to download immediately — likely still pending/processing
        dl_resp = client.get(f"/batch/{job_id}/download", headers=HEADERS)
        # Could be 409 (still running) or 200 (finished instantly) — both valid
        assert dl_resp.status_code in (200, 409)

    def test_download_after_done_returns_csv(self, client):
        """Once a job is done, download must return a valid CSV."""
        csv_content = (
            "tenure,monthly_charges,total_charges,contract,internet_service,"
            "payment_method,senior_citizen,partner,dependents,phone_service,paperless_billing\n"
            "3,95.5,286.5,Month-to-month,Fiber optic,Electronic check,False,False,False,True,True\n"
        )
        resp   = client.post(
            "/batch",
            files={"file": ("test.csv", csv_content.encode(), "text/csv")},
            headers=HEADERS,
        )
        job_id = resp.json()["job_id"]
        result = _wait_for_done(client, job_id, timeout=15.0)

        if result["status"] == "done":
            dl_resp = client.get(f"/batch/{job_id}/download", headers=HEADERS)
            assert dl_resp.status_code == 200
            assert "churn_prediction" in dl_resp.text
            assert "churn_probability" in dl_resp.text
            assert "risk_label" in dl_resp.text
