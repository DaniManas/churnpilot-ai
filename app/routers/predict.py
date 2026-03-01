"""
app/routers/predict.py
----------------------
POST /predict — single customer churn prediction.

Flow:
  1. Pydantic validates the request body (HTTP 422 on bad input)
  2. API key is verified (HTTP 401 if missing/wrong)
  3. ml_service.predict() runs the pipeline
  4. Result is logged to prediction_logs table
  5. Response is returned with prediction, probability, risk label, latency
"""

import time
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.schemas.predict import PredictRequest, PredictResponse
from app.models.db_models import PredictionLog
from app.services.ml_service import ml_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Single customer churn prediction",
    description=(
        "Submit one customer's details and receive a churn probability, "
        "binary prediction (0/1), and risk tier (Low / Medium / High)."
    ),
)
def predict(
    body: PredictRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """
    The Depends(verify_api_key) line means FastAPI calls verify_api_key()
    before this function runs. If the key is missing or invalid, the request
    is rejected at that point — this function is never reached.

    The Depends(get_db) injects a DB session scoped to this request.
    SQLAlchemy closes it automatically when the request finishes.
    """
    start_ts = time.perf_counter()
    request_id = str(uuid.uuid4())

    try:
        result = ml_service.predict(body.model_dump())
    except Exception as e:
        logger.error(f"Prediction failed for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")

    latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)

    # ── Persist to DB ──────────────────────────────────────────────────────────
    log_entry = PredictionLog(
        request_id=request_id,
        endpoint="predict",
        input_features=body.model_dump(),
        prediction=result["prediction"],
        probability=result["probability"],
        risk_label=result["risk_label"],
        model_version=result["model_version"],
        latency_ms=latency_ms,
    )
    db.add(log_entry)
    db.commit()

    logger.info(
        f"[{request_id}] prob={result['probability']} "
        f"risk={result['risk_label']} latency={latency_ms}ms"
    )

    return PredictResponse(
        request_id=request_id,
        prediction=result["prediction"],
        probability=result["probability"],
        risk_label=result["risk_label"],
        model_version=result["model_version"],
        latency_ms=latency_ms,
    )
