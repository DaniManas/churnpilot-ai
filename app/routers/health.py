"""
app/routers/health.py
---------------------
GET /health — returns app status, model version, DB connectivity, RAG readiness, uptime.

No auth required — health checks must always be reachable by load balancers
and monitoring systems, even when a bad API key is the reason things are broken.
"""

import time
import logging
from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db
from app.config import settings
from app.schemas.health import HealthResponse
from app.services.ml_service import ml_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health_check(request: Request, db: Session = Depends(get_db)):
    """
    Returns the health of every subsystem:
    - app_version / model_version — for tracing which code is deployed
    - db_connected — confirms DB is reachable (catches SQLite file permission issues)
    - rag_index_ready — confirms FAISS index has been built
    - uptime_seconds — time since the process started
    - total_requests — total predictions logged (quick proxy for traffic volume)
    """
    # ── DB connectivity ────────────────────────────────────────────────────────
    db_ok = False
    total_requests = 0
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
        # Count total predictions as a cheap activity metric
        result = db.execute(text("SELECT COUNT(*) FROM prediction_logs"))
        total_requests = result.scalar() or 0
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")

    # ── RAG index readiness ────────────────────────────────────────────────────
    from pathlib import Path
    faiss_path = Path(settings.FAISS_INDEX_PATH)
    rag_ready = faiss_path.exists() and any(faiss_path.iterdir()) if faiss_path.exists() else False

    # ── Uptime ────────────────────────────────────────────────────────────────
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = round(time.time() - start_time, 2)

    overall_status = "ok" if (db_ok and ml_service.pipeline is not None) else "degraded"

    return HealthResponse(
        status=overall_status,
        app_version=settings.APP_VERSION,
        model_version=ml_service.model_version,
        db_connected=db_ok,
        rag_index_ready=rag_ready,
        uptime_seconds=uptime,
        total_requests=total_requests,
    )
