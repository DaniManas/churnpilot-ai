"""
app/routers/monitor.py
----------------------
GET /monitor — model monitoring report.

Returns prediction distribution and PSI drift scores for the past N days.
Used by ML engineers to catch data drift before model accuracy degrades.
"""

import logging
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.schemas.monitor import MonitorResponse
from app.services import monitor_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/monitor",
    response_model=MonitorResponse,
    summary="Model monitoring — drift detection report",
    description=(
        "Returns prediction distribution and PSI drift scores for each feature "
        "over the past `window_days` days. PSI > 0.2 indicates significant drift."
    ),
)
def get_monitor_report(
    window_days: int = Query(default=7, ge=1, le=90, description="Days of history to analyse"),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    return monitor_service.get_monitor_report(db=db, window_days=window_days)
