"""
app/routers/batch.py
--------------------
POST /batch         — submit a CSV file for batch scoring (returns job_id immediately)
GET  /batch/{job_id} — poll job status
GET  /batch/{job_id}/download — download the scored CSV once complete

Why async batch and not synchronous?
- Scoring 50,000 rows can take seconds to minutes depending on hardware.
- Returning immediately with a job_id lets the client poll at its own pace.
- This is the standard pattern for any ML inference job that isn't real-time.
"""

import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.schemas.batch import BatchJobResponse, BatchStatusResponse
from app.models.db_models import BatchJob
from app.services import batch_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Batch"])


@router.post(
    "/batch",
    response_model=BatchJobResponse,
    summary="Submit a CSV for batch scoring",
    description=(
        "Upload a CSV with customer records. Returns a job_id immediately. "
        "Poll GET /batch/{job_id} to check status. "
        "Download results from GET /batch/{job_id}/download when done."
    ),
)
async def submit_batch(
    file: UploadFile = File(..., description="CSV file with columns matching PredictRequest fields"),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    csv_bytes = await file.read()
    if len(csv_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Create DB record first (synchronous)
    job = batch_service.create_job(db)

    # Launch background thread — returns immediately
    batch_service.submit_job(job.job_id, csv_bytes)

    logger.info(f"Batch job {job.job_id} created for file '{file.filename}'")

    return BatchJobResponse(
        job_id=job.job_id,
        status="pending",
        message=f"Job accepted. Poll GET /batch/{job.job_id} for status.",
        created_at=job.created_at,
    )


@router.get(
    "/batch/{job_id}",
    response_model=BatchStatusResponse,
    summary="Poll batch job status",
)
def get_batch_status(
    job_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    download_url = None
    if job.status == "done":
        download_url = f"/batch/{job_id}/download"

    return BatchStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total_records=job.total_records,
        processed_records=job.processed_records,
        created_at=job.created_at,
        completed_at=job.completed_at,
        download_url=download_url,
        error_message=job.error_message,
    )


@router.get(
    "/batch/{job_id}/download",
    summary="Download scored CSV",
    description="Available only when job status is 'done'.",
)
def download_batch_result(
    job_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not complete yet. Current status: '{job.status}'.",
        )

    result_path = Path(job.result_path)
    if not result_path.exists():
        raise HTTPException(status_code=500, detail="Result file missing on disk.")

    return FileResponse(
        path=str(result_path),
        media_type="text/csv",
        filename=f"scored_{job_id[:8]}.csv",
    )
