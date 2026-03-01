"""
app/services/batch_service.py
------------------------------
Handles async batch scoring in a background thread.

Why a background thread and not FastAPI BackgroundTasks?
- FastAPI BackgroundTasks run after the response is sent but still block
  the event loop for CPU-bound work. A real thread via ThreadPoolExecutor
  keeps the event loop free.
- For very large jobs (100k+ rows) you'd use Celery + Redis. For a
  single-instance platform, a thread pool is the right tradeoff.

Job lifecycle: pending → processing → done / failed
"""

import io
import uuid
import logging
import threading
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models.db_models import BatchJob, PredictionLog
from app.services.ml_service import ml_service

logger = logging.getLogger(__name__)

# Output directory for scored CSVs
RESULTS_DIR = Path("data/batch_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Thread pool — limits concurrent batch jobs
_executor_lock = threading.Semaphore(3)   # max 3 concurrent batch jobs


def create_job(db: Session) -> BatchJob:
    """
    Create a new BatchJob row in the DB and return it.
    Called synchronously before the background thread starts.
    """
    job = BatchJob(
        job_id=str(uuid.uuid4()),
        status="pending",
        processed_records=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _process_job(job_id: str, csv_bytes: bytes):
    """
    Runs in a background thread.
    Opens its own DB session (the request session is closed by this point).
    """
    # Each thread needs its own DB session
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    ThreadSession = sessionmaker(bind=engine)
    db = ThreadSession()

    with _executor_lock:
        try:
            # Mark as processing
            job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
            job.status = "processing"
            db.commit()

            # Parse CSV
            df = pd.read_csv(io.BytesIO(csv_bytes))
            job.total_records = len(df)
            db.commit()

            # Score
            scored_df = ml_service.predict_batch(df)
            job.processed_records = len(scored_df)

            # Save result CSV
            result_path = RESULTS_DIR / f"{job_id}.csv"
            scored_df.to_csv(result_path, index=False)

            # Log each prediction to prediction_logs for monitoring
            for _, row in scored_df.iterrows():
                log = PredictionLog(
                    request_id=f"{job_id}_{row.name}",
                    endpoint="batch",
                    input_features={
                        col: row[col]
                        for col in ml_service.pipeline.feature_names_in_
                        if col in row.index
                    } if hasattr(ml_service.pipeline, "feature_names_in_") else {},
                    prediction=int(row["churn_prediction"]),
                    probability=float(row["churn_probability"]),
                    risk_label=str(row["risk_label"]),
                    model_version=ml_service.model_version,
                    latency_ms=0.0,   # not meaningful for batch
                )
                db.add(log)
            db.commit()

            # Mark done
            job.status = "done"
            job.result_path = str(result_path)
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

            logger.info(f"Batch job {job_id} complete — {len(scored_df)} rows scored")

        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {e}")
            try:
                job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
            except Exception:
                pass
        finally:
            db.close()


def submit_job(job_id: str, csv_bytes: bytes):
    """
    Launch the batch job in a daemon thread so it doesn't block the API.
    """
    t = threading.Thread(
        target=_process_job,
        args=(job_id, csv_bytes),
        daemon=True,
        name=f"batch-{job_id[:8]}",
    )
    t.start()
    logger.info(f"Batch job {job_id} submitted — thread started")
