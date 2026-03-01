from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class BatchJobResponse(BaseModel):
    """Returned immediately when a batch job is submitted — before processing starts."""
    job_id:     str
    status:     str       # always "pending" at creation time
    message:    str
    created_at: datetime


class BatchStatusResponse(BaseModel):
    """Returned when polling /batch/{job_id} — shows current progress."""
    job_id:            str
    status:            str              # pending / processing / done / failed
    total_records:     Optional[int]   # filled in once the CSV is parsed
    processed_records: int
    created_at:        datetime
    completed_at:      Optional[datetime]
    download_url:      Optional[str]   # available only when status == "done"
    error_message:     Optional[str]   # available only when status == "failed"
