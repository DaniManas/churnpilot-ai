from pydantic import BaseModel


class HealthResponse(BaseModel):
    status:          str    # "ok" or "degraded"
    app_version:     str
    model_version:   str
    db_connected:    bool
    rag_index_ready: bool
    uptime_seconds:  float
    total_requests:  int
