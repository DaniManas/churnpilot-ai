from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base


class PredictionLog(Base):
    """
    One row per prediction — both /predict (single) and /batch (bulk) write here.
    This is what powers the /monitor endpoint later.
    """
    __tablename__ = "prediction_logs"

    id               = Column(Integer, primary_key=True, index=True)
    request_id       = Column(String, index=True, nullable=False)
    timestamp        = Column(DateTime(timezone=True), server_default=func.now())
    endpoint         = Column(String, nullable=False)          # "predict" or "batch"
    input_features   = Column(JSON, nullable=False)            # raw input dict
    prediction       = Column(Integer, nullable=False)         # 0 or 1
    probability      = Column(Float, nullable=False)           # 0.0 – 1.0
    risk_label       = Column(String, nullable=False)          # Low / Medium / High
    model_version    = Column(String, nullable=False)
    latency_ms       = Column(Float, nullable=False)


class BatchJob(Base):
    """
    Tracks the lifecycle of a batch scoring job.
    Status flow: pending → processing → done (or failed)
    """
    __tablename__ = "batch_jobs"

    id                = Column(Integer, primary_key=True, index=True)
    job_id            = Column(String, unique=True, index=True, nullable=False)
    status            = Column(String, nullable=False, default="pending")
    created_at        = Column(DateTime(timezone=True), server_default=func.now())
    completed_at      = Column(DateTime(timezone=True), nullable=True)
    total_records     = Column(Integer, nullable=True)
    processed_records = Column(Integer, default=0)
    result_path       = Column(String, nullable=True)   # path to output CSV
    error_message     = Column(String, nullable=True)


class AssistLog(Base):
    """
    One row per /assist call.
    Stores IDs and scores (NOT full chunk text) to keep the DB lean.
    Full text lives in the FAISS docstore on disk.
    """
    __tablename__ = "assist_logs"

    id                    = Column(Integer, primary_key=True, index=True)
    request_id            = Column(String, index=True, nullable=False)
    timestamp             = Column(DateTime(timezone=True), server_default=func.now())
    question              = Column(String, nullable=False)
    retrieved_chunk_ids   = Column(JSON, nullable=False)   # ["chunk_3", "chunk_7", ...]
    retrieved_doc_ids     = Column(JSON, nullable=False)   # ["guide", "dict", ...]
    top_chunk_snippet     = Column(String, nullable=True)  # first 100 chars of top hit
    top_score             = Column(Float, nullable=True)   # similarity score of top hit
    retrieval_latency_ms  = Column(Float, nullable=False)
    llm_latency_ms        = Column(Float, nullable=False)
    response_length       = Column(Integer, nullable=False)
    model_used            = Column(String, nullable=False)
    embedding_provider    = Column(String, nullable=False)
