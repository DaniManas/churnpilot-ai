"""
app/routers/assist.py
---------------------
POST /assist — RAG-based knowledge assistant.

Retrieves relevant document chunks via FAISS, then generates an answer
using either OpenAI or a local template fallback.

Logs every call to assist_logs for observability (chunk IDs + scores,
not full text — keeps the DB lean).
"""

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.schemas.assist import AssistRequest, AssistResponse
from app.models.db_models import AssistLog
from app.services import rag_service
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Assistant"])


@router.post(
    "/assist",
    response_model=AssistResponse,
    summary="RAG-based knowledge assistant",
    description=(
        "Ask a question in plain English. The system retrieves relevant "
        "chunks from the knowledge base and generates an answer. "
        "Citations show which documents were used."
    ),
)
def assist(
    body: AssistRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    request_id = str(uuid.uuid4())

    # Lazy-initialise the RAG index on first call
    if not rag_service.is_ready():
        logger.warning("RAG index not ready — returning degraded response")

    # ── Retrieval ──────────────────────────────────────────────────────────────
    citations, retrieval_latency_ms = rag_service.retrieve(
        question=body.question,
        top_k=body.top_k,
    )

    # ── Generation ────────────────────────────────────────────────────────────
    answer, llm_latency_ms, model_used = rag_service.generate_answer(
        question=body.question,
        citations=citations,
    )

    # ── Persist to DB (lean — IDs and scores only) ─────────────────────────────
    log = AssistLog(
        request_id=request_id,
        question=body.question,
        retrieved_chunk_ids=[c.chunk_id for c in citations],
        retrieved_doc_ids=list({c.doc_id for c in citations}),
        top_chunk_snippet=citations[0].snippet if citations else None,
        top_score=citations[0].score if citations else None,
        retrieval_latency_ms=retrieval_latency_ms,
        llm_latency_ms=llm_latency_ms,
        response_length=len(answer),
        model_used=model_used,
        embedding_provider=settings.EMBEDDING_PROVIDER,
    )
    db.add(log)
    db.commit()

    logger.info(
        f"[{request_id}] assist | chunks={len(citations)} "
        f"retrieval={retrieval_latency_ms}ms llm={llm_latency_ms}ms model={model_used}"
    )

    return AssistResponse(
        request_id=request_id,
        answer=answer,
        citations=citations,
        retrieval_latency_ms=retrieval_latency_ms,
        llm_latency_ms=llm_latency_ms,
        model_used=model_used,
    )
