from pydantic import BaseModel, Field
from typing import List


class Citation(BaseModel):
    """One retrieved chunk that was used to build the answer."""
    doc_id:  str    # e.g. "churn_model_guide"
    chunk_id: str   # e.g. "churn_model_guide_chunk_3"
    source:  str    # human-readable filename
    snippet: str    # first ~100 chars of the chunk
    score:   float  # similarity score (higher = more relevant)


class AssistRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Your question in plain English")
    top_k:    int  = Field(default=5, ge=1, le=10, description="Number of chunks to retrieve")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What does a high churn score mean?",
                "top_k": 5
            }
        }
    }


class AssistResponse(BaseModel):
    request_id:           str
    answer:               str
    citations:            List[Citation]
    retrieval_latency_ms: float
    llm_latency_ms:       float
    model_used:           str
