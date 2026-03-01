"""
app/services/rag_service.py
----------------------------
RAG (Retrieval-Augmented Generation) service for the /assist endpoint.

Architecture:
  1. User question → embed with same method used at index build time
  2. FAISS flat L2 index → retrieve top-k most similar document chunks
  3. Retrieved chunks → prompt → LLM (OpenAI or local fallback) → answer
  4. Return answer + citations (chunk_id, source, snippet, score)

Embedding strategies (set EMBEDDING_PROVIDER in .env):
  local:  TF-IDF + SVD pipeline saved at index build time (offline, zero cost)
  openai: OpenAI text-embedding-3-small (higher quality, requires API key)

The embedder.pkl saved by scripts/build_index.py is loaded alongside the
FAISS index — this guarantees queries are embedded in exactly the same vector
space as the indexed documents.

Graceful degradation:
- If the FAISS index hasn't been built yet, returns a clear message.
- If OpenAI is not configured, falls back to a template answer from top chunk.
"""

import pickle
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple

from app.config import settings
from app.schemas.assist import Citation

logger = logging.getLogger(__name__)

# Module-level singletons — lazy-loaded on first /assist request
_faiss_index   = None
_docstore: dict = {}
_embedder      = None   # sklearn Pipeline (TF-IDF) or SentenceTransformer
_embed_method  = None   # "tfidf" | "sentence_transformer" | "openai"
_index_ready   = False


def _load_index():
    """Lazy-load the FAISS index, docstore, and embedder on first /assist call."""
    global _faiss_index, _docstore, _embedder, _embed_method, _index_ready

    faiss_dir     = Path(settings.FAISS_INDEX_PATH)
    index_file    = faiss_dir / "index.faiss"
    docstore_file = faiss_dir / "docstore.pkl"
    embedder_file = faiss_dir / "embedder.pkl"

    if not index_file.exists() or not docstore_file.exists():
        logger.warning(
            "FAISS index not found. Run:  python scripts/build_index.py  "
            "to build it. /assist will return fallback responses until then."
        )
        _index_ready = False
        return

    try:
        import faiss

        _faiss_index = faiss.read_index(str(index_file))

        with open(docstore_file, "rb") as f:
            _docstore = pickle.load(f)

        # Load embedder — TF-IDF pipeline saved at build time
        if embedder_file.exists():
            with open(embedder_file, "rb") as f:
                _embedder = pickle.load(f)
            _embed_method = "tfidf"
            logger.info("Loaded TF-IDF embedder from disk")
        else:
            # Fallback: try sentence-transformers (requires network on first run)
            try:
                from sentence_transformers import SentenceTransformer
                _embedder = SentenceTransformer("all-MiniLM-L6-v2")
                _embed_method = "sentence_transformer"
                logger.info("Loaded sentence-transformer embedder")
            except Exception:
                logger.warning("No embedder available — retrieval disabled")
                _index_ready = False
                return

        _index_ready = True
        logger.info(
            f"RAG ready — {_faiss_index.ntotal} vectors, "
            f"{len(_docstore)} chunks, embedder={_embed_method}"
        )

    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        _index_ready = False


def is_ready() -> bool:
    if not _index_ready:
        _load_index()
    return _index_ready


def _embed_query(question: str) -> np.ndarray:
    """Embed a query using the same method as index build time."""
    if _embed_method == "tfidf":
        vec = _embedder.transform([question]).astype("float32")
        # L2 normalise to match indexed vectors
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.clip(norm, 1e-10, None)

    elif _embed_method == "sentence_transformer":
        vec = _embedder.encode([question], convert_to_numpy=True)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return (vec / np.clip(norm, 1e-10, None)).astype("float32")

    elif _embed_method == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = client.embeddings.create(model="text-embedding-3-small", input=[question])
        vec = np.array([resp.data[0].embedding], dtype="float32")
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.clip(norm, 1e-10, None)

    raise ValueError(f"Unknown embed method: {_embed_method}")


def retrieve(question: str, top_k: int = 3) -> Tuple[List[Citation], float]:
    """
    Embed the question and retrieve the top-k most similar chunks.
    Returns (citations, retrieval_latency_ms).
    """
    if not _index_ready:
        return [], 0.0

    t0 = time.perf_counter()

    query_vec = _embed_query(question)
    distances, indices = _faiss_index.search(query_vec, top_k)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    chunk_ids  = list(_docstore.keys())
    citations  = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunk_ids):
            continue
        chunk_id = chunk_ids[idx]
        chunk    = _docstore[chunk_id]
        text     = chunk.get("text", "")
        # Convert L2 distance to similarity score in (0, 1]
        score    = round(1.0 / (1.0 + float(dist)), 4)

        citations.append(Citation(
            doc_id=chunk.get("doc_id", "unknown"),
            chunk_id=chunk_id,
            source=chunk.get("source", "unknown"),
            snippet=text[:150],
            score=score,
        ))

    return citations, latency_ms


def _low_confidence(citations: List[Citation]) -> bool:
    """
    Heuristic retrieval quality check.
    If the top match is weak, we prefer a controlled fallback over vague answers.
    """
    if not citations:
        return True
    return citations[0].score < 0.28


def _definition_fallback(question: str) -> str | None:
    """Short, deterministic definitions for common platform terms."""
    q = question.lower()

    if "churn prediction" in q or ("churn" in q and "what is" in q):
        return (
            "Churn prediction estimates the probability that a customer will leave "
            "a service in the near future. In this platform, the model returns a "
            "probability (0 to 1), a binary prediction, and a risk label "
            "(Low / Medium / High)."
        )
    if "psi" in q or "population stability index" in q:
        return (
            "PSI (Population Stability Index) measures how much live feature "
            "distributions have shifted from training-time distributions. "
            "Typical thresholds are: <0.10 stable, 0.10-0.20 warning, >0.20 drift."
        )
    if "drift" in q and "what is" in q:
        return (
            "Data drift means incoming production data no longer follows the same "
            "distribution as training data. That can reduce model reliability even "
            "when the service is technically healthy."
        )
    return None


def generate_answer(question: str, citations: List[Citation]) -> Tuple[str, float, str]:
    """
    Generate an answer from retrieved chunks.
    OpenAI -> configured OPENAI_MODEL if enabled, else template from top chunk.
    Returns (answer, llm_latency_ms, model_used).
    """
    t0 = time.perf_counter()

    if not citations:
        return (
            "I couldn't find relevant information in the knowledge base. "
            "Please ensure the document index has been built with:  "
            "python scripts/build_index.py",
            0.0,
            "fallback",
        )

    context = "\n\n".join(
        f"[{c.source} | score={c.score:.3f}]: {c.snippet}" for c in citations
    )

    # If retrieval quality is weak, prefer a concise deterministic definition
    # for common terms instead of an under-grounded LLM response.
    if _low_confidence(citations):
        definition = _definition_fallback(question)
        if definition:
            return definition, round((time.perf_counter() - t0) * 1000, 2), "kb-definition-fallback"

    # ── OpenAI path ────────────────────────────────────────────────────────────
    if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant for an ML churn prediction platform. "
                            "Use the provided context first. "
                            "If context is partial but enough for a high-level answer, answer briefly and "
                            "explicitly note uncertainty. "
                            "If context clearly does not contain the answer, say that directly. "
                            "Do not hallucinate implementation details."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
            return answer, round((time.perf_counter() - t0) * 1000, 2), settings.OPENAI_MODEL
        except Exception as e:
            logger.warning(f"OpenAI call failed, falling back to template: {e}")

    # ── Local fallback — synthesise from retrieved chunks ─────────────────────
    top = citations[0]
    answer = (
        f"Based on the knowledge base ({top.source}): {top.snippet}"
        + (" ..." if len(top.snippet) >= 150 else "")
    )
    return answer, round((time.perf_counter() - t0) * 1000, 2), "local-template"
