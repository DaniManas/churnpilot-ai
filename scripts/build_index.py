"""
scripts/build_index.py
-----------------------
Builds the FAISS vector index from documents in data/docs/.

Run this once before starting the API server, and re-run whenever
documents are added or updated.

Usage:
    python scripts/build_index.py [--embedder local|openai]

Embedding strategy (controlled by EMBEDDING_PROVIDER in .env):
  local:  TF-IDF vectors via sklearn — zero network, zero cost, works offline.
          Great for dev and for a small internal knowledge base.
  openai: OpenAI text-embedding-3-small — higher quality, requires API key.

Why TF-IDF for local mode?
- Sentence-transformers (all-MiniLM-L6-v2) is the production choice, but
  it requires downloading ~80MB from HuggingFace on first run.
- TF-IDF runs entirely offline with sklearn, which is already a dependency.
- For a small knowledge base (< 1000 chunks) TF-IDF retrieval quality is
  perfectly usable for keyword-heavy technical docs.
- The FAISS index format is identical regardless of embedding method —
  swapping to sentence-transformers later requires only re-running this script.

Why flat L2 index?
- Exact nearest-neighbour search — no approximation error.
- Fast enough for < 100k chunks.
- No training step required (unlike IVF indices).
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR       = Path("data/docs")
INDEX_DIR      = Path("data/faiss_index")
CHUNK_SIZE     = 400   # characters per chunk
CHUNK_OVERLAP  = 80    # overlap between consecutive chunks


# ── Document loading ──────────────────────────────────────────────────────────

def load_documents() -> list[dict]:
    """Read all .txt and .md files and return as list of {doc_id, source, text}."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix not in (".txt", ".md"):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append({"doc_id": path.stem, "source": path.name, "text": text})
        logger.info(f"Loaded: {path.name} ({len(text):,} chars)")
    return docs


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping character-level chunks.
    Overlap ensures sentences spanning chunk boundaries are always retrievable.
    """
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 20]


def build_chunks(docs: list[dict]) -> tuple[list[str], list[dict]]:
    """Return (texts, metadata_list) across all documents."""
    texts, metas = [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunk_id = f"{doc['doc_id']}_chunk_{i}"
            texts.append(chunk)
            metas.append({
                "chunk_id": chunk_id,
                "doc_id":   doc["doc_id"],
                "source":   doc["source"],
                "text":     chunk,
            })
    return texts, metas


# ── Embedding strategies ──────────────────────────────────────────────────────

def embed_tfidf(texts: list[str]) -> tuple[np.ndarray, object]:
    """
    Embed texts using TF-IDF + SVD (Latent Semantic Analysis).
    SVD reduces to 128 dims — comparable to small neural embeddings.
    Returns (embeddings [N, 128], fitted vectorizer pipeline).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline

    n_components = min(128, len(texts) - 1)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)),
        ("svd",   TruncatedSVD(n_components=n_components, random_state=42)),
    ])
    embeddings = pipe.fit_transform(texts).astype("float32")
    # L2 normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, None)
    logger.info(f"TF-IDF embeddings: {embeddings.shape}")
    return embeddings, pipe


def embed_openai(texts: list[str]) -> tuple[np.ndarray, None]:
    """Embed texts via OpenAI text-embedding-3-small (requires OPENAI_API_KEY)."""
    from openai import OpenAI
    from app.config import settings

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp  = client.embeddings.create(model="text-embedding-3-small", input=batch)
        vecs  = [r.embedding for r in resp.data]
        all_embeddings.extend(vecs)
        logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    embeddings = np.array(all_embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, None)
    logger.info(f"OpenAI embeddings: {embeddings.shape}")
    return embeddings, None


# ── Index building ────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray):
    import faiss
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_artifacts(index, docstore: dict, embedder_pipeline, index_dir: Path):
    import faiss
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "docstore.pkl", "wb") as f:
        pickle.dump(docstore, f)
    # Save the embedder pipeline so rag_service can embed queries the same way
    if embedder_pipeline is not None:
        with open(index_dir / "embedder.pkl", "wb") as f:
            pickle.dump(embedder_pipeline, f)

    logger.info(f"Saved index     → {index_dir / 'index.faiss'} ({index.ntotal} vectors)")
    logger.info(f"Saved docstore  → {index_dir / 'docstore.pkl'} ({len(docstore)} chunks)")
    if embedder_pipeline is not None:
        logger.info(f"Saved embedder  → {index_dir / 'embedder.pkl'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedder", choices=["local", "openai"], default="local",
        help="Embedding strategy. 'local' uses TF-IDF (offline). 'openai' uses text-embedding-3-small.",
    )
    args = parser.parse_args()

    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("*.txt")) and not any(DOCS_DIR.glob("*.md")):
        logger.error(f"No .txt or .md documents found in {DOCS_DIR}.")
        sys.exit(1)

    docs = load_documents()
    texts, metas = build_chunks(docs)
    logger.info(f"Total chunks: {len(texts)}")

    if args.embedder == "openai":
        embeddings, pipeline = embed_openai(texts)
    else:
        embeddings, pipeline = embed_tfidf(texts)

    index    = build_faiss_index(embeddings)
    docstore = {m["chunk_id"]: m for m in metas}
    save_artifacts(index, docstore, pipeline, INDEX_DIR)
    logger.info("Done — start the API server and /assist will use this index.")


if __name__ == "__main__":
    main()
