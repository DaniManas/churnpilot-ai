# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install all dependencies in a separate stage so the final image is smaller.
# We copy only the installed packages, not the build tools.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed for some native packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for security — never run production containers as root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY app/       ./app/
COPY ml/        ./ml/
COPY data/docs/ ./data/docs/
COPY scripts/   ./scripts/

# Create directories the app writes to at runtime
# (SQLite DB file + batch results + FAISS index)
RUN mkdir -p data/faiss_index data/batch_results && \
    chown -R appuser:appuser /app

USER appuser

# Build the FAISS index from the bundled docs at image build time
# This means /assist is ready immediately on container start.
# Re-run `docker build` whenever docs change.
RUN python scripts/build_index.py

# Expose the port uvicorn listens on
EXPOSE 8000

# Health check — Docker will restart the container if /health fails repeatedly
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the API server
# --host 0.0.0.0  required inside Docker (default 127.0.0.1 is unreachable from outside)
# --workers 2     two processes for CPU-bound inference; tune to core count
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
