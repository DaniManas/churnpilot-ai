"""
app/main.py
-----------
FastAPI application entry point.

Creates the app, registers all routers, and wires up startup/shutdown events.
Run with:  uvicorn app.main:app --reload
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_tables

# ── Routers ───────────────────────────────────────────────────────────────────
from app.routers import health, predict, batch, monitor, assist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Track app start time so /health can report uptime
APP_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown logic.
    The `yield` separates startup (before) from shutdown (after).
    This is the modern FastAPI replacement for @app.on_event("startup").
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    create_tables()   # create DB tables if they don't exist yet
    logger.info("Database tables ready")
    yield
    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — goodbye")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready ML platform for customer churn prediction. "
        "Includes single prediction, async batch scoring, model monitoring, "
        "and a RAG-based knowledge assistant."
    ),
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allows any origin in dev. Lock this down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(batch.router)
app.include_router(monitor.router)
app.include_router(assist.router)

# Expose start time so /health can compute uptime
app.state.start_time = APP_START_TIME
