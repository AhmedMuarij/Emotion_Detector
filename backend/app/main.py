"""EmotionAI FastAPI application entry point.

Startup sequence:
    1. Parse settings from environment variables / .env file.
    2. Load Keras model + metadata into memory (once, at startup).
    3. If model loading fails, the application exits immediately with
       a clear error message — no silent fallback to a broken state.
    4. Register CORS middleware.
    5. Mount API router.

The application is intentionally stateless per-request:
- No session state.
- No database connection for MVP.
- No image storage.

To start locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

For production (Render/Railway):
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.ml import loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model at startup; nothing to clean up on shutdown."""
    logger.info("EmotionAI backend starting up...")
    logger.info("Model path:    %s", settings.model_path)
    logger.info("Metadata path: %s", settings.model_metadata_path)
    logger.info("CORS origins:  %s", settings.cors_origins_list)

    try:
        loader.load_model(
            model_path=settings.model_path,
            metadata_path=settings.model_metadata_path,
        )
    except RuntimeError as exc:
        logger.critical("FATAL: Model loading failed — %s", exc)
        logger.critical(
            "The server will continue but /predict will return 503 until the model is available."
        )
        # We don't sys.exit() here so /health still responds and deployment
        # health checks don't immediately kill the container.
        # The loader.is_loaded() flag will be False, causing /predict to 503.

    yield
    logger.info("EmotionAI backend shutting down.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="EmotionAI API",
    description=(
        "Real-time facial expression recognition API. "
        "Classifies visible facial expressions (Angry, Happy, Sad, Surprise, Neutral). "
        "Does not determine internal emotional state."
    ),
    version="0.1.0",
    lifespan=lifespan,
    # In production, set docs_url=None and redoc_url=None to hide API docs.
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,     # No cookies/auth in MVP
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(router)
