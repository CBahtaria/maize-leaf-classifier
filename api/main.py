"""FastAPI application factory.

Endpoints:
  POST /predict          — single image binary classification
  POST /predict/batch    — batch of up to 10 images
  GET  /health           — liveness check
  GET  /model/info       — model metadata

Security:
  - CORS restricted to ALLOWED_ORIGINS
  - Rate limiting via slowapi (20 req/min per IP)
  - File validation in middleware (MIME, size, PIL verify, EXIF strip)
  - Non-root user in Docker
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.config import settings
from api.dependencies import load_model

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    logger.info("Loading model from %s", settings.MODEL_PATH)
    try:
        load_model(settings.MODEL_PATH, settings.MODEL_META_PATH)
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        # Continue startup — health endpoint will report degraded
    app.state.start_time = time.time()
    yield
    logger.info("Shutting down API")


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
        headers={"Retry-After": "60"},
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Maize Leaf Disease Classifier API",
        description="Binary CNN classifier: Healthy vs. Diseased maize leaves for SSA farmers",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    # Routers
    from api.routes.batch import router as batch_router
    from api.routes.health import router as health_router
    from api.routes.predict import router as predict_router

    app.include_router(predict_router)
    app.include_router(batch_router)
    app.include_router(health_router)

    return app


app = create_app()
