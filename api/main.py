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
import logging
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

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


def _download_model_if_needed() -> None:
    """Download model artifact from MODEL_DOWNLOAD_URL if the local file is missing."""
    if not settings.MODEL_DOWNLOAD_URL:
        return
    dest = Path(settings.MODEL_PATH)
    if dest.exists():
        logger.info("Model already present at %s — skipping download", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model from %s → %s", settings.MODEL_DOWNLOAD_URL, dest)
    urllib.request.urlretrieve(settings.MODEL_DOWNLOAD_URL, dest)
    logger.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1_048_576)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download model if needed, load it, then clean up on shutdown."""
    try:
        _download_model_if_needed()
    except Exception as exc:
        logger.error("Model download failed: %s", exc)
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
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]

    # Routers
    from api.routes.batch import router as batch_router
    from api.routes.health import router as health_router
    from api.routes.predict import router as predict_router

    app.include_router(predict_router)
    app.include_router(batch_router)
    app.include_router(health_router)

    return app


app = create_app()
