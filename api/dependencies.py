"""FastAPI dependency injection: shared model wrapper singleton."""
import json
import logging
from pathlib import Path

from fastapi import HTTPException

from model.predict import ModelWrapper

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once at startup, shared across all requests
_model: ModelWrapper | None = None
_model_meta: dict = {}


def get_model() -> ModelWrapper:
    """Dependency: returns the loaded model singleton. Raises 503 if not loaded."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model


def get_model_meta() -> dict:
    """Dependency: returns model metadata dict."""
    return _model_meta


def load_model(model_path: str, meta_path: str) -> None:
    """Called once at application startup to load model into module-level singleton."""
    global _model, _model_meta
    from model.predict import load_model_for_inference
    _model = load_model_for_inference(model_path)
    meta_file = Path(meta_path)
    if meta_file.exists():
        _model_meta = json.loads(meta_file.read_text())
    else:
        # Graceful degradation: serve with minimal metadata if file missing
        logger.warning("Model meta file not found: %s — using defaults", meta_path)
        _model_meta = {"version": "unknown", "arch_name": "unknown"}
    logger.info("Model loaded: %s  version=%s", model_path, _model_meta.get("version", "?"))
