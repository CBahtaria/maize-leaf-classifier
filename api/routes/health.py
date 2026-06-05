"""GET /health and GET /model/info — liveness and model metadata endpoints."""
import time

from fastapi import APIRouter, Request

import api.dependencies as deps
from api.schemas import HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API liveness and model load status",
)
async def health(request: Request) -> HealthResponse:
    """Check that the API is running and the model is loaded."""
    model_loaded = deps._model is not None
    meta = deps._model_meta
    uptime = time.time() - getattr(request.app.state, "start_time", time.time())
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=meta.get("version", "unknown"),
        uptime_seconds=round(uptime, 1),
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata: architecture, version, performance metrics",
)
async def model_info(request: Request) -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    meta = deps._model_meta
    metrics = meta.get("metrics", {})
    return ModelInfoResponse(
        architecture=meta.get("arch_name", "unknown"),
        version=meta.get("version", "unknown"),
        tflite_size_mb=meta.get("tflite_size_mb"),
        accuracy=metrics.get("accuracy"),
        sensitivity=metrics.get("sensitivity"),
        specificity=metrics.get("specificity"),
        auc_roc=metrics.get("auc_roc"),
        model_path=meta.get("model_path", "unknown"),
    )
