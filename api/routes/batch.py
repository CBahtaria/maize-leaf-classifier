"""POST /predict/batch — batch image classification (up to 10 images)."""
import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile

from api.config import settings
from api.dependencies import get_model, get_model_meta
from api.middleware.rate_limit import BATCH_LIMIT, limiter
from api.middleware.validation import validate_image_file
from api.schemas import BatchPredictionResponse, PredictionResponse
from model.predict import predict_image

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_BATCH_SIZE = 10


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Classify up to 10 maize leaf images in a single request",
    responses={
        400: {"description": "Invalid image in batch"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Too many images (max 10)"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Model not loaded"},
    },
)
@limiter.limit(BATCH_LIMIT)
async def predict_batch(
    request: Request,
    response: Response,
    files: list[UploadFile] = File(..., description=f"Up to {MAX_BATCH_SIZE} leaf images"),
    model=Depends(get_model),
    model_meta: dict = Depends(get_model_meta),
) -> BatchPredictionResponse:
    """Classify multiple maize leaf images in a single request.

    - Accepts up to 10 images per request
    - Each image must be JPEG, PNG, or WebP, max 10 MB
    - Predictions are returned in the same order as the uploaded files
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Too many images. Maximum batch size: {MAX_BATCH_SIZE}",
        )
    if len(files) == 0:
        raise HTTPException(status_code=422, detail="No images provided")

    start_total = time.perf_counter()
    predictions: list[PredictionResponse] = []
    version = model_meta.get("version", "unknown")

    for upload_file in files:
        sanitised_bytes = await validate_image_file(upload_file, max_bytes=settings.max_file_size_bytes)
        result = predict_image(model, sanitised_bytes)
        predictions.append(PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            processing_time_ms=result["processing_time_ms"],
            model_version=version,
        ))

    total_ms = (time.perf_counter() - start_total) * 1000
    return BatchPredictionResponse(
        predictions=predictions,
        total_images=len(predictions),
        total_time_ms=round(total_ms, 2),
    )
