"""POST /predict — single image binary classification endpoint."""

import logging

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile

from api.config import settings
from api.dependencies import get_model, get_model_meta, verify_api_key
from api.middleware.rate_limit import PREDICT_LIMIT, limiter
from api.middleware.validation import validate_image_file
from api.schemas import PredictionResponse
from model.predict import predict_image

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a maize leaf image as Healthy or Diseased",
    responses={
        400: {"description": "Invalid or corrupt image"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Model not loaded"},
    },
)
@limiter.limit(PREDICT_LIMIT)
async def predict(
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="Leaf image (JPEG/PNG/WebP, max 10 MB)"),
    _auth: None = Depends(verify_api_key),
    model=Depends(get_model),
    model_meta: dict = Depends(get_model_meta),
) -> PredictionResponse:
    """Classify a single maize leaf image as Healthy or Diseased.

    - Accepts JPEG, PNG, or WebP images up to 10 MB
    - Returns classification label, confidence score (0-1), and processing time
    - Rate limited to 20 requests per minute per IP address
    - EXIF metadata (including GPS) is stripped before processing
    """
    sanitised_bytes = await validate_image_file(file, max_bytes=settings.max_file_size_bytes)
    result = predict_image(model, sanitised_bytes)
    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        processing_time_ms=result["processing_time_ms"],
        model_version=model_meta.get("version", "unknown"),
    )
