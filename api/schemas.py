"""Pydantic request and response schemas for the prediction API."""
from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    label: Literal["Healthy", "Diseased"]
    confidence: float = Field(ge=0.0, le=1.0, description="P(Diseased | image)")
    processing_time_ms: float = Field(ge=0.0)
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_images: int
    total_time_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    architecture: str
    version: str
    tflite_size_mb: float | None = None
    accuracy: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    auc_roc: float | None = None
    model_path: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
