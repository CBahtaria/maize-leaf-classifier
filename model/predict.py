"""Single-image inference module used by the FastAPI backend.

Supports both .keras and .tflite models. Preprocessing is embedded in the model (FIX-1),
so this module passes raw image bytes with NO manual normalization.
"""
import io
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from model.config import CLASS_LABEL, IMG_SIZE

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Unified inference interface for .keras and .tflite models."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model: Any = None
        self._interpreter: Any = None
        self._is_tflite = self.model_path.suffix == ".tflite"
        self._load()

    def _load(self) -> None:
        if self._is_tflite:
            self._interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_dtype = self._input_details[0]["dtype"]
            logger.info("Loaded TFLite model: %s", self.model_path)
        else:
            self._model = tf.keras.models.load_model(str(self.model_path))
            logger.info("Loaded Keras model: %s", self.model_path)

    def predict_raw(self, image_array: np.ndarray) -> float:
        """Run inference on a preprocessed image array (H, W, 3) float32 in [0,255].

        Returns sigmoid probability P(Diseased | image) in [0.0, 1.0].
        """
        if self._is_tflite:
            inp = image_array[np.newaxis]
            if self._input_dtype == np.uint8:
                inp = inp.astype(np.uint8)
            else:
                inp = inp.astype(np.float32)
            self._interpreter.set_tensor(self._input_details[0]["index"], inp)
            self._interpreter.invoke()
            return float(self._interpreter.get_tensor(self._output_details[0]["index"])[0][0])
        else:
            inp = image_array[np.newaxis].astype(np.float32)
            return float(self._model.predict(inp, verbose=0)[0][0])


def load_model_for_inference(model_path: str | Path) -> ModelWrapper:
    """Load a .keras or .tflite model and return a ModelWrapper."""
    return ModelWrapper(model_path)


def predict_image(model: ModelWrapper, image_bytes: bytes) -> dict:
    """Classify a single maize leaf image.

    FIX-1: No external preprocessing applied here. The model's embedded Lambda layer
           handles architecture-specific normalization internally.

    Args:
        model: ModelWrapper instance (from load_model_for_inference).
        image_bytes: Raw JPEG/PNG/WebP image bytes.

    Returns:
        dict with: label ("Healthy" | "Diseased"), confidence (float), processing_time_ms (float)
    """
    start = time.perf_counter()

    # Load and resize image to 224×224 — no normalization (model handles it)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)  # [0, 255] — preprocess_fn in model handles rest

    confidence = model.predict_raw(img_array)
    label = CLASS_LABEL[int(confidence >= 0.5)]
    processing_ms = (time.perf_counter() - start) * 1000

    return {
        "label": label,
        "confidence": float(confidence),
        "processing_time_ms": round(processing_ms, 2),
    }
