"""Weekly regression accuracy gate — asserts model accuracy >= 99.0% on held-out images.

Expected directory layout (populated by CI download step):
  tests/regression/data/healthy/*.jpg
  tests/regression/data/diseased/*.jpg

Runs against the live model artifact at model_artifacts/model.tflite.
If no held-out data is present, the test is skipped (not failed) so the
workflow remains green on branches without access to secrets.
"""

import glob
import io
import os

import numpy as np
import pytest
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "99.0"))


def _load_images(pattern: str) -> list[bytes]:
    paths = glob.glob(pattern)
    images = []
    for p in paths:
        with open(p, "rb") as f:
            images.append(f.read())
    return images


def _healthy_images() -> list[bytes]:
    return _load_images(os.path.join(DATA_DIR, "healthy", "*.jpg"))


def _diseased_images() -> list[bytes]:
    return _load_images(os.path.join(DATA_DIR, "diseased", "*.jpg"))


@pytest.fixture(scope="module")
def model_predict():
    """Return a callable (image_bytes) -> str label using the live TFLite model."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "model_artifacts", "model.tflite"
    )
    if not os.path.exists(model_path):
        pytest.skip("No model artifact found at model_artifacts/model.tflite")

    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow as tf
            tflite = tf.lite
        except ImportError:
            pytest.skip("Neither tflite_runtime nor tensorflow is installed")

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    def predict(image_bytes: bytes) -> str:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        interpreter.set_tensor(inp["index"], arr)
        interpreter.invoke()
        confidence = float(interpreter.get_tensor(out["index"])[0][0])
        return "Diseased" if confidence >= 0.5 else "Healthy"

    return predict


def test_regression_accuracy_gate(model_predict):
    healthy = _healthy_images()
    diseased = _diseased_images()
    total = len(healthy) + len(diseased)

    if total == 0:
        pytest.skip("No held-out images in tests/regression/data/ — skipping accuracy gate")

    correct = 0
    for img in healthy:
        if model_predict(img) == "Healthy":
            correct += 1
    for img in diseased:
        if model_predict(img) == "Diseased":
            correct += 1

    accuracy = (correct / total) * 100.0
    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Accuracy {accuracy:.2f}% is below threshold {ACCURACY_THRESHOLD}% "
        f"({correct}/{total} correct)"
    )
