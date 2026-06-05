import os
import tempfile
import numpy as np
import tensorflow as tf
import pytest
from model.export import export_tflite


def _tiny_model() -> tf.keras.Model:
    """Minimal model for test speed — not a real classifier."""
    inp = tf.keras.Input((224, 224, 3), dtype=tf.uint8, name="input")
    x = tf.keras.layers.Lambda(
        lambda v: tf.cast(v, tf.float32) / 127.5 - 1.0, name="preprocess"
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inp, out)


@pytest.fixture
def tiny_model():
    return _tiny_model()


@pytest.fixture
def representative_images():
    return np.random.randint(0, 256, (50, 224, 224, 3), dtype=np.uint8)


def test_tflite_output_shape(tiny_model, representative_images):
    """FIX-3: TFLite model output must be shape [1, 1] (binary probability)."""
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
        path = f.name
    try:
        export_tflite(tiny_model, path, representative_images, quantize=False)
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        out_details = interp.get_output_details()
        assert out_details[0]["shape"].tolist() == [1, 1], (
            f"Expected output shape [1, 1], got {out_details[0]['shape'].tolist()}"
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_tflite_file_created(tiny_model, representative_images):
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
        path = f.name
    try:
        export_tflite(tiny_model, path, representative_images, quantize=False)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000  # at least 1KB
    finally:
        if os.path.exists(path):
            os.unlink(path)
