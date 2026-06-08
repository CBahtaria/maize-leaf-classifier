import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from model.augmentation import get_augmentation_layer  # noqa: E402


def test_augmentation_preserves_shape():
    layer = get_augmentation_layer()
    x = tf.random.uniform((4, 224, 224, 3), minval=0, maxval=255)
    out = layer(x, training=True)
    assert out.shape == (4, 224, 224, 3), f"Shape mismatch: {out.shape}"


def test_augmentation_disabled_at_inference():
    """At inference time (training=False), augmentation layers must be pass-through."""
    layer = get_augmentation_layer()
    x = tf.constant(np.ones((1, 224, 224, 3), dtype=np.float32) * 128.0)
    out = layer(x, training=False)
    np.testing.assert_array_almost_equal(
        x.numpy(), out.numpy(),
        err_msg="Augmentation applied during inference — should be pass-through when training=False",
    )


def test_augmentation_returns_float():
    layer = get_augmentation_layer()
    x = tf.random.uniform((1, 224, 224, 3))
    out = layer(x, training=True)
    assert out.dtype in (tf.float32, tf.float16), f"Unexpected dtype: {out.dtype}"
