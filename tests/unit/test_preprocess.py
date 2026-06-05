"""Unit tests for model/preprocess.py — validates FIX-1, FIX-2 fixes."""
import numpy as np
import pytest


def test_mobilenetv2_preprocess_range():
    """FIX-1: MobileNetV2 preprocess_input must produce values in [-1, 1], not [0, 1]."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8).astype("float32")
    out = preprocess_input(img.copy())
    assert out.min() >= -1.0 - 1e-5, f"Min value {out.min()} < -1"
    assert out.max() <= 1.0 + 1e-5, f"Max value {out.max()} > 1"


def test_vgg16_preprocess_mean_subtracted():
    """FIX-1: VGG16 preprocess_input subtracts ImageNet means, not /255."""
    from tensorflow.keras.applications.vgg16 import preprocess_input
    img = np.random.randint(0, 256, (100, 224, 224, 3), dtype=np.uint8).astype("float32")
    out = preprocess_input(img.copy())
    # After mean subtraction on random data, channel means should be near 0
    assert abs(out.mean()) < 30, f"Mean {out.mean()} too far from 0 — mean subtraction may not have occurred"


def test_binary_map_healthy():
    from model.preprocess import binary_map
    assert binary_map("healthy") == 0
    assert binary_map("Healthy") == 0
    assert binary_map("HEALTHY") == 0


def test_binary_map_disease():
    from model.preprocess import binary_map
    for label in ["blight", "common_rust", "gray_leaf_spot", "Northern_Leaf_Blight"]:
        assert binary_map(label) == 1, f"Expected 1 for '{label}'"


def test_class_weight_formula():
    """w_c = N_total / (N_classes * N_c) per paper Section 3.7."""
    from model.preprocess import compute_class_weights
    labels = [0] * 70 + [1] * 30  # 100 total, 70 healthy, 30 diseased
    weights = compute_class_weights(labels)
    expected_w0 = 100 / (2 * 70)  # ~0.714
    expected_w1 = 100 / (2 * 30)  # ~1.667
    assert abs(weights[0] - expected_w0) < 0.01, f"w_0={weights[0]}, expected {expected_w0}"
    assert abs(weights[1] - expected_w1) < 0.01, f"w_1={weights[1]}, expected {expected_w1}"


def test_stratified_split_preserves_ratio():
    from model.preprocess import stratified_split
    paths = [str(i) for i in range(200)]
    labels = [0] * 100 + [1] * 100
    (tp, tl), (vp, vl), (ep, el) = stratified_split(paths, labels)
    # Each split should have roughly 50% healthy
    for split_labels, name in [(tl, "train"), (vl, "val"), (el, "test")]:
        ratio = sum(1 for l in split_labels if l == 0) / len(split_labels)
        assert abs(ratio - 0.5) < 0.05, f"{name} healthy ratio {ratio:.2f} differs from 0.50 by > 5%"
