"""CNN model builders for all 5 architectures with embedded architecture-specific preprocessing.

FIX-1: Each architecture's preprocess_input is embedded as the first layer of the model.
       This makes the model self-contained: raw [0,255] float32 input → binary probability output.
       No external preprocessing is needed in the API or frontend.

FIX-5: Layer unfreezing uses dynamic len(base_model.layers) - fine_tune_n, not hardcoded indices.
"""
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import (
    VGG16,
    InceptionV3,
    MobileNetV2,
    ResNet50,
    Xception,
    inception_v3,
    mobilenet_v2,
    resnet50,
    vgg16,
    xception,
)

from model.config import FINE_TUNE_LAYERS, HEAD, IMG_SIZE

# Registry of architecture-specific configs
# preprocess: maps raw [0,255] float32 to architecture's expected range
#   MobileNetV2, Xception, InceptionV3: → [-1, 1] via (x/127.5) - 1
#   VGG16, ResNet50: → ImageNet mean-subtracted (different ranges per channel)
ARCH_REGISTRY: dict[str, dict] = {
    "mobilenetv2": {
        "cls": MobileNetV2,
        "preprocess": mobilenet_v2.preprocess_input,
        "fine_tune_n": FINE_TUNE_LAYERS["mobilenetv2"],
        "native_input": 224,
    },
    "xception": {
        "cls": Xception,
        "preprocess": xception.preprocess_input,
        "fine_tune_n": FINE_TUNE_LAYERS["xception"],
        "native_input": 299,  # standardised to 224×224 for experimental consistency
    },
    "inceptionv3": {
        "cls": InceptionV3,
        "preprocess": inception_v3.preprocess_input,
        "fine_tune_n": FINE_TUNE_LAYERS["inceptionv3"],
        "native_input": 299,
    },
    "vgg16": {
        "cls": VGG16,
        "preprocess": vgg16.preprocess_input,
        "fine_tune_n": FINE_TUNE_LAYERS["vgg16"],
        "native_input": 224,
    },
    "resnet50": {
        "cls": ResNet50,
        "preprocess": resnet50.preprocess_input,
        "fine_tune_n": FINE_TUNE_LAYERS["resnet50"],
        "native_input": 224,
    },
}


def _get_base_model(arch_name: str, img_size: tuple[int, int] = IMG_SIZE) -> tf.keras.Model:
    """Instantiate the base model with ImageNet weights, no top layers."""
    cfg = ARCH_REGISTRY[arch_name]
    return cfg["cls"](
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
    )


def build_model(
    arch_name: str,
    img_size: tuple[int, int] = IMG_SIZE,
    include_augmentation: bool = False,
) -> tf.keras.Model:
    """Build a complete binary classification model.

    Architecture:
        Input(H, W, 3) — raw float32 in [0, 255]
        → Lambda(preprocess_fn)           FIX-1: arch-specific normalization embedded
        → [optional augmentation layer]   FIX-2: GPU augmentation during training
        → base_model (frozen)             ImageNet weights, include_top=False
        → GlobalAveragePooling2D          replaces Flatten — structural regularizer
        → Dense(256, relu)
        → BatchNormalization
        → Dropout(0.5)
        → Dense(1, sigmoid)               P(Diseased | image)

    Args:
        arch_name: Key in ARCH_REGISTRY.
        img_size: Spatial dimensions (H, W). All architectures use 224×224 for consistency.
        include_augmentation: If True, augmentation layer is part of model graph
                              (alternative to dataset-level augmentation).

    Returns:
        Compiled-ready Keras Model with base model FROZEN (Phase 1 ready).
    """
    if arch_name not in ARCH_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch_name}'. Choose from: {list(ARCH_REGISTRY)}")

    cfg = ARCH_REGISTRY[arch_name]
    preprocess_fn = cfg["preprocess"]

    # --- Build functional model ---
    inputs = layers.Input(shape=(*img_size, 3), name="image_input")

    # FIX-1: Embed architecture-specific preprocessing as first layer
    x = layers.Lambda(
        preprocess_fn,
        name="arch_preprocess",
    )(inputs)

    # Optional: in-model augmentation (alternative to dataset augmentation)
    if include_augmentation:
        from model.augmentation import get_augmentation_layer
        aug = get_augmentation_layer()
        x = aug(x)

    # Base model (frozen)
    base = _get_base_model(arch_name, img_size)
    base.trainable = False
    x = base(x, training=False)

    # Custom classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(HEAD.dense_units, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(
        momentum=HEAD.bn_momentum,
        epsilon=HEAD.bn_epsilon,
        name="batch_norm",
    )(x)
    x = layers.Dropout(HEAD.dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"{arch_name}_binary_classifier")
    return model


def unfreeze_top_n(model: tf.keras.Model, arch_name: str) -> int:
    """Unfreeze the top fine_tune_n layers of the base model for Phase 2 fine-tuning.

    FIX-5: Uses len(base_model.layers) - fine_tune_n dynamically.
           Hardcoded indices (as in the original paper) break when TF changes layer counts.

    Args:
        model: The full Keras model (must have been built by build_model).
        arch_name: Architecture name key.

    Returns:
        Number of layers that are now trainable (for logging).
    """
    base_model = _find_base_model_layer(model, arch_name)
    fine_tune_n = ARCH_REGISTRY[arch_name]["fine_tune_n"]

    # FIX-5: dynamic unfreezing
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - fine_tune_n

    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= fine_tune_at

    trainable_count = sum(1 for _ in model.trainable_variables)
    return trainable_count


def _find_base_model_layer(model: tf.keras.Model, arch_name: str) -> tf.keras.Model:
    """Locate the base model sub-layer within the full model."""
    arch_cls = ARCH_REGISTRY[arch_name]["cls"]
    for layer in model.layers:
        if isinstance(layer, arch_cls):
            return layer
    raise RuntimeError(f"Could not find {arch_cls.__name__} sub-layer in model.")


def get_arch_names() -> list[str]:
    """Return the list of supported architecture names."""
    return list(ARCH_REGISTRY.keys())
