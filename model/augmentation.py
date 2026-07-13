"""Data augmentation pipeline using tf.keras.layers (FIX-2: replaces deprecated ImageDataGenerator).

Augmentation operations match Table 3.4 of the research paper:
- RandomFlip horizontal (simulates left/right leaf orientation)
- RandomFlip vertical (upward vs downward facing surfaces)
- RandomRotation ±20° (handheld camera angle variation)
- RandomTranslation 10% (off-centre framing)
- RandomZoom ±15% (varying camera-to-leaf distance)
- RandomBrightness ±20% (outdoor illumination variation in SSA)

Applied only during training (training=True). GPU-executed — runs in the training graph,
not on CPU like ImageDataGenerator.
"""

import tensorflow as tf


def get_augmentation_layer() -> tf.keras.Sequential:
    """Build and return the training augmentation Sequential layer.

    Usage:
        aug = get_augmentation_layer()
        augmented = aug(image_tensor, training=True)   # training=True enables stochastic ops
        passthrough = aug(image_tensor, training=False) # training=False is identity (no aug)
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(20.0 / 360.0),
            tf.keras.layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
            tf.keras.layers.RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15)),
            tf.keras.layers.RandomBrightness(factor=0.20),
        ],
        name="data_augmentation",
    )
