"""Dataset loading, binary label mapping, splitting, and tf.data pipeline construction.

IMPORTANT: FIX-1 — preprocess_fn is architecture-specific. This module accepts it as a
parameter rather than hardcoding /255 normalization (which would be wrong for all 5 architectures).
FIX-2 — Uses tf.data.Dataset instead of deprecated ImageDataGenerator.
"""
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

from model.config import BATCH_SIZE, CHANNELS, IMG_SIZE, SEED, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


def binary_map(label: str) -> int:
    """Map a directory/class label to binary {0=Healthy, 1=Diseased}.

    Case-insensitive. Any label not matching 'healthy' → 1 (Diseased).
    This is intentionally permissive: unknown disease labels default to Diseased.
    """
    return 0 if label.strip().lower() == "healthy" else 1


def scan_dataset(data_dir: str | Path) -> tuple[list[str], list[int]]:
    """Walk data_dir, collect image paths and binary labels.

    Expected directory structure:
        data_dir/
          Healthy/     → label 0
          Blight/      → label 1
          Common_Rust/ → label 1
          ...

    Returns (image_paths, binary_labels) — parallel lists.
    """
    data_dir = Path(data_dir)
    paths, labels = [], []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = binary_map(class_dir.name)
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix in SUPPORTED_EXTENSIONS:
                paths.append(str(img_path))
                labels.append(label)
    logger.info("Scanned %d images: %d healthy, %d diseased",
                len(paths), labels.count(0), labels.count(1))
    return paths, labels


def stratified_split(
    paths: list[str],
    labels: list[int],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = SEED,
) -> tuple[
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
]:
    """Stratified 70/15/15 split preserving class ratio in each partition.

    Returns: ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels))
    """
    paths_arr = np.array(paths)
    labels_arr = np.array(labels)

    # First split: carve out test set
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(splitter.split(paths_arr, labels_arr))

    trainval_paths = paths_arr[trainval_idx]
    trainval_labels = labels_arr[trainval_idx]
    test_paths = paths_arr[test_idx].tolist()
    test_labels = labels_arr[test_idx].tolist()

    # Second split: carve val from train+val
    val_size = val_frac / (1.0 - test_frac)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter2.split(trainval_paths, trainval_labels))

    train_paths = trainval_paths[train_idx].tolist()
    train_labels = trainval_labels[train_idx].tolist()
    val_paths = trainval_paths[val_idx].tolist()
    val_labels = trainval_labels[val_idx].tolist()

    logger.info("Split: train=%d  val=%d  test=%d", len(train_paths), len(val_paths), len(test_paths))
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    """Compute inverse-frequency class weights: w_c = N_total / (N_classes * N_c).

    Matches the formula in the research paper Section 3.7.
    Upweights minority class to counteract dataset imbalance.
    """
    n_total = len(labels)
    n_classes = 2
    counts = {0: labels.count(0), 1: labels.count(1)}
    weights = {c: n_total / (n_classes * counts[c]) for c in counts}
    logger.info("Class weights: healthy=%.3f, diseased=%.3f", weights[0], weights[1])
    return weights


def _load_image(path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    """Load, decode, and resize a single image. Returns float32 tensor in [0,255].

    NOTE: We return [0,255] uint8-cast-to-float here. The architecture-specific
    preprocess_fn (FIX-1) is applied as a model-embedded Lambda layer, NOT here.
    This keeps the preprocessing self-contained within the model artefact.
    """
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=CHANNELS)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)  # Still in [0, 255] range — preprocess_fn handles normalization
    return img, label


def build_tf_dataset(
    paths: list[str],
    labels: list[int],
    preprocess_fn: Callable[..., Any],
    augment: bool,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset pipeline.

    FIX-1: preprocess_fn is architecture-specific (e.g., mobilenet_v2.preprocess_input).
           It is applied per-image BEFORE batching.
    FIX-2: Uses tf.data.Dataset + tf.keras.layers augmentation, NOT ImageDataGenerator.

    Args:
        paths: Image file paths.
        labels: Corresponding binary labels (0=Healthy, 1=Diseased).
        preprocess_fn: Architecture-specific preprocessing function.
        augment: If True, apply augmentation layer (training only).
        batch_size: Mini-batch size.
        img_size: Target (H, W) spatial dimensions.

    Returns:
        Batched, prefetched tf.data.Dataset yielding (image_tensor, label_tensor).
    """
    from model.augmentation import get_augmentation_layer

    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    path_ds = path_ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True) if augment else path_ds

    def load_fn(p, lbl):
        return _load_image(p, lbl, img_size)

    ds = path_ds.map(load_fn, num_parallel_calls=AUTOTUNE)

    # Apply architecture-specific preprocessing (FIX-1)
    ds = ds.map(lambda img, lbl: (preprocess_fn(img), lbl), num_parallel_calls=AUTOTUNE)

    if augment:
        aug_layer = get_augmentation_layer()
        ds = ds.map(
            lambda img, lbl: (aug_layer(img, training=True), lbl),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
