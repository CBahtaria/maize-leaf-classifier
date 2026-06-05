"""Two-phase transfer learning pipeline.

Phase 1: Train classification head only (base frozen). Fast convergence.
Phase 2: Unfreeze top N layers of base, fine-tune with reduced LR + linear warmup.

Research protocol: Adam optimizer, class-weighted BCE loss, EarlyStopping, ModelCheckpoint,
ReduceLROnPlateau (Phase 2 only), LinearWarmupCallback (Phase 2, FIX-4).
"""
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from model.architectures import build_model, unfreeze_top_n
from model.callbacks import CSVEpochLogger, LinearWarmupCallback
from model.config import PHASE1, PHASE2

logger = logging.getLogger(__name__)


def _make_phase1_callbacks(output_dir: Path, arch_name: str) -> list:
    checkpoint_path = output_dir / f"{arch_name}_phase1_best.keras"
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PHASE1.patience,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        CSVEpochLogger(output_dir / f"{arch_name}_training_log.csv", phase="1"),
    ]


def _make_phase2_callbacks(output_dir: Path, arch_name: str) -> list:
    checkpoint_path = output_dir / f"{arch_name}_phase2_best.keras"
    return [
        LinearWarmupCallback(PHASE2.warmup_epochs, PHASE2.lr),  # FIX-4
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PHASE2.patience,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=PHASE2.reduce_lr_factor,
            patience=PHASE2.reduce_lr_patience,
            min_lr=PHASE2.min_lr,
            verbose=1,
        ),
        CSVEpochLogger(output_dir / f"{arch_name}_training_log.csv", phase="2"),
    ]


def train(
    arch_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    class_weights: dict[int, float],
    output_dir: str | Path,
    representative_images: np.ndarray | None = None,
) -> dict:
    """Full two-phase training pipeline.

    Args:
        arch_name: Architecture name ('mobilenetv2', 'xception', etc.)
        train_ds: Training dataset (augmented, batched).
        val_ds: Validation dataset (no augmentation).
        test_ds: Test dataset (no augmentation, never seen during training).
        class_weights: {0: w_healthy, 1: w_diseased} from compute_class_weights.
        output_dir: Where to save checkpoints, logs, and exported models.
        representative_images: 100-200 sample float32 images for TFLite quantization calibration.

    Returns:
        dict with keys: arch_name, phase1_history, phase2_history, model_path, tflite_path,
                        metrics (from evaluate_classification).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 1: Feature Extraction (%s) ===", arch_name)
    model = build_model(arch_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1.lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1.epochs,
        class_weight=class_weights,
        callbacks=_make_phase1_callbacks(output_dir, arch_name),
        verbose=1,
    )

    logger.info("=== Phase 2: Selective Fine-tuning (%s) ===", arch_name)
    n_trainable = unfreeze_top_n(model, arch_name)
    logger.info("Phase 2: %d trainable variable tensors", n_trainable)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2.lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2.epochs,
        class_weight=class_weights,
        callbacks=_make_phase2_callbacks(output_dir, arch_name),
        verbose=1,
    )

    # Save final .keras model
    model_path = output_dir / f"{arch_name}_final.keras"
    model.save(str(model_path))
    logger.info("Saved model: %s", model_path)

    # Export TFLite (FIX-3)
    from model.export import export_tflite
    tflite_path = output_dir / f"{arch_name}_int8.tflite"
    tflite_size_mb = export_tflite(
        model,
        str(tflite_path),
        representative_images=representative_images,
        quantize=True,
    )
    logger.info("TFLite export: %.2f MB → %s", tflite_size_mb, tflite_path)

    # Evaluate on held-out test set
    from model.evaluate import evaluate_classification
    metrics = evaluate_classification(model, test_ds)

    # Save metadata JSON
    meta = {
        "arch_name": arch_name,
        "model_path": str(model_path),
        "tflite_path": str(tflite_path),
        "tflite_size_mb": tflite_size_mb,
        "metrics": {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()
                    if k not in ("confusion_matrix",)},
        "version": "1.0.0",
    }
    meta_path = output_dir / f"{arch_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "arch_name": arch_name,
        "phase1_history": history1.history,
        "phase2_history": history2.history,
        "model_path": str(model_path),
        "tflite_path": str(tflite_path),
        "metrics": metrics,
    }
