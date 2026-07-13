"""Model export utilities: TFLite INT8 quantization and TF.js conversion.

FIX-3: The research paper claims mobile deployment but never specifies TFLite conversion.
       This module implements the export pipeline that makes mobile deployment actually possible:
       - INT8 quantized TFLite: ~3.5 MB (vs ~14 MB .keras), 2-4x faster inference
       - TF.js: for browser-based offline inference in the PWA
"""

import logging
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def export_tflite(
    model: tf.keras.Model,
    output_path: str | Path,
    representative_images: np.ndarray | None = None,
    quantize: bool = True,
) -> float:
    """Convert a Keras model to TFLite with optional INT8 post-training quantization.

    FIX-3: The paper measured CPU inference on a full .h5 model but claimed mobile deployment.
           This function produces the actual deployable mobile format.

    The model's preprocessing Lambda layer (FIX-1) is preserved in the TFLite graph,
    so the TFLite model also accepts raw [0,255] uint8 input — no external preprocessing needed.

    Args:
        model: Trained Keras model (with embedded preprocess Lambda layer).
        output_path: Path to save the .tflite file.
        representative_images: 100-200 sample images (float32 [0,255]) for INT8 calibration.
                               If None, uses dynamic range quantization (larger model).
        quantize: If False, export float32 TFLite (larger, for debugging).

    Returns:
        File size in MB.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_images is not None:

            def rep_dataset():
                for img in representative_images[:200]:
                    yield [img[np.newaxis].astype(np.float32)]

            converter.representative_dataset = rep_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32
            logger.info(
                "TFLite: using full INT8 quantization with %d calibration images",
                len(representative_images[:200]),
            )
        else:
            logger.info("TFLite: using dynamic range quantization (no calibration images provided)")

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    size_mb = output_path.stat().st_size / 1e6
    logger.info("TFLite exported: %.2f MB → %s", size_mb, output_path)
    return size_mb


def export_tfjs(
    model: tf.keras.Model,
    output_dir: str | Path,
    frontend_dest: str | Path | None = None,
) -> None:
    """Convert a Keras model to TF.js GraphModel format for browser offline inference.

    The exported model retains the embedded preprocess Lambda (FIX-1), so the browser
    code passes raw pixel values — no manual /255 division needed in tfjs-inference.js.

    Args:
        model: Trained Keras model.
        output_dir: Directory to save TF.js model files (model.json + weight shards).
        frontend_dest: If provided, copy output to this frontend public directory.
                       E.g., 'frontend/public/models/tfjs'
    """
    try:
        import tensorflowjs as tfjs
    except ImportError as e:
        raise ImportError(
            "tensorflowjs not installed. Run: pip install tensorflowjs==4.20.0"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tfjs.converters.save_keras_model(model, str(output_dir))
    logger.info("TF.js model saved to %s", output_dir)

    if frontend_dest is not None:
        frontend_dest = Path(frontend_dest)
        if frontend_dest.exists():
            shutil.rmtree(frontend_dest)
        shutil.copytree(output_dir, frontend_dest)
        logger.info("TF.js model copied to %s", frontend_dest)
