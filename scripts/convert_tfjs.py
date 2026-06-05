"""Convert the best Keras model to TF.js GraphModel for browser offline inference."""
import shutil
from pathlib import Path

MODEL_PATH = "model_artifacts/mobilenetv2_best.keras"
TFJS_OUT = "model_artifacts/tfjs_model"
DEST = "frontend/public/models/tfjs"

import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, TFJS_OUT)
print(f"TF.js model saved to {TFJS_OUT}")

dest = Path(DEST)
dest.mkdir(parents=True, exist_ok=True)
for f in Path(TFJS_OUT).iterdir():
    shutil.copy(f, dest / f.name)
print(f"Copied to {DEST}")
