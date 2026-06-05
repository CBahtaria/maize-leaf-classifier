"""Convert the best Keras model to TF.js GraphModel for browser offline inference."""
import shutil
from pathlib import Path

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "model_artifacts" / "mobilenetv2_best.keras"
TFJS_OUT   = ROOT / "model_artifacts" / "tfjs_model"
DEST       = ROOT / "frontend" / "public" / "models" / "tfjs"

import tensorflow as tf  # noqa: E402

model = tf.keras.models.load_model(MODEL_PATH)

import tensorflowjs as tfjs  # noqa: E402

tfjs.converters.save_keras_model(model, str(TFJS_OUT))
print(f"TF.js model saved to {TFJS_OUT}")

DEST.mkdir(parents=True, exist_ok=True)
for f in TFJS_OUT.iterdir():
    shutil.copy(f, DEST / f.name)
print(f"Copied to {DEST}")
