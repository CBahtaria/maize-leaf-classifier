"""Central configuration for training hyperparameters and constants."""

from dataclasses import dataclass

# InceptionV3 canonical input is 299×299. This was initially set to 224×224 (MobileNetV2).
# Change to (299, 299) before the next training run, then retrain and re-export the TFLite model.
# Both the API (model/predict.py) and the training pipeline use this constant, so one change
# propagates everywhere. The frontend canvas size (CameraCapture.jsx CANVAS_SIZE) must match.
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CHANNELS = 3


@dataclass(frozen=True)
class Phase1Config:
    lr: float = 1e-3
    epochs: int = 25
    patience: int = 5


@dataclass(frozen=True)
class Phase2Config:
    lr: float = 1e-5
    epochs: int = 50
    patience: int = 10
    warmup_epochs: int = 3
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7


@dataclass(frozen=True)
class HeadConfig:
    dense_units: int = 256
    dropout: float = 0.5
    bn_momentum: float = 0.99
    bn_epsilon: float = 0.001


PHASE1 = Phase1Config()
PHASE2 = Phase2Config()
HEAD = HeadConfig()

FINE_TUNE_LAYERS: dict[str, int] = {
    "mobilenetv2": 50,
    "xception": 40,
    "inceptionv3": 93,
    "vgg16": 4,
    "resnet50": 53,
}

DISEASE_LABELS: frozenset[str] = frozenset(
    {
        "blight",
        "common_rust",
        "gray_leaf_spot",
        "northern_leaf_blight",
        "leaf_blight",
        "northern_corn_leaf_blight",
        "cercospora_leaf_spot",
        "common_rust_",
    }
)

CLASS_LABEL: dict[int, str] = {0: "Healthy", 1: "Diseased"}
CLASS_INDEX: dict[str, int] = {v: k for k, v in CLASS_LABEL.items()}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"})
