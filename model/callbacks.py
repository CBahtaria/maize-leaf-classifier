"""Custom Keras callbacks for the two-phase transfer learning protocol.

FIX-4: LinearWarmupCallback implements the learning rate warmup described in
        the paper (Section 3.10.2). Keras has no built-in linear LR warmup;
        without this callback the warmup described in the paper would never execute.
"""
import csv
import logging
from pathlib import Path
from typing import IO

import tensorflow as tf

logger = logging.getLogger(__name__)


class LinearWarmupCallback(tf.keras.callbacks.Callback):
    """Linearly ramp learning rate from base_lr/10 to base_lr over warmup_epochs.

    Implements the warmup schedule from research paper Section 3.10.2:
        "Learning rate linearly ramps from α/10 to α over first 3 epochs."

    FIX-4: This callback was missing from the original paper's methodology.
           Without it, Phase 2 training starts at the full fine-tuning LR immediately,
           which risks large gradient updates that corrupt pretrained representations.

    Args:
        warmup_epochs: Number of epochs over which to ramp up the LR.
        base_lr: Target learning rate (reached at the end of warmup).
    """

    def __init__(self, warmup_epochs: int, base_lr: float) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.start_lr = base_lr / 10.0

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        if epoch < self.warmup_epochs:
            # Linear interpolation from start_lr to base_lr
            progress = epoch / self.warmup_epochs
            lr = self.start_lr + progress * (self.base_lr - self.start_lr)
            self.model.optimizer.learning_rate.assign(lr)
            logger.debug("Warmup epoch %d: LR set to %.2e", epoch, lr)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if epoch == self.warmup_epochs - 1:
            # Ensure LR is exactly base_lr after warmup completes
            self.model.optimizer.learning_rate.assign(self.base_lr)
            logger.info("LR warmup complete. LR = %.2e", self.base_lr)


class CSVEpochLogger(tf.keras.callbacks.Callback):
    """Appends per-epoch metrics to a CSV file for experiment tracking.

    Columns: epoch, phase, loss, val_loss, accuracy, val_accuracy, lr
    """

    def __init__(self, filepath: str | Path, phase: str = "1") -> None:
        super().__init__()
        self.filepath = Path(filepath)
        self.phase = phase
        self._writer: csv.DictWriter | None = None
        self._file: IO[str] | None = None

    def on_train_begin(self, logs: dict | None = None) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.filepath.exists()
        self._file = open(self.filepath, "a", newline="")  # noqa: SIM115
        fieldnames = ["epoch", "phase", "loss", "val_loss", "accuracy", "val_accuracy", "lr"]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if write_header:
            self._writer.writeheader()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            logs = {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        row = {
            "epoch": epoch,
            "phase": self.phase,
            "loss": logs.get("loss", ""),
            "val_loss": logs.get("val_loss", ""),
            "accuracy": logs.get("accuracy", ""),
            "val_accuracy": logs.get("val_accuracy", ""),
            "lr": lr,
        }
        if self._writer and self._file:
            self._writer.writerow(row)
            self._file.flush()

    def on_train_end(self, logs: dict | None = None) -> None:
        if self._file:
            self._file.close()
