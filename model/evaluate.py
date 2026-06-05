"""Classification performance metrics and computational efficiency benchmarks.

Implements all 6 classification metrics from the research paper Section 3.12.1:
accuracy, precision, sensitivity (recall), specificity, F1, AUC-ROC.

FIX-7: Benchmarks BOTH .keras/.h5 CPU inference time (paper-compatible academic comparison)
       AND TFLite CPU inference time (deployment-realistic benchmark for mobile devices).
"""
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_classification(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict:
    """Compute all classification metrics on a test dataset.

    Returns dict with: accuracy, precision, sensitivity, specificity, f1, auc_roc,
                       confusion_matrix (2x2 array), wilson_ci_95 (lower, upper).
    """
    y_true, y_scores = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_scores.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores, dtype=float)
    y_pred = (y_scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # TPR / Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    ci_lower, ci_upper = wilson_ci(acc, n)

    results = {
        "accuracy": float(acc),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "auc_roc": float(auc),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_test": int(n),
        "wilson_ci_95": (float(ci_lower), float(ci_upper)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    logger.info("Evaluation: acc=%.4f  sens=%.4f  spec=%.4f  AUC=%.4f  F1=%.4f",
                acc, sensitivity, specificity, auc, f1)
    return results


def wilson_ci(accuracy: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    From paper Section 3.12.3. Appropriate for proportions from finite test sets.

    Args:
        accuracy: Observed proportion (test accuracy).
        n: Total number of test samples.
        z: z-score for confidence level (1.96 for 95%).

    Returns:
        (lower_bound, upper_bound) of the confidence interval.
    """
    p = accuracy
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    margin = (z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def measure_inference_time_keras(
    model_path: str,
    test_images: np.ndarray,
    n_runs: int = 50,
) -> float:
    """Measure mean CPU inference time (ms) for a .keras or .h5 model.

    Paper-compatible benchmark (academic comparison). Runs on CPU only.
    Returns mean time per single image in milliseconds.
    """
    # Force CPU (simulate mobile device without NPU)
    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(model_path)
        # Warm-up pass
        _ = model.predict(test_images[:1], verbose=0)
        # Timed runs
        times = []
        for img in test_images[:n_runs]:
            start = time.perf_counter()
            _ = model.predict(img[np.newaxis], verbose=0)
            times.append((time.perf_counter() - start) * 1000)
    mean_ms = float(np.mean(times))
    logger.info("Keras CPU inference: %.1f ms/image (n=%d)", mean_ms, n_runs)
    return mean_ms


def measure_inference_time_tflite(
    tflite_path: str,
    test_images: np.ndarray,
    n_runs: int = 50,
) -> float:
    """Measure mean CPU inference time (ms) for a TFLite model.

    FIX-7: Deployment-realistic benchmark. TFLite is the actual format used on Android/iOS.
    The .keras/.h5 benchmark above is for academic comparison only.
    Returns mean time per single image in milliseconds.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine input dtype (uint8 for INT8 quantized, float32 for float)
    input_dtype = input_details[0]["dtype"]

    times = []
    for img in test_images[:n_runs]:
        if input_dtype == np.uint8:
            inp = img.astype(np.uint8)[np.newaxis]
        else:
            inp = img.astype(np.float32)[np.newaxis]
        interpreter.set_tensor(input_details[0]["index"], inp)
        start = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - start) * 1000)
        _ = interpreter.get_tensor(output_details[0]["index"])

    mean_ms = float(np.mean(times))
    logger.info("TFLite CPU inference: %.1f ms/image (n=%d)", mean_ms, n_runs)
    return mean_ms


def generate_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    history: dict,
    output_dir: str | Path,
    arch_name: str = "model",
) -> None:
    """Save confusion matrix, ROC curve, and training history plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                xticklabels=["Healthy", "Diseased"],
                yticklabels=["Healthy", "Diseased"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {arch_name}")
    fig.savefig(output_dir / f"{arch_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="#2d6a4f")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curve — {arch_name}")
    ax.legend()
    fig.savefig(output_dir / f"{arch_name}_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for phase, hist, color in [("Phase 1", history.get("phase1", {}), "#2d6a4f"),
                                 ("Phase 2", history.get("phase2", {}), "#9b2226")]:
        if not hist:
            continue
        epochs = range(1, len(hist.get("loss", [])) + 1)
        axes[0].plot(epochs, hist.get("loss", []), label=f"{phase} train", color=color)
        axes[0].plot(epochs, hist.get("val_loss", []), "--", label=f"{phase} val", color=color, alpha=0.7)
        axes[1].plot(epochs, hist.get("accuracy", []), label=f"{phase} train", color=color)
        axes[1].plot(epochs, hist.get("val_accuracy", []), "--", label=f"{phase} val", color=color, alpha=0.7)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    fig.suptitle(f"Training History — {arch_name}")
    fig.savefig(output_dir / f"{arch_name}_training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_eval_markdown(results_by_arch: dict[str, dict], output_path: str | Path) -> None:
    """Write a Markdown table of evaluation results for all architectures."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Model Evaluation Results\n",
        "| Architecture | Accuracy | Precision | Sensitivity | Specificity | F1 | AUC-ROC | Size (MB) | Inference (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arch, r in results_by_arch.items():
        m = r.get("metrics", {})
        ci = m.get("wilson_ci_95", (0, 0))
        lines.append(
            f"| {arch} "
            f"| {m.get('accuracy', 0):.4f} [{ci[0]:.3f},{ci[1]:.3f}] "
            f"| {m.get('precision', 0):.4f} "
            f"| {m.get('sensitivity', 0):.4f} "
            f"| {m.get('specificity', 0):.4f} "
            f"| {m.get('f1', 0):.4f} "
            f"| {m.get('auc_roc', 0):.4f} "
            f"| {r.get('tflite_size_mb', 'N/A')} "
            f"| {r.get('tflite_inference_ms', 'N/A')} |"
        )
    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Evaluation table written to %s", output_path)
