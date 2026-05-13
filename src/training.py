"""Face detector fine-tuning via YOLOv8.

This module encapsulates the training logic so that it can be invoked from
a Colab notebook, a CLI script, or any other entry point.  The ultralytics
dependency is imported lazily inside :func:`train_face_detector`, following
the same pattern used in :mod:`src.detector` to keep module-level imports
free of heavy third-party libraries.

Typical usage::

    from pathlib import Path
    from src.training import TrainingConfig, train_face_detector

    config = TrainingConfig(
        dataset_yaml_path=Path("data/face-detection-dataset/data.yaml"),
        output_dir=Path("runs/face"),
    )
    result = train_face_detector(config)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_ALLOWED_BASE_MODELS = {"yolov8n.pt", "yolov8s.pt"}


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration for a face-detector training run.

    Attributes:
        dataset_yaml_path: Path to the YOLOv8-format ``data.yaml`` that
            describes the training/validation split and class names.
        output_dir: Directory where ultralytics will write training
            artefacts (checkpoints, logs, plots).
        base_model: Pre-trained YOLO checkpoint to start from.  Must be
            one of ``"yolov8n.pt"`` or ``"yolov8s.pt"``.
        epochs: Number of training epochs.  Must be > 0.
        image_size: Input image size in pixels (square).
        batch_size: Mini-batch size.  Must be > 0.
        device: Device string forwarded to ultralytics (``"cuda"``,
            ``"cpu"``, ``"0"``, etc.).

    Raises:
        ValueError: If any parameter violates its constraints.
        FileNotFoundError: If *dataset_yaml_path* does not exist.
    """

    dataset_yaml_path: Path
    output_dir: Path
    base_model: str = "yolov8n.pt"
    epochs: int = 50
    image_size: int = 640
    batch_size: int = 16
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validate configuration values after dataclass initialisation."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.base_model not in _ALLOWED_BASE_MODELS:
            raise ValueError(
                f"base_model must be one of {_ALLOWED_BASE_MODELS}, got {self.base_model!r}"
            )
        if not self.dataset_yaml_path.exists():
            raise FileNotFoundError(f"dataset_yaml_path does not exist: {self.dataset_yaml_path}")


@dataclass(frozen=True)
class TrainingResult:
    """Immutable summary of a completed training run.

    Attributes:
        best_weights_path: Path to the ``best.pt`` checkpoint selected by
            ultralytics based on validation mAP.
        final_metrics: Dictionary of key validation metrics.  Expected keys
            are ``mAP50``, ``mAP50-95``, ``precision``, and ``recall``.
        elapsed_seconds: Wall-clock training time in seconds.
    """

    best_weights_path: Path
    final_metrics: dict[str, float]
    elapsed_seconds: float


def train_face_detector(config: TrainingConfig) -> TrainingResult:
    """Fine-tune a YOLOv8 model on a face-detection dataset.

    The function loads the base model, runs training with the parameters
    specified in *config*, and returns a summary of the results including
    the path to the best checkpoint.

    Args:
        config: A :class:`TrainingConfig` instance with all training
            parameters.

    Returns:
        A :class:`TrainingResult` with the best weights path, final
        validation metrics, and elapsed wall-clock time.
    """
    from ultralytics import YOLO  # noqa: WPS433 — lazy import by design

    logger.info(
        "Starting face-detector training: base_model=%s, epochs=%d, "
        "image_size=%d, batch_size=%d, device=%s",
        config.base_model,
        config.epochs,
        config.image_size,
        config.batch_size,
        config.device,
    )

    model = YOLO(config.base_model)
    start = time.monotonic()

    model.train(
        data=str(config.dataset_yaml_path),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        device=config.device,
        project=str(config.output_dir),
        name="train",
        exist_ok=True,
    )

    elapsed = time.monotonic() - start

    best_weights = config.output_dir / "train" / "weights" / "best.pt"
    logger.info("Training complete in %.1fs — best weights at %s", elapsed, best_weights)

    metrics = model.trainer.metrics
    final_metrics = {
        "mAP50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(metrics.get("metrics/precision(B)", 0.0)),
        "recall": float(metrics.get("metrics/recall(B)", 0.0)),
    }

    logger.info("Final metrics: %s", final_metrics)

    return TrainingResult(
        best_weights_path=best_weights,
        final_metrics=final_metrics,
        elapsed_seconds=elapsed,
    )
