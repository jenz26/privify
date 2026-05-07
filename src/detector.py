"""Face and license-plate detector wrapping YOLOv8.

This module provides a stable detection API for the rest of the project.
Other modules (tracker, anonymizer) depend on :class:`Detector` and
:class:`Detection`, never on ``ultralytics`` directly.  This wrapper pattern
confines the third-party coupling to a single file, making it straightforward
to swap the backend (e.g. ONNX, a different YOLO version) without touching
downstream code.

**Provisional behaviour:** until the model is fine-tuned on WIDER FACE + CCPD,
the detector uses ``yolov8n.pt`` pre-trained on COCO.  COCO class 0 ("person")
is used as a temporary proxy for "face".  License-plate detection is not
available at this stage.  See ``docs/decisions.md`` for the rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A single object detection in a video frame.

    Attributes:
        bbox: Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
        confidence: Detection confidence score in ``[0.0, 1.0]``.
        class_id: Integer class identifier (0 = face, 1 = plate after
            fine-tuning; COCO class ids in the provisional stage).
        class_name: Human-readable class label.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class Detector:
    """YOLOv8-based object detector with lazy model loading.

    The model weights are loaded on the first call to :meth:`detect`, not
    during ``__init__``.  This keeps instantiation free of I/O and GPU
    allocation, which simplifies testing and allows early configuration.

    Args:
        model_path: Path to a YOLO weights file (e.g. ``yolov8n.pt``).
        conf_threshold: Minimum confidence for a detection to be returned.
            Must be in ``(0.0, 1.0]``.
        device: Device string forwarded to ultralytics (``"cpu"``,
            ``"cuda:0"``, etc.).  ``None`` lets ultralytics auto-select.

    Raises:
        ValueError: If *conf_threshold* is not in ``(0.0, 1.0]``.
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        if not 0.0 < conf_threshold <= 1.0:
            raise ValueError(f"conf_threshold must be in (0.0, 1.0], got {conf_threshold}")

        self._model_path = Path(model_path)
        self._conf_threshold = conf_threshold
        self._device = device
        self._model: YOLO | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> YOLO:
        """Load the YOLO model from disk.

        Returns:
            The loaded ``ultralytics.YOLO`` model instance.
        """
        from ultralytics import YOLO  # noqa: WPS433 — lazy import by design

        logger.info("Loading YOLO model from %s", self._model_path)
        model = YOLO(str(self._model_path))
        if self._device is not None:
            model.to(self._device)
        return model

    def _ensure_model(self) -> YOLO:
        """Return the model, loading it on first access.

        Returns:
            The cached ``ultralytics.YOLO`` model instance.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a single BGR frame.

        Args:
            frame: A ``(H, W, 3)`` NumPy array in BGR colour order, as
                returned by ``cv2.imread`` or ``cv2.VideoCapture.read``.

        Returns:
            A list of :class:`Detection` instances, one per detected object
            whose confidence exceeds the configured threshold.

        Raises:
            ValueError: If *frame* is not a 3-channel image array.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected a 3-channel image (H, W, 3), got shape {frame.shape}")

        model = self._ensure_model()
        results = model.predict(
            source=frame,
            conf=self._conf_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = result.names[cls_id]
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        logger.debug("Detected %d objects in frame", len(detections))
        return detections
