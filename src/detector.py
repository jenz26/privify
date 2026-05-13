"""Face and license-plate detector wrapping YOLOv8.

This module provides a stable detection API for the rest of the project.
Other modules (tracker, anonymizer) depend on :class:`Detector` and
:class:`Detection`, never on ``ultralytics`` directly.  This wrapper pattern
confines the third-party coupling to a single file, making it straightforward
to swap the backend (e.g. ONNX, a different YOLO version) without touching
downstream code.

By default the detector uses ``face-detector-v0.1.pt``, a YOLOv8n model
fine-tuned for single-class face detection.  The weights are downloaded
automatically from the GitHub Release on first use and cached locally in
``models/``.  Pass a custom *model_path* to override (e.g. for testing or
to use a different checkpoint).
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

# -- Fine-tuned weights distributed via GitHub Releases --------------------
WEIGHTS_URL = "https://github.com/jenz26/privify/releases/download/face-detector-v0.1/best.pt"
WEIGHTS_SHA256 = "af443fa561da808b007c62e526d07d947c58e21518b7e10784c704fd9b822d30"
WEIGHTS_FILENAME = "face-detector-v0.1.pt"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS_PATH = _PROJECT_ROOT / "models" / WEIGHTS_FILENAME


def _verify_sha256(filepath: Path, expected: str) -> bool:
    """Compute SHA-256 of a file and compare against an expected digest.

    Args:
        filepath: Path to the file to hash.
        expected: Lowercase hex-encoded SHA-256 digest to compare against.

    Returns:
        ``True`` if the computed digest matches *expected*, ``False``
        otherwise.
    """
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def _ensure_weights_available(weights_path: Path) -> None:
    """Download fine-tuned weights if not already cached.

    If the file exists and its SHA-256 matches, this is a no-op.  If the
    hash does not match (corrupted or tampered file), the file is deleted
    and re-downloaded.

    Args:
        weights_path: Local path where the weights should reside.

    Raises:
        RuntimeError: If the freshly downloaded file fails the integrity
            check — indicates a stale hash constant or an upstream issue.
    """
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if weights_path.exists():
        if _verify_sha256(weights_path, WEIGHTS_SHA256):
            return
        logger.warning("Weights at %s failed integrity check — re-downloading", weights_path)
        weights_path.unlink()

    logger.info("Downloading fine-tuned weights from %s", WEIGHTS_URL)
    urllib.request.urlretrieve(WEIGHTS_URL, weights_path)

    if not _verify_sha256(weights_path, WEIGHTS_SHA256):
        weights_path.unlink()
        raise RuntimeError(
            f"Downloaded weights at {weights_path} failed SHA-256 integrity "
            f"check. Expected {WEIGHTS_SHA256}, but file hash differs. "
            f"The Release asset may have been modified, or the constant in "
            f"detector.py is stale."
        )


@dataclass(frozen=True)
class Detection:
    """A single object detection in a video frame.

    Attributes:
        bbox: Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
        confidence: Detection confidence score in ``[0.0, 1.0]``.
        class_id: Integer class identifier (0 = face with the default
            fine-tuned model; COCO class ids if using a generic model).
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

    When *model_path* is left at its default, the fine-tuned face-detector
    weights are downloaded automatically from the GitHub Release on first
    use and cached in ``models/``.  Pass a custom path to use different
    weights (useful for testing or for swapping to a future version).

    Args:
        model_path: Path to a YOLO weights file.  Defaults to the
            fine-tuned ``face-detector-v0.1.pt`` in ``models/``.
        conf_threshold: Minimum confidence for a detection to be returned.
            Must be in ``(0.0, 1.0]``.
        device: Device string forwarded to ultralytics (``"cpu"``,
            ``"cuda:0"``, etc.).  ``None`` lets ultralytics auto-select.

    Raises:
        ValueError: If *conf_threshold* is not in ``(0.0, 1.0]``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        if not 0.0 < conf_threshold <= 1.0:
            raise ValueError(f"conf_threshold must be in (0.0, 1.0], got {conf_threshold}")

        self._model_path = Path(model_path) if model_path is not None else _DEFAULT_WEIGHTS_PATH
        self._conf_threshold = conf_threshold
        self._device = device
        self._model: YOLO | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> YOLO:
        """Load the YOLO model from disk.

        If the default fine-tuned weights are configured, they are
        downloaded and integrity-checked before loading.  Custom paths
        are assumed to be user-managed and are loaded directly.

        Returns:
            The loaded ``ultralytics.YOLO`` model instance.
        """
        # Auto-download default fine-tuned weights if missing.
        # Custom paths are user-managed; we don't touch them.
        if self._model_path == _DEFAULT_WEIGHTS_PATH:
            _ensure_weights_available(self._model_path)

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
