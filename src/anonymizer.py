"""Frame anonymization via Gaussian blur on detected regions.

This module applies Gaussian blur to bounding-box regions identified by
:class:`~src.detector.Detection` instances.  It is the last stage of the
pipeline: it transforms "here are the sensitive subjects" into "the video
is anonymized".

Design invariants:

- **Stateless**: configuration lives in the constructor; :meth:`anonymize`
  is a pure transformation with no side effects.
- **Non-destructive**: the input frame is never modified; a copy is returned.
- **No ultralytics dependency**: consistent with the wrapper pattern, this
  module depends only on NumPy, OpenCV, and the :class:`Detection` dataclass.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.detector import Detection

logger = logging.getLogger(__name__)

# Minimum dimension (width or height) for a bbox to be blurred.
DEFAULT_MIN_BBOX_SIZE: int = 4


class Anonymizer:
    """Applies Gaussian blur to detected regions in a video frame.

    Args:
        blur_kernel_size: Side length of the square Gaussian kernel.
            Must be a positive odd integer.
        min_bbox_size: Minimum width **and** height (in pixels) for a
            detection to be blurred.  Detections smaller than this on
            either axis are silently skipped.  Defaults to 4.

    Raises:
        ValueError: If *blur_kernel_size* is not a positive odd integer.
    """

    def __init__(
        self,
        blur_kernel_size: int = 51,
        min_bbox_size: int = DEFAULT_MIN_BBOX_SIZE,
    ) -> None:
        if blur_kernel_size <= 0 or blur_kernel_size % 2 == 0:
            raise ValueError(
                f"blur_kernel_size must be a positive odd integer, got {blur_kernel_size}"
            )
        if min_bbox_size < 0:
            raise ValueError(f"min_bbox_size must be >= 0, got {min_bbox_size}")

        self._kernel_size = blur_kernel_size
        self._min_bbox_size = min_bbox_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anonymize(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Return a copy of *frame* with detected regions blurred.

        Args:
            frame: A ``(H, W, 3)`` BGR image as a NumPy array.
            detections: Detections whose bounding boxes will be blurred.

        Returns:
            A **new** array with Gaussian blur applied inside each valid
            bounding box.  The input *frame* is never modified.
        """
        output = frame.copy()

        if not detections:
            return output

        h, w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Clip coordinates to frame boundaries.
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            roi_w = x2 - x1
            roi_h = y2 - y1

            if roi_w < self._min_bbox_size or roi_h < self._min_bbox_size:
                logger.debug(
                    "Skipping detection with bbox %s: ROI size %dx%d below minimum %d",
                    det.bbox,
                    roi_w,
                    roi_h,
                    self._min_bbox_size,
                )
                continue

            roi = output[y1:y2, x1:x2]
            output[y1:y2, x1:x2] = cv2.GaussianBlur(
                roi,
                (self._kernel_size, self._kernel_size),
                0,
            )

        logger.debug("Anonymized %d regions in frame", len(detections))
        return output
