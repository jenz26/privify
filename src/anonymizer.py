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
from typing import Literal

import cv2
import numpy as np

from src.detector import Detection

logger = logging.getLogger(__name__)

# Minimum dimension (width or height) for a bbox to be blurred.
DEFAULT_MIN_BBOX_SIZE: int = 4

# Default fraction of the bbox height blurred in "upper" box_mode.
DEFAULT_UPPER_FRACTION: float = 0.30


class Anonymizer:
    """Applies Gaussian blur to detected regions in a video frame.

    Two blur regions are supported via *box_mode*:

    - ``"full"`` (default) blurs the entire bounding box.  This is correct
      for a face detector, whose boxes already tightly frame the sensitive
      region.
    - ``"upper"`` blurs only the top *upper_fraction* of the box.  This is a
      head-shoulders proxy for CCTV scenes where a COCO ``person`` detector
      returns a full-body box: blurring just the upper slice anonymizes the
      identifying region (the head) without obscuring the whole silhouette.

    Args:
        blur_kernel_size: Side length of the square Gaussian kernel.
            Must be a positive odd integer.
        min_bbox_size: Minimum width **and** height (in pixels) for a
            detection to be blurred.  Detections smaller than this on
            either axis are silently skipped.  Defaults to 4.
        box_mode: ``"full"`` to blur the whole bbox, or ``"upper"`` to blur
            only its top *upper_fraction* (head-shoulders proxy).
        upper_fraction: Fraction of the bbox height blurred in ``"upper"``
            mode.  Must be in ``(0.0, 1.0]``.  Defaults to 0.30.

    Raises:
        ValueError: If *blur_kernel_size* is not a positive odd integer, if
            *box_mode* is not ``"full"`` or ``"upper"``, or if
            *upper_fraction* is not in ``(0.0, 1.0]``.
    """

    def __init__(
        self,
        blur_kernel_size: int = 51,
        min_bbox_size: int = DEFAULT_MIN_BBOX_SIZE,
        box_mode: Literal["full", "upper"] = "full",
        upper_fraction: float = DEFAULT_UPPER_FRACTION,
    ) -> None:
        if blur_kernel_size <= 0 or blur_kernel_size % 2 == 0:
            raise ValueError(
                f"blur_kernel_size must be a positive odd integer, got {blur_kernel_size}"
            )
        if min_bbox_size < 0:
            raise ValueError(f"min_bbox_size must be >= 0, got {min_bbox_size}")
        if box_mode not in ("full", "upper"):
            raise ValueError(f"box_mode must be 'full' or 'upper', got {box_mode!r}")
        if not 0.0 < upper_fraction <= 1.0:
            raise ValueError(f"upper_fraction must be in (0.0, 1.0], got {upper_fraction}")

        self._kernel_size = blur_kernel_size
        self._min_bbox_size = min_bbox_size
        self._box_mode = box_mode
        self._upper_fraction = upper_fraction

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
            A **new** array with Gaussian blur applied to each valid
            detection.  In ``"full"`` mode the entire bounding box is
            blurred; in ``"upper"`` mode only its top *upper_fraction*.
            The input *frame* is never modified.
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

            # In "upper" mode blur only the top slice of the box (head-shoulders
            # proxy for full-body person boxes); guarantee at least 1px height.
            if self._box_mode == "upper":
                blur_y2 = max(y1 + 1, y1 + int(roi_h * self._upper_fraction))
            else:
                blur_y2 = y2

            roi = output[y1:blur_y2, x1:x2]
            output[y1:blur_y2, x1:x2] = cv2.GaussianBlur(
                roi,
                (self._kernel_size, self._kernel_size),
                0,
            )

        logger.debug("Anonymized %d regions in frame", len(detections))
        return output
