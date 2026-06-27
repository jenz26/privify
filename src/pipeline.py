"""Video processing pipeline: read → detect → anonymize → write.

This module orchestrates the full anonymization pipeline by composing
:class:`~src.detector.Detector` and :class:`~src.anonymizer.Anonymizer`
over every frame of an input video.

Both collaborators are received via dependency injection so that callers
can configure them freely and tests can substitute mocks without touching
the filesystem or loading model weights.

The output video uses the ``mp4v`` codec (MPEG-4 Part 2) via
:func:`cv2.VideoWriter_fourcc`.  This codec is compiled into every OpenCV
build, including the default ``opencv-python-headless`` wheel available on
Google Colab, so no external ``ffmpeg`` with proprietary H.264 codecs is
required.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2

from src.anonymizer import Anonymizer
from src.detector import Detector

logger = logging.getLogger(__name__)

# Pre-trained COCO weights used by the person-based modes. This is the bare
# registry name (no directory): ultralytics resolves it from its own model
# registry and downloads/caches the COCO weights automatically on first use,
# on any host, without depending on a local models/ directory.
_COCO_WEIGHTS_PATH = "yolov8n.pt"
_PERSON_CLASS_FILTER = ["person"]
_UPPER_FRACTION = 0.30


@dataclass(frozen=True)
class ProcessingStats:
    """Immutable summary of a completed pipeline run.

    Attributes:
        total_frames: Number of frames read from the input video.
        total_detections: Cumulative detections across all frames.
        elapsed_seconds: Wall-clock time of the processing loop.
        fps_processed: Throughput in frames per second.
    """

    total_frames: int
    total_detections: int
    elapsed_seconds: float

    @property
    def fps_processed(self) -> float:
        """Effective processing throughput (frames / second)."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_frames / self.elapsed_seconds


def resolve_components(
    mode: Literal["face", "person_upper", "person_full"] | None,
    detector: Detector | None,
    anonymizer: Anonymizer | None,
) -> tuple[Detector, Anonymizer]:
    """Select the detector/anonymizer pair for a run.

    This is the single source of truth mapping a preset *mode* to its
    ``(Detector, Anonymizer)`` pair, shared by the video pipeline and any
    other entry point (such as a web front-end) that needs the same presets.

    Two resolution strategies are supported:

    - **By mode**: when *mode* is given, build the preset pair for that mode
      and ignore any injected *detector*/*anonymizer*.
    - **By dependency injection**: when *mode* is ``None``, both *detector*
      and *anonymizer* must be supplied and are returned unchanged.

    Args:
        mode: Preset hybrid configuration, or ``None`` to use dependency
            injection.  One of ``"face"``, ``"person_upper"``,
            ``"person_full"``.
        detector: A :class:`~src.detector.Detector`.  Used only when *mode*
            is ``None``; ignored otherwise.
        anonymizer: An :class:`~src.anonymizer.Anonymizer`.  Used only when
            *mode* is ``None``; ignored otherwise.

    Returns:
        The resolved ``(detector, anonymizer)`` pair.

    Raises:
        ValueError: If *mode* is unknown, or if neither *mode* nor both
            *detector* and *anonymizer* are provided.
    """
    if mode is not None:
        if mode == "face":
            return Detector(), Anonymizer(box_mode="full")
        if mode == "person_upper":
            return (
                Detector(model_path=_COCO_WEIGHTS_PATH, class_filter=_PERSON_CLASS_FILTER),
                Anonymizer(box_mode="upper", upper_fraction=_UPPER_FRACTION),
            )
        if mode == "person_full":
            return (
                Detector(model_path=_COCO_WEIGHTS_PATH, class_filter=_PERSON_CLASS_FILTER),
                Anonymizer(box_mode="full"),
            )
        raise ValueError(f"unknown mode {mode!r}")
    if detector is not None and anonymizer is not None:
        return detector, anonymizer
    raise ValueError("either 'mode' or ('detector' and 'anonymizer') must be provided")


def process_video(
    input_path: Path,
    output_path: Path,
    *,
    mode: Literal["face", "person_upper", "person_full"] | None = None,
    detector: Detector | None = None,
    anonymizer: Anonymizer | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ProcessingStats:
    """Read *input_path* frame-by-frame, anonymize, and write to *output_path*.

    The detector/anonymizer pair is selected in one of two ways:

    - **By mode** (recommended): pass *mode* to pick a preset hybrid
      configuration.  This forces an explicit policy decision per deployment
      context and ignores any *detector*/*anonymizer* arguments.

      - ``"face"``: fine-tuned face detector, full-box blur (close-up scenes).
      - ``"person_upper"``: COCO ``person`` detector, upper-region blur
        (head-shoulders proxy, conservative for CCTV).
      - ``"person_full"``: COCO ``person`` detector, full-box blur
        (full-body anonymization, maximally conservative).

    - **By dependency injection** (advanced): leave *mode* as ``None`` and
      pass both *detector* and *anonymizer* explicitly.

    Args:
        input_path: Path to the source video file.
        output_path: Path where the anonymized video will be written.
            Parent directories must already exist.
        mode: Preset hybrid configuration (keyword-only).  When given,
            *detector*/*anonymizer* are built automatically and any injected
            ones are ignored.
        detector: A :class:`~src.detector.Detector` (keyword-only).  Used
            only when *mode* is ``None``.
        anonymizer: An :class:`~src.anonymizer.Anonymizer` (keyword-only).
            Used only when *mode* is ``None``.
        on_progress: Optional callback (keyword-only) invoked once per
            processed frame with ``(current, total)``, where *current* is the
            1-indexed frame number and *total* is the frame count reported by
            the container, or ``0`` when unknown.

    Returns:
        A :class:`ProcessingStats` summarising the run.

    Raises:
        ValueError: If neither *mode* nor both *detector* and *anonymizer*
            are provided.
        FileNotFoundError: If *input_path* does not exist or cannot be
            opened by OpenCV.
    """
    detector, anonymizer = resolve_components(mode, detector, anonymizer)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Processing %s → %s (%dx%d, %.1f fps, ~%d frames)",
        input_path,
        output_path,
        width,
        height,
        fps,
        frame_count,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = 0
    total_detections = 0
    progress_total = frame_count if frame_count > 0 else 0
    start = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            anonymized = anonymizer.anonymize(frame, detections)

            writer.write(anonymized)

            total_detections += len(detections)
            total_frames += 1

            if total_frames % 100 == 0:
                logger.debug("Processed %d frames so far …", total_frames)

            if on_progress is not None:
                on_progress(total_frames, progress_total)
    finally:
        cap.release()
        writer.release()

    elapsed = time.monotonic() - start

    stats = ProcessingStats(
        total_frames=total_frames,
        total_detections=total_detections,
        elapsed_seconds=elapsed,
    )

    logger.info(
        "Done: %d frames, %d detections, %.1f s (%.1f fps)",
        stats.total_frames,
        stats.total_detections,
        stats.elapsed_seconds,
        stats.fps_processed,
    )

    return stats
