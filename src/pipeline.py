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
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.anonymizer import Anonymizer
from src.detector import Detector

logger = logging.getLogger(__name__)


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


def process_video(
    input_path: Path,
    output_path: Path,
    detector: Detector,
    anonymizer: Anonymizer,
) -> ProcessingStats:
    """Read *input_path* frame-by-frame, anonymize, and write to *output_path*.

    Args:
        input_path: Path to the source video file.
        output_path: Path where the anonymized video will be written.
            Parent directories must already exist.
        detector: A :class:`~src.detector.Detector` (or compatible object)
            whose ``detect(frame)`` method returns a list of detections.
        anonymizer: An :class:`~src.anonymizer.Anonymizer` (or compatible
            object) whose ``anonymize(frame, detections)`` method returns
            the blurred frame.

    Returns:
        A :class:`ProcessingStats` summarising the run.

    Raises:
        FileNotFoundError: If *input_path* does not exist or cannot be
            opened by OpenCV.
    """
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
