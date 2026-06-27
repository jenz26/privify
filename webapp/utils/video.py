"""Pure video I/O utilities for the web front-end.

This module provides small, dependency-light helpers to inspect a video and to
pull a handful of frames for a UI preview.  It deliberately holds no model,
detection, or ``ffmpeg`` logic: it depends only on OpenCV (``cv2``) and NumPy,
so it stays reusable and unit-testable without a running Streamlit app.

The frame-selection logic (:func:`equispaced_indices`) is a *pure* function
with no I/O, which makes its sampling contract trivial to test in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Immutable summary of a video's basic metadata.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second reported by the container.  May be ``0.0`` when
            the codec does not expose a frame rate.
        frame_count: Number of frames reported by the container.
    """

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        """Video duration in seconds, or ``0.0`` when *fps* is non-positive."""
        if self.fps > 0:
            return self.frame_count / self.fps
        return 0.0


def get_video_info(path: Path) -> VideoInfo:
    """Read basic metadata from the video at *path*.

    Opens the file with OpenCV, reads width/height/fps/frame-count from the
    container, and always releases the capture before returning.

    Args:
        path: Path to the source video file.

    Returns:
        A :class:`VideoInfo` describing the video.

    Raises:
        FileNotFoundError: If *path* does not exist or cannot be opened by
            OpenCV.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count)


def equispaced_indices(frame_count: int, n_samples: int) -> list[int]:
    """Pick *n_samples* frame indices spread evenly across a video.

    Indices are sampled at the *centre* of ``n_samples`` equal segments, which
    keeps them away from the first and last frames (often black or transition
    frames) and spreads them uniformly in between.  This is a pure function: it
    performs no I/O and only computes integer indices.

    Args:
        frame_count: Total number of frames in the video.  Must be ``>= 1``.
        n_samples: Desired number of sample indices.  Must be ``>= 1``.

    Returns:
        A sorted list of unique frame indices in ``[0, frame_count - 1]``.  When
        *n_samples* is greater than or equal to *frame_count*, every frame index
        is returned (``list(range(frame_count))``).  Otherwise at most
        *n_samples* indices are returned; rounding collisions may yield slightly
        fewer, which is acceptable for a robust preview selection.

    Raises:
        ValueError: If *frame_count* or *n_samples* is less than ``1``.
    """
    if frame_count < 1:
        raise ValueError(f"frame_count must be >= 1, got {frame_count}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    if n_samples >= frame_count:
        return list(range(frame_count))

    seen: set[int] = set()
    indices: list[int] = []
    for k in range(n_samples):
        raw = round((k + 0.5) * frame_count / n_samples)
        idx = max(0, min(raw, frame_count - 1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)

    return indices


def extract_frames(path: Path, indices: Sequence[int]) -> list[np.ndarray]:
    """Extract the frames at *indices* from the video at *path*.

    Opens the capture once and, for each requested index, seeks with
    ``CAP_PROP_POS_FRAMES`` and reads a single frame.  Indices that fall out of
    range or whose read fails are skipped silently, so the result contains only
    the frames that could actually be decoded.  This makes the helper safe to
    feed an arbitrary index list for a best-effort UI preview.

    Note:
        Seeking by frame position can be approximate on some codecs (the backend
        may land on the nearest keyframe).  This is acceptable for a preview and
        the function does not attempt frame-accurate seeking.

    Args:
        path: Path to the source video file.
        indices: Frame indices to extract, in any order.

    Returns:
        A list of BGR frames as :class:`numpy.ndarray` of shape ``(H, W, 3)``,
        one per successfully read index, in the order of *indices*.

    Raises:
        FileNotFoundError: If *path* does not exist or cannot be opened by
            OpenCV.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {path}")

    frames: list[np.ndarray] = []
    try:
        for index in indices:
            if index < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
    finally:
        cap.release()

    return frames
