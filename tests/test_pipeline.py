"""Tests for the video processing pipeline.

Every test runs without real video files or YOLO model weights.
A small synthetic video (10 frames, 64×64, solid colour) is generated
via a pytest fixture and written to ``tmp_path`` for automatic cleanup.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.detector import Detection
from src.pipeline import ProcessingStats, process_video

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

NUM_TEST_FRAMES = 10
FRAME_W, FRAME_H = 64, 64

_STUB_DETECTION = Detection(
    bbox=(10, 10, 50, 50),
    confidence=0.9,
    class_id=0,
    class_name="person",
)


@pytest.fixture()
def fake_video(tmp_path: Path) -> Path:
    """Create a minimal 10-frame .mp4 video and return its path."""
    video_path = tmp_path / "input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, (FRAME_W, FRAME_H))

    for i in range(NUM_TEST_FRAMES):
        frame = np.full((FRAME_H, FRAME_W, 3), fill_value=i * 25, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


def _make_mock_detector(detections_per_frame: list[Detection] | None = None):
    """Return a mock whose ``detect`` returns *detections_per_frame*."""
    if detections_per_frame is None:
        detections_per_frame = [_STUB_DETECTION]
    mock = MagicMock()
    mock.detect.return_value = detections_per_frame
    return mock


def _make_mock_anonymizer():
    """Return a mock whose ``anonymize`` returns the frame unchanged."""
    mock = MagicMock()
    mock.anonymize.side_effect = lambda frame, _dets: frame
    return mock


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestProcessVideo:
    """Integration-style tests with mocked collaborators."""

    def test_process_video_returns_stats(self, fake_video: Path, tmp_path: Path):
        """Stats reflect 10 frames × 1 detection each."""
        detector = _make_mock_detector()
        anonymizer = _make_mock_anonymizer()
        output = tmp_path / "output.mp4"

        stats = process_video(fake_video, output, detector, anonymizer)

        assert stats.total_frames == NUM_TEST_FRAMES
        assert stats.total_detections == NUM_TEST_FRAMES
        assert stats.elapsed_seconds > 0
        assert output.exists()

    def test_process_video_missing_input_raises_filenotfound(self, tmp_path: Path):
        """A non-existent input path must raise FileNotFoundError."""
        detector = _make_mock_detector()
        anonymizer = _make_mock_anonymizer()

        with pytest.raises(FileNotFoundError):
            process_video(
                tmp_path / "nonexistent.mp4",
                tmp_path / "output.mp4",
                detector,
                anonymizer,
            )

    def test_process_video_calls_detector_and_anonymizer_per_frame(
        self, fake_video: Path, tmp_path: Path
    ):
        """Detector and anonymizer must be invoked once per frame."""
        detector = _make_mock_detector()
        anonymizer = _make_mock_anonymizer()

        process_video(fake_video, tmp_path / "output.mp4", detector, anonymizer)

        assert detector.detect.call_count == NUM_TEST_FRAMES
        assert anonymizer.anonymize.call_count == NUM_TEST_FRAMES


class TestProcessingStats:
    """Unit tests for the stats dataclass."""

    def test_processing_stats_is_frozen(self):
        """ProcessingStats must be immutable."""
        stats = ProcessingStats(total_frames=10, total_detections=5, elapsed_seconds=1.0)
        with pytest.raises(FrozenInstanceError):
            stats.total_frames = 99  # type: ignore[misc]

    def test_fps_processed_computed_correctly(self):
        """60 frames in 2 seconds → 30 fps."""
        stats = ProcessingStats(total_frames=60, total_detections=0, elapsed_seconds=2.0)
        assert stats.fps_processed == 30.0

    def test_fps_processed_zero_elapsed(self):
        """Zero elapsed seconds must not raise; returns 0."""
        stats = ProcessingStats(total_frames=10, total_detections=0, elapsed_seconds=0.0)
        assert stats.fps_processed == 0.0
