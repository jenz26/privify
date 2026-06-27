"""Tests for the video processing pipeline.

Every test runs without real video files or YOLO model weights.
A small synthetic video (10 frames, 64×64, solid colour) is generated
via a pytest fixture and written to ``tmp_path`` for automatic cleanup.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.anonymizer import Anonymizer
from src.detector import Detection, Detector
from src.pipeline import (
    _PERSON_CLASS_FILTER,
    _UPPER_FRACTION,
    ProcessingStats,
    process_video,
    resolve_components,
)

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

        stats = process_video(fake_video, output, detector=detector, anonymizer=anonymizer)

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
                detector=detector,
                anonymizer=anonymizer,
            )

    def test_process_video_calls_detector_and_anonymizer_per_frame(
        self, fake_video: Path, tmp_path: Path
    ):
        """Detector and anonymizer must be invoked once per frame."""
        detector = _make_mock_detector()
        anonymizer = _make_mock_anonymizer()

        process_video(fake_video, tmp_path / "output.mp4", detector=detector, anonymizer=anonymizer)

        assert detector.detect.call_count == NUM_TEST_FRAMES
        assert anonymizer.anonymize.call_count == NUM_TEST_FRAMES

    def test_process_video_reports_progress_per_frame(self, fake_video: Path, tmp_path: Path):
        """on_progress fires once per frame: current goes 1..N, total stays constant."""
        detector = _make_mock_detector()
        anonymizer = _make_mock_anonymizer()
        calls: list[tuple[int, int]] = []

        process_video(
            fake_video,
            tmp_path / "output.mp4",
            detector=detector,
            anonymizer=anonymizer,
            on_progress=lambda current, total: calls.append((current, total)),
        )

        # One callback per processed frame.
        assert len(calls) == NUM_TEST_FRAMES
        # current is 1-indexed and strictly monotonic up to N.
        currents = [current for current, _ in calls]
        assert currents == list(range(1, NUM_TEST_FRAMES + 1))
        # total is the same value in every call.
        totals = {total for _, total in calls}
        assert len(totals) == 1


class TestProcessVideoModes:
    """The mode parameter wires the correct detector/anonymizer per scenario."""

    def test_process_video_face_mode(self, fake_video: Path, tmp_path: Path):
        """mode='face' uses the default face Detector and full-box Anonymizer."""
        with (
            patch("src.pipeline.Detector") as mock_det,
            patch("src.pipeline.Anonymizer") as mock_anon,
        ):
            mock_det.return_value.detect.return_value = []
            mock_anon.return_value.anonymize.side_effect = lambda frame, _dets: frame

            process_video(fake_video, tmp_path / "out.mp4", mode="face")

        mock_det.assert_called_once_with()
        mock_anon.assert_called_once_with(box_mode="full")

    def test_process_video_person_upper_mode(self, fake_video: Path, tmp_path: Path):
        """mode='person_upper' uses the COCO person filter and upper-region blur."""
        with (
            patch("src.pipeline.Detector") as mock_det,
            patch("src.pipeline.Anonymizer") as mock_anon,
        ):
            mock_det.return_value.detect.return_value = []
            mock_anon.return_value.anonymize.side_effect = lambda frame, _dets: frame

            process_video(fake_video, tmp_path / "out.mp4", mode="person_upper")

        mock_det.assert_called_once_with(model_path="yolov8n.pt", class_filter=["person"])
        mock_anon.assert_called_once_with(box_mode="upper", upper_fraction=0.30)

    def test_process_video_person_full_mode(self, fake_video: Path, tmp_path: Path):
        """mode='person_full' uses the COCO person filter and full-box blur."""
        with (
            patch("src.pipeline.Detector") as mock_det,
            patch("src.pipeline.Anonymizer") as mock_anon,
        ):
            mock_det.return_value.detect.return_value = []
            mock_anon.return_value.anonymize.side_effect = lambda frame, _dets: frame

            process_video(fake_video, tmp_path / "out.mp4", mode="person_full")

        mock_det.assert_called_once_with(model_path="yolov8n.pt", class_filter=["person"])
        mock_anon.assert_called_once_with(box_mode="full")

    def test_process_video_requires_mode_or_detectors(self, fake_video: Path, tmp_path: Path):
        """Neither mode nor a detector/anonymizer pair provided → ValueError."""
        with pytest.raises(ValueError, match="either 'mode'"):
            process_video(fake_video, tmp_path / "out.mp4")


class TestResolveComponents:
    """Unit tests for the resolve_components factory.

    These tests build each preset pair through the factory and inspect its
    configuration only.  Because the detector loads YOLO lazily (on the first
    ``detect`` call, not in ``__init__``), construction touches no model
    weights, disk, or network, so the suite runs even with ultralytics absent.
    The classes expose no public accessor for the mode-distinguishing config,
    so the assertions read the stored attributes (``_class_filter``,
    ``_box_mode``, ``_upper_fraction``) directly.  Values are checked against
    the module constants, not literals, so the tests track the source of truth.
    """

    def test_face_mode_wires_face_detector_and_full_blur(self):
        """mode='face': no class filter, full-box anonymizer."""
        detector, anonymizer = resolve_components("face", None, None)

        assert isinstance(detector, Detector)
        assert isinstance(anonymizer, Anonymizer)
        assert detector._class_filter is None
        assert anonymizer._box_mode == "full"

    def test_person_upper_mode_wires_person_filter_and_upper_blur(self):
        """mode='person_upper': person class filter, upper-region blur."""
        detector, anonymizer = resolve_components("person_upper", None, None)

        assert isinstance(detector, Detector)
        assert isinstance(anonymizer, Anonymizer)
        assert detector._class_filter == _PERSON_CLASS_FILTER
        assert anonymizer._box_mode == "upper"
        assert anonymizer._upper_fraction == _UPPER_FRACTION

    def test_person_full_mode_wires_person_filter_and_full_blur(self):
        """mode='person_full': person class filter, full-box blur."""
        detector, anonymizer = resolve_components("person_full", None, None)

        assert isinstance(detector, Detector)
        assert isinstance(anonymizer, Anonymizer)
        assert detector._class_filter == _PERSON_CLASS_FILTER
        assert anonymizer._box_mode == "full"

    def test_dependency_injection_returns_injected_objects(self):
        """mode=None with both collaborators returns them unchanged (by identity)."""
        detector = Detector()
        anonymizer = Anonymizer()

        out_detector, out_anonymizer = resolve_components(None, detector, anonymizer)

        assert out_detector is detector
        assert out_anonymizer is anonymizer

    def test_no_mode_and_no_components_raises(self):
        """mode=None with neither collaborator supplied → ValueError."""
        with pytest.raises(ValueError):
            resolve_components(None, None, None)

    def test_partial_injection_detector_only_raises(self):
        """mode=None requires BOTH collaborators: detector alone → ValueError."""
        with pytest.raises(ValueError):
            resolve_components(None, Detector(), None)

    def test_partial_injection_anonymizer_only_raises(self):
        """mode=None requires BOTH collaborators: anonymizer alone → ValueError."""
        with pytest.raises(ValueError):
            resolve_components(None, None, Anonymizer())


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
