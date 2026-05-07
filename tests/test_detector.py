"""Unit tests for src.detector — no real model weights required."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector import Detection, Detector


# ------------------------------------------------------------------
# Detection dataclass tests
# ------------------------------------------------------------------


class TestDetectionDataclass:
    """Tests for the :class:`Detection` frozen dataclass."""

    def test_detection_dataclass_is_frozen(self) -> None:
        """Assigning to a field of a frozen Detection raises an error."""
        det = Detection(
            bbox=(10, 20, 100, 200),
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            det.confidence = 0.5  # type: ignore[misc]

    def test_detection_dataclass_fields(self) -> None:
        """Detection exposes exactly the expected fields and types."""
        det = Detection(
            bbox=(0, 0, 50, 50),
            confidence=0.8,
            class_id=1,
            class_name="plate",
        )
        assert det.bbox == (0, 0, 50, 50)
        assert isinstance(det.confidence, float)
        assert isinstance(det.class_id, int)
        assert isinstance(det.class_name, str)

        field_names = {f.name for f in dataclasses.fields(det)}
        assert field_names == {"bbox", "confidence", "class_id", "class_name"}


# ------------------------------------------------------------------
# Detector tests (all mocked, no weights download)
# ------------------------------------------------------------------


class TestDetector:
    """Tests for :class:`Detector` lazy loading and detect() pipeline."""

    def test_detector_lazy_loading(self) -> None:
        """Model is None after __init__; loaded only when detect() is called."""
        detector = Detector(model_path="fake.pt", conf_threshold=0.5)
        assert detector._model is None  # noqa: SLF001 — testing internals

        # Mock ultralytics.YOLO so no real weights are needed.
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # predict() must return an iterable of result objects.
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = lambda self: 0
        mock_model.predict.return_value = [mock_result]

        with patch("src.detector.YOLO", mock_yolo_cls, create=True):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detector.detect(frame)

        assert detector._model is not None  # noqa: SLF001

    def test_detector_detect_returns_list_of_detections(self) -> None:
        """detect() converts raw YOLO results into a list[Detection]."""
        detector = Detector(model_path="fake.pt", conf_threshold=0.3)

        # Build a mock result that looks like a single-box YOLO output.
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 1
        mock_boxes.xyxy = [np.array([10.0, 20.0, 110.0, 220.0])]
        mock_boxes.conf = [np.float32(0.92)]
        mock_boxes.cls = [np.float32(0)]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        mock_yolo_cls = MagicMock(return_value=mock_model)

        with patch("src.detector.YOLO", mock_yolo_cls, create=True):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)

        assert isinstance(detections, list)
        assert len(detections) == 1

        det = detections[0]
        assert isinstance(det, Detection)
        assert det.bbox == (10, 20, 110, 220)
        assert 0.0 < det.confidence <= 1.0
        assert det.class_id == 0
        assert det.class_name == "person"

    def test_detector_rejects_invalid_conf_threshold(self) -> None:
        """Detector raises ValueError for out-of-range conf_threshold."""
        with pytest.raises(ValueError, match="conf_threshold"):
            Detector(model_path="fake.pt", conf_threshold=0.0)
        with pytest.raises(ValueError, match="conf_threshold"):
            Detector(model_path="fake.pt", conf_threshold=1.5)

    def test_detector_rejects_invalid_frame(self) -> None:
        """detect() raises ValueError for non-3-channel input."""
        detector = Detector(model_path="fake.pt")

        mock_model = MagicMock()
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with patch("src.detector.YOLO", mock_yolo_cls, create=True):
            grayscale = np.zeros((480, 640), dtype=np.uint8)
            with pytest.raises(ValueError, match="3-channel"):
                detector.detect(grayscale)
