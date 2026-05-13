"""Unit tests for src.detector — no real model weights required."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector import Detection, Detector, _ensure_weights_available, _verify_sha256

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

        with patch("ultralytics.YOLO", mock_yolo_cls):
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

        with patch("ultralytics.YOLO", mock_yolo_cls):
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

        with patch("ultralytics.YOLO", mock_yolo_cls):
            grayscale = np.zeros((480, 640), dtype=np.uint8)
            with pytest.raises(ValueError, match="3-channel"):
                detector.detect(grayscale)


# ------------------------------------------------------------------
# SHA-256 verification tests
# ------------------------------------------------------------------


class TestVerifySha256:
    """Tests for the :func:`_verify_sha256` helper."""

    def test_verify_sha256_correct_hash(self, tmp_path: Path) -> None:
        """Returns True when the file hash matches the expected digest."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert _verify_sha256(f, expected) is True

    def test_verify_sha256_wrong_hash(self, tmp_path: Path) -> None:
        """Returns False when the file hash does not match."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        assert _verify_sha256(f, "0" * 64) is False


# ------------------------------------------------------------------
# Weight download tests (mocked, no real network)
# ------------------------------------------------------------------


class TestEnsureWeightsAvailable:
    """Tests for :func:`_ensure_weights_available` with mocked downloads."""

    def test_skips_download_when_valid_file_exists(self, tmp_path: Path) -> None:
        """No download if file exists and SHA-256 matches."""
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"fake weights")
        valid_hash = hashlib.sha256(b"fake weights").hexdigest()

        with (
            patch("src.detector.WEIGHTS_SHA256", valid_hash),
            patch("src.detector.urllib.request.urlretrieve") as mock_dl,
        ):
            _ensure_weights_available(weights)

        mock_dl.assert_not_called()

    def test_downloads_when_file_missing(self, tmp_path: Path) -> None:
        """Downloads weights and passes integrity check."""
        weights = tmp_path / "models" / "model.pt"
        payload = b"downloaded weights"
        expected_hash = hashlib.sha256(payload).hexdigest()

        def fake_download(url: str, dest: Path) -> None:
            dest.write_bytes(payload)

        with (
            patch("src.detector.WEIGHTS_SHA256", expected_hash),
            patch("src.detector.urllib.request.urlretrieve", side_effect=fake_download),
        ):
            _ensure_weights_available(weights)

        assert weights.exists()

    def test_raises_on_integrity_failure(self, tmp_path: Path) -> None:
        """Raises RuntimeError if downloaded file fails SHA-256 check."""
        weights = tmp_path / "model.pt"

        def fake_download(url: str, dest: Path) -> None:
            dest.write_bytes(b"corrupted")

        with (
            patch("src.detector.WEIGHTS_SHA256", "0" * 64),
            patch("src.detector.urllib.request.urlretrieve", side_effect=fake_download),
            pytest.raises(RuntimeError, match="integrity check"),
        ):
            _ensure_weights_available(weights)

        assert not weights.exists()
