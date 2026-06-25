"""Unit tests for src.anonymizer — no GPU, no model weights."""

from __future__ import annotations

import numpy as np
import pytest

from src.anonymizer import Anonymizer
from src.detector import Detection


def _make_frame(
    height: int = 480,
    width: int = 640,
    value: int = 128,
) -> np.ndarray:
    """Create a solid-colour BGR test frame."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _make_detection(
    x1: int = 100,
    y1: int = 100,
    x2: int = 200,
    y2: int = 200,
) -> Detection:
    """Create a Detection with a configurable bbox."""
    return Detection(bbox=(x1, y1, x2, y2), confidence=0.9, class_id=0, class_name="person")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestAnonymizer:
    """Tests for :class:`Anonymizer`."""

    def test_anonymize_no_detections_returns_unchanged_frame(self) -> None:
        """Empty detection list produces a pixel-identical output."""
        frame = _make_frame()
        anonymizer = Anonymizer(blur_kernel_size=51)

        result = anonymizer.anonymize(frame, [])

        assert np.array_equal(result, frame)

    def test_anonymize_does_not_modify_input_frame(self) -> None:
        """The original frame must be bit-identical after anonymize()."""
        frame = _make_frame()
        original = frame.copy()
        anonymizer = Anonymizer(blur_kernel_size=51)
        det = _make_detection()

        anonymizer.anonymize(frame, [det])

        assert np.array_equal(frame, original)

    def test_anonymize_blurs_only_inside_bbox(self) -> None:
        """Pixels inside the bbox change; pixels outside remain identical."""
        # Use a checkerboard-like pattern so blur has visible effect.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[::2, ::2] = 255  # every other pixel is white

        det = _make_detection(x1=30, y1=30, x2=70, y2=70)
        anonymizer = Anonymizer(blur_kernel_size=15)

        result = anonymizer.anonymize(frame, [det])

        # Outside the bbox: unchanged.
        assert np.array_equal(result[:30, :, :], frame[:30, :, :])
        assert np.array_equal(result[70:, :, :], frame[70:, :, :])
        assert np.array_equal(result[:, :30, :], frame[:, :30, :])
        assert np.array_equal(result[:, 70:, :], frame[:, 70:, :])

        # Inside the bbox: blurred (at least some pixels differ).
        assert not np.array_equal(result[30:70, 30:70, :], frame[30:70, 30:70, :])

    def test_invalid_kernel_size_raises(self) -> None:
        """Even, zero, or negative kernel sizes raise ValueError."""
        with pytest.raises(ValueError, match="blur_kernel_size"):
            Anonymizer(blur_kernel_size=0)
        with pytest.raises(ValueError, match="blur_kernel_size"):
            Anonymizer(blur_kernel_size=50)
        with pytest.raises(ValueError, match="blur_kernel_size"):
            Anonymizer(blur_kernel_size=-3)

    def test_bbox_outside_frame_does_not_crash(self) -> None:
        """Bboxes exceeding frame boundaries are clipped, not crashed."""
        frame = _make_frame(height=100, width=100)
        original = frame.copy()

        # bbox extends well beyond frame on all sides.
        det = _make_detection(x1=-50, y1=-50, x2=200, y2=200)
        anonymizer = Anonymizer(blur_kernel_size=15)

        result = anonymizer.anonymize(frame, [det])

        # Should not crash and should return a valid frame.
        assert result.shape == frame.shape
        # Input unchanged.
        assert np.array_equal(frame, original)

    def test_bbox_smaller_than_min_size_is_skipped(self) -> None:
        """Tiny bboxes below the minimum size threshold are ignored."""
        frame = _make_frame()
        det = _make_detection(x1=50, y1=50, x2=52, y2=52)  # 2x2 bbox
        anonymizer = Anonymizer(blur_kernel_size=51, min_bbox_size=4)

        result = anonymizer.anonymize(frame, [det])

        assert np.array_equal(result, frame)

    def test_invalid_upper_fraction_raises(self) -> None:
        """upper_fraction outside (0.0, 1.0] raises ValueError."""
        for bad in (0.0, 1.1, -0.5):
            with pytest.raises(ValueError, match="upper_fraction"):
                Anonymizer(upper_fraction=bad)

    def test_invalid_box_mode_raises(self) -> None:
        """An unknown box_mode raises ValueError."""
        with pytest.raises(ValueError, match="box_mode"):
            Anonymizer(box_mode="middle")  # type: ignore[arg-type]

    def test_box_mode_full_matches_default_behavior(self) -> None:
        """box_mode='full' (explicit) blurs the same region as the default."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        det = _make_detection(x1=30, y1=30, x2=70, y2=70)

        default_result = Anonymizer(blur_kernel_size=15).anonymize(frame, [det])
        full_result = Anonymizer(blur_kernel_size=15, box_mode="full").anonymize(frame, [det])

        assert np.array_equal(default_result, full_result)

    def test_box_mode_upper_blurs_only_top_fraction(self) -> None:
        """upper mode blurs the top 30% of the bbox and leaves the rest intact."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[::2, ::2] = 255  # high-frequency pattern so blur is detectable

        x1, y1, x2, y2 = 40, 40, 160, 140  # 120 wide, 100 tall
        det = _make_detection(x1=x1, y1=y1, x2=x2, y2=y2)
        upper_fraction = 0.30
        anonymizer = Anonymizer(
            blur_kernel_size=15, box_mode="upper", upper_fraction=upper_fraction
        )

        result = anonymizer.anonymize(frame, [det])

        height = y2 - y1
        split = y1 + int(height * upper_fraction)  # boundary row (blurred height = 30)

        # Top slice [y1:split] is blurred (changed).
        assert not np.array_equal(result[y1:split, x1:x2], frame[y1:split, x1:x2])
        # Lower slice [split:y2] inside the bbox is untouched.
        assert np.array_equal(result[split:y2, x1:x2], frame[split:y2, x1:x2])
        # Everything outside the bbox is untouched.
        assert np.array_equal(result[:y1, :], frame[:y1, :])
        assert np.array_equal(result[y2:, :], frame[y2:, :])
