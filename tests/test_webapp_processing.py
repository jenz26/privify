"""Tests for the web front-end orchestration layer ``webapp.utils.processing``.

The detection/anonymization logic is exercised with **fakes**, never the real
YOLO model, so the suite runs without a GPU, without downloading weights, and
without network access (mirroring ``tests/test_pipeline.py``).

The two ``ffmpeg``-dependent tests (:func:`transcode_h264` and
:func:`run_full`) run for real against the static ``ffmpeg`` binary bundled by
``imageio-ffmpeg``. They are intentionally **not** skipped when the binary is
missing: a missing transcode dependency should fail loudly, not be hidden.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.detector import Detection
from src.pipeline import ProcessingStats
from webapp.utils.processing import (
    run_full,
    run_preview,
    select_representative_frames,
    transcode_h264,
)

# ------------------------------------------------------------------
# Fixtures and fakes
# ------------------------------------------------------------------

NUM_TEST_FRAMES = 10
# Even dimensions: libx264 with yuv420p requires width/height divisible by 2.
FRAME_W, FRAME_H = 64, 48
TEST_FPS = 25.0

# A stub detection reused everywhere a "detection" is needed; only its presence
# (count) matters to the orchestration layer, never its geometry.
_STUB_DETECTION = Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0, class_name="face")


def _write_synthetic_video(
    path: Path,
    *,
    num_frames: int = NUM_TEST_FRAMES,
    width: int = FRAME_W,
    height: int = FRAME_H,
    fps: float = TEST_FPS,
) -> Path:
    """Write a minimal solid-colour ``mp4v`` video and return its path.

    Frame ``i`` is filled with the constant value ``i * 25`` so frames are
    visually distinct, matching the fixtures in the other test modules.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=i * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def fake_video(tmp_path: Path) -> Path:
    """Create a 10-frame synthetic ``mp4v`` video and return its path."""
    return _write_synthetic_video(tmp_path / "input.mp4")


class _FakeDetector:
    """Detector stub returning a scripted number of detections per call.

    ``counts[k]`` is the number of detections returned by the *k*-th call to
    :meth:`detect`. ``select_representative_frames`` probes frames in ascending
    index order, so ``counts[k]`` is the detection count assigned to the
    *k*-th probed frame. Calls past the end of *counts* return zero detections,
    which keeps the fake robust if fewer frames decode than were probed.
    """

    def __init__(self, counts: list[int]) -> None:
        self._counts = list(counts)
        self._calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        n = self._counts[self._calls] if self._calls < len(self._counts) else 0
        self._calls += 1
        return [_STUB_DETECTION] * n


class _FakeAnonymizer:
    """Anonymizer stub returning an all-zero frame (recognizable output)."""

    def anonymize(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        return frame * 0


# ------------------------------------------------------------------
# select_representative_frames
# ------------------------------------------------------------------


class TestSelectRepresentativeFrames:
    """Frame ranking by detection count, with an ascending-order contract."""

    def test_keeps_frames_with_most_detections_in_temporal_order(self, fake_video: Path):
        """The 3 most detection-rich probed frames are returned, sorted by index."""
        # 10 probes (== frame count, so indices are 0..9). Detection counts make
        # indices 2 (5 dets), 5 (3 dets) and 8 (1 det) the clear winners.
        counts = [0, 0, 5, 0, 0, 3, 0, 0, 1, 0]
        detector = _FakeDetector(counts)

        selected = select_representative_frames(
            fake_video, detector, n_preview=3, n_probe=NUM_TEST_FRAMES
        )

        indices = [index for index, _ in selected]
        assert indices == [2, 5, 8]  # winners, ascending (temporal) order
        for _, frame in selected:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (FRAME_H, FRAME_W, 3)

    def test_fallback_returns_first_n_when_no_detections(self, fake_video: Path):
        """All-zero detections → first n_preview probed frames, ascending."""
        detector = _FakeDetector([0] * NUM_TEST_FRAMES)

        selected = select_representative_frames(
            fake_video, detector, n_preview=3, n_probe=NUM_TEST_FRAMES
        )

        indices = [index for index, _ in selected]
        assert indices == [0, 1, 2]  # first three probed frames, in order


# ------------------------------------------------------------------
# run_preview
# ------------------------------------------------------------------


class TestRunPreview:
    """The preview replays the exact detect→anonymize sequence per frame."""

    def test_pairs_original_with_anonymized_output(self):
        """Each pair is (original_frame, fake_anonymizer_output); length matches."""
        frames = [np.full((FRAME_H, FRAME_W, 3), v, dtype=np.uint8) for v in (10, 20, 30)]
        detector = _FakeDetector([0, 1, 2])
        anonymizer = _FakeAnonymizer()

        pairs = run_preview(frames, detector, anonymizer)

        assert len(pairs) == len(frames)
        for original, (before, after) in zip(frames, pairs, strict=True):
            assert np.array_equal(before, original)  # before is the untouched input
            assert np.array_equal(after, np.zeros_like(original))  # fake zeroed it


# ------------------------------------------------------------------
# transcode_h264
# ------------------------------------------------------------------


class TestTranscodeH264:
    """Real transcode against the imageio-ffmpeg bundled binary."""

    def test_produces_a_playable_h264_file(self, tmp_path: Path):
        """mp4v → H.264 yields a non-empty file OpenCV can open."""
        src = _write_synthetic_video(tmp_path / "src.mp4")
        dst = tmp_path / "src_h264.mp4"

        returned = transcode_h264(src, dst)

        assert returned == dst
        assert dst.exists()
        assert dst.stat().st_size > 0
        cap = cv2.VideoCapture(str(dst))
        try:
            assert cap.isOpened()
        finally:
            cap.release()


# ------------------------------------------------------------------
# run_full
# ------------------------------------------------------------------


class TestRunFull:
    """The full run delegates to process_video, then transcodes to H.264."""

    def test_returns_stats_and_existing_h264_path(self, tmp_path: Path):
        """process_video is mocked; the real mp4v output is transcoded to H.264."""
        input_path = tmp_path / "input.mp4"  # never read: process_video is mocked
        output_path = tmp_path / "output.mp4"
        # The pipeline would write this mp4v file; we create a real one so the
        # transcode step has genuine input to convert.
        _write_synthetic_video(output_path, num_frames=5)

        known_stats = ProcessingStats(total_frames=5, total_detections=2, elapsed_seconds=0.5)
        detector = _FakeDetector([0])
        anonymizer = _FakeAnonymizer()

        with patch(
            "webapp.utils.processing.process_video", return_value=known_stats
        ) as mock_process:
            stats, h264_path = run_full(
                input_path, output_path, detector=detector, anonymizer=anonymizer
            )

        assert stats is known_stats
        assert h264_path == output_path.with_name("output_h264.mp4")
        assert h264_path.exists()
        assert h264_path.stat().st_size > 0
        mock_process.assert_called_once()
