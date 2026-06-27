"""Detection + anonymization orchestration layer for the web front-end.

This module is a thin *façade* over the already-tested core
(:mod:`src.pipeline`, :mod:`src.detector`, :mod:`src.anonymizer`): it adds no
detection or blur logic of its own, it only sequences the core for two
front-end use cases — a fast preview and a full run.

Architectural invariant — **resolve once, reuse everywhere**:
the ``(detector, anonymizer)`` pair is resolved a single time upstream (in the
Streamlit layer) via :func:`src.pipeline.resolve_components` and is then passed
unchanged to both :func:`run_preview` and :func:`run_full`. The mode→components
mapping lives *only* in :func:`resolve_components`; it is deliberately never
reconstructed here. As a convenience, that factory is re-exported from this
module so the UI layer has a single orchestration import surface.

Because the preview and the full run share the very same detector/anonymizer
objects and call them through the same ``detect`` → ``anonymize`` sequence as
:func:`src.pipeline.process_video`, the preview is pixel-faithful to the final
output, up to codec compression only.

Note:
    :class:`src.detector.Detector` loads its YOLO weights lazily (on the first
    :meth:`~src.detector.Detector.detect` call, not at construction), so
    resolving the components does **not** load the model. The first ``detect``
    call — issued by :func:`select_representative_frames` — is what pays the
    one-time model-loading cost for the preview.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.anonymizer import Anonymizer
from src.detector import Detector
from src.pipeline import ProcessingStats, process_video, resolve_components
from webapp.utils.video import equispaced_indices, extract_frames, get_video_info

__all__ = [
    "resolve_components",
    "select_representative_frames",
    "run_preview",
    "run_full",
    "transcode_h264",
]


def select_representative_frames(
    path: Path,
    detector: Detector,
    *,
    n_preview: int = 3,
    n_probe: int = 10,
) -> list[tuple[int, np.ndarray]]:
    """Pick the most detection-rich frames of a video for a UI preview.

    The selection probes ``n_probe`` frames spread evenly across the video,
    counts the detections on each, and keeps the ``n_preview`` frames with the
    most detections — so the preview shows meaningful scenes (people / plates
    actually present) instead of arbitrary or near-empty frames.

    This is the **only** point in the preview path that invokes the model:
    ``detector.detect`` is called once per probed frame. Since the detector
    loads YOLO lazily, this is also where the one-time model-loading cost is
    paid for the preview.

    Args:
        path: Path to the source video file.
        detector: A resolved :class:`~src.detector.Detector`. Called once per
            probed frame; not mutated.
        n_preview: Number of frames to keep for the preview (keyword-only).
        n_probe: Number of evenly spaced frames to probe (keyword-only). Should
            be ``>= n_preview`` for a meaningful selection.

    Returns:
        A list of ``(original_index, frame_bgr)`` pairs, sorted by original
        frame index ascending so the preview scrolls in temporal order. At most
        ``n_preview`` pairs are returned (fewer only if the video yields fewer
        decodable probe frames).

    Note:
        When **every** probed frame has zero detections, the function falls
        back to the first ``n_preview`` probed frames (by index), so the
        preview is never empty.
    """
    info = get_video_info(path)
    probe_indices = equispaced_indices(info.frame_count, n_probe)
    frames = extract_frames(path, probe_indices)

    # ``extract_frames`` returns frames in index order, skipping any that fail
    # to decode; zip aligns each decoded frame with its index (best-effort).
    paired = list(zip(probe_indices, frames, strict=False))

    # The single model-invocation point of the preview path: one detect per
    # probed frame. We only need the detection *count* to rank frames.
    scored = [(index, frame, len(detector.detect(frame))) for index, frame in paired]

    # Fallback: nothing was detected anywhere → keep the first n_preview frames
    # by index so the preview is never empty. ``paired`` is already ascending.
    if all(count == 0 for _, _, count in scored):
        return [(index, frame) for index, frame in paired[:n_preview]]

    # Rank by detection count descending; break ties by index ascending
    # (stable, deterministic), then keep the top n_preview.
    ranked = sorted(scored, key=lambda item: (-item[2], item[0]))
    top = ranked[:n_preview]

    # Re-sort the kept frames by index so the preview is in temporal order.
    top.sort(key=lambda item: item[0])
    return [(index, frame) for index, frame, _ in top]


def run_preview(
    frames: list[np.ndarray],
    detector: Detector,
    anonymizer: Anonymizer,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Anonymize a handful of frames for a before/after preview.

    Each frame goes through the exact same two calls as the
    :func:`src.pipeline.process_video` loop — ``detector.detect`` followed by
    ``anonymizer.anonymize`` — which is what guarantees the preview is
    pixel-faithful to the full run (the only difference at full scale is codec
    compression of the written video).

    Args:
        frames: BGR frames to anonymize, as ``(H, W, 3)`` arrays.
        detector: The resolved detector (same object used by the full run).
        anonymizer: The resolved anonymizer (same object used by the full run).

    Returns:
        One ``(before, after)`` pair per input frame, in input order, where
        *before* is the original frame and *after* is the anonymized result.
    """
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in frames:
        detections = detector.detect(frame)
        after = anonymizer.anonymize(frame, detections)
        pairs.append((frame, after))
    return pairs


def run_full(
    input_path: Path,
    output_path: Path,
    *,
    detector: Detector,
    anonymizer: Anonymizer,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[ProcessingStats, Path]:
    """Anonymize a full video, then transcode it for in-browser playback.

    Delegates the frame-by-frame work to :func:`src.pipeline.process_video`
    using the injected ``(detector, anonymizer)`` pair (``mode=None`` path),
    then transcodes the ``mp4v`` output produced by OpenCV to browser-playable
    H.264 in a sibling file (``<name>_h264.mp4``).

    Args:
        input_path: Path to the source video file.
        output_path: Path for the intermediate ``mp4v`` output written by the
            pipeline. Its parent directory must already exist.
        detector: The resolved detector (keyword-only).
        anonymizer: The resolved anonymizer (keyword-only).
        on_progress: Optional ``(current, total)`` per-frame callback forwarded
            to :func:`src.pipeline.process_video` (keyword-only).

    Returns:
        A ``(stats, h264_path)`` tuple, where *stats* is the
        :class:`~src.pipeline.ProcessingStats` of the run and *h264_path* is
        the path to the browser-playable H.264 file.
    """
    stats = process_video(
        input_path,
        output_path,
        detector=detector,
        anonymizer=anonymizer,
        on_progress=on_progress,
    )
    h264_path = output_path.with_name(f"{output_path.stem}_h264.mp4")
    transcode_h264(output_path, h264_path)
    return stats, h264_path


def transcode_h264(src: Path, dst: Path) -> Path:
    """Re-encode an ``mp4v`` video to H.264 for in-browser playback.

    OpenCV's :class:`cv2.VideoWriter` writes ``mp4v`` (MPEG-4 Part 2), which is
    universally available but not reliably playable by HTML5 ``<video>`` (and
    thus by Streamlit's ``st.video``). This re-encodes to H.264 with the
    ``yuv420p`` pixel format and the ``+faststart`` flag, both required for
    broad browser compatibility and progressive streaming (moov atom moved to
    the front of the file).

    The ``ffmpeg`` binary is provided by ``imageio-ffmpeg`` (a self-contained
    static build), so no system-wide ``ffmpeg`` install is required.

    Args:
        src: Path to the source ``mp4v`` video.
        dst: Path where the H.264 video will be written.

    Returns:
        The destination path *dst*.

    Raises:
        RuntimeError: If the ``ffmpeg`` invocation exits with a non-zero status;
            the tail of its stderr is included in the message.
    """
    from imageio_ffmpeg import get_ffmpeg_exe

    ffmpeg = get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip()[-2000:]
        raise RuntimeError(
            f"ffmpeg transcode {src} → {dst} failed (exit {result.returncode}):\n{stderr_tail}"
        )

    return dst
