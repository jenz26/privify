"""Privify — Streamlit web front-end.

Entry point for the GDPR video-anonymization demo. Run it with::

    streamlit run app.py

This module is pure UI orchestration: it wires Streamlit widgets to the
already-tested :mod:`webapp.utils` layer (video I/O, representative-frame
selection, detection, anonymization, H.264 transcoding) and holds no detection
or blur logic of its own.

Architectural invariant — **resolve once, reuse everywhere**:
the ``(detector, anonymizer)`` pair is resolved a single time per selected mode
via :func:`webapp.utils.processing.resolve_components` and cached in
``st.session_state``. Both the preview and the full run reuse those very same
instances. Because :class:`src.detector.Detector` loads its YOLO weights lazily
(on the first ``detect`` call, which happens during the preview), the model is
loaded exactly once per session: the full run reuses the already-loaded
instance and never reloads. The pair is re-resolved only when the user changes
mode.

Note:
    The user-facing copy is intentionally in Italian (the demo targets Italian
    SMEs). Code, comments and docstrings remain in English.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from webapp.utils.processing import (
    resolve_components,
    run_full,
    run_preview,
    select_representative_frames,
)
from webapp.utils.video import get_video_info

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------
LOGO_PATH = Path("assets/logo_epicode.png")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB soft limit, also stated in the UI

# Anonymization presets: mode key -> Italian label and description shown next to
# the selector. The keys must match the modes accepted by ``resolve_components``.
MODE_OPTIONS: dict[str, dict[str, str]] = {
    "face": {
        "label": "Volti",
        "description": "Anonimizza solo i volti riconosciuti (rapido, adatto a scene close-up)",
    },
    "person_upper": {
        "label": "Testa e spalle",
        "description": "Anonimizza testa e spalle delle persone (recall alto su CCTV)",
    },
    "person_full": {
        "label": "Persona intera",
        "description": "Anonimizza l'intera persona (massima privacy)",
    },
}

# session_state keys cleared by the "process another video" reset. The resolved
# components and mode are deliberately kept so the loaded model can be reused.
_RESETTABLE_KEYS = ("video_path", "uploaded_key", "video_info", "previews", "result")


# --- State and temp-file helpers ---------------------------------------------
def init_state() -> None:
    """Seed every ``session_state`` key so reruns never raise ``KeyError``."""
    st.session_state.setdefault("resolved_mode", None)
    st.session_state.setdefault("detector", None)
    st.session_state.setdefault("anonymizer", None)
    st.session_state.setdefault("video_path", None)
    st.session_state.setdefault("uploaded_key", None)
    st.session_state.setdefault("video_info", None)
    st.session_state.setdefault("previews", None)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("uploader_round", 0)


def get_work_dir() -> Path:
    """Return the per-session temp directory, creating it on first use.

    ``mkdtemp`` is called lazily here instead of through ``setdefault`` whose
    default argument is eagerly evaluated and would leak a fresh directory on
    every rerun.

    Returns:
        Path to the session's working directory for uploads and outputs.
    """
    work_dir = st.session_state.get("work_dir")
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="privify_webapp_")
        st.session_state["work_dir"] = work_dir
    return Path(work_dir)


def reset_video_state() -> None:
    """Clear all video-derived state and its on-disk temp files.

    Single source of truth for dropping everything derived from the *uploaded
    video* — the temp path, metadata, preview, result and the cached download
    bytes (all of :data:`_RESETTABLE_KEYS`) — plus the working directory on disk
    (best-effort: ``rmtree`` ignores a missing tree). The resolved
    ``(detector, anonymizer)`` pair and the selected mode are deliberately kept
    so the loaded YOLO model is reused for the next video rather than reloaded.
    """
    work_dir = st.session_state.get("work_dir")
    if work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    st.session_state.pop("work_dir", None)
    for key in _RESETTABLE_KEYS:
        st.session_state[key] = None


def reset_session() -> None:
    """Reset used by the "process another video" button.

    Clears the video-derived state via :func:`reset_video_state` and also bumps
    the uploader round so the ``file_uploader`` widget itself is emptied.
    """
    reset_video_state()
    st.session_state["uploader_round"] += 1


def resolve_for_mode(mode: str) -> None:
    """(Re)build and cache the components for *mode*, only when it changed.

    Implements the resolve-once invariant: a new ``(detector, anonymizer)``
    pair is built only when the selected mode differs from the cached one.
    ``resolve_components`` does not load the model (the detector is lazy), so
    this stays cheap; changing mode also invalidates any stale preview.

    Args:
        mode: One of the keys in :data:`MODE_OPTIONS`.
    """
    if st.session_state["resolved_mode"] == mode:
        return
    detector, anonymizer = resolve_components(mode, None, None)
    st.session_state["detector"] = detector
    st.session_state["anonymizer"] = anonymizer
    st.session_state["resolved_mode"] = mode
    # Derived outputs built with the previous mode are now stale: drop both the
    # preview and any finished result so the UI never serves a different mode's
    # anonymization than the one currently selected.
    st.session_state["previews"] = None
    st.session_state["result"] = None


# --- Page config (must precede every other Streamlit call) -------------------
st.set_page_config(
    page_title="Privify",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="centered",
)

init_state()

# ---- Header (branding) ----
logo_found = LOGO_PATH.exists()
if logo_found:
    st.image(str(LOGO_PATH), width=180)
    st.title("Privify")
else:
    # Fallback wordmark when the Epicode logo asset is missing.
    st.markdown(
        "<h1 style='color:#1C2541;letter-spacing:1px;margin-bottom:0.2rem'>Privify</h1>",
        unsafe_allow_html=True,
    )
st.markdown("Anonimizzazione GDPR-compliant di volti e persone in video CCTV.")
st.divider()

# ---- Sezione 1: Upload ----
st.subheader("1. Carica il video")
uploaded = st.file_uploader(
    "Carica un video",
    type=["mp4"],
    help="Formato MP4. Limite consigliato 100MB; per la demo usa clip brevi.",
    key=f"uploader_{st.session_state['uploader_round']}",
)

if uploaded is not None:
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.warning(
            f"Il file e' troppo grande ({uploaded.size / 1024 / 1024:.0f} MB). "
            "Per questa demo carica un video sotto i 100MB."
        )
        # Reject and do not proceed: drop any previously loaded video so the
        # downstream sections stop operating on the now-replaced file.
        reset_video_state()
    else:
        # Persist the upload only when it actually changes (file_uploader returns
        # the same object across reruns until the user picks a new file).
        file_key = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
        if st.session_state["uploaded_key"] != file_key:
            video_path = get_work_dir() / "input.mp4"
            video_path.write_bytes(uploaded.getbuffer())
            st.session_state["video_path"] = str(video_path)
            st.session_state["uploaded_key"] = file_key
            st.session_state["video_info"] = None
            st.session_state["previews"] = None
            st.session_state["result"] = None
elif st.session_state["video_path"] is not None:
    # The uploader was cleared (the user removed the file) but video state from a
    # previous upload still lives in session_state. Reset it once, on this
    # transition, so sections 2-4 stop rendering and no orphan metric is left on
    # screen. The guard makes this a no-op on every later empty-uploader rerun.
    reset_video_state()

# Read and display the basic metadata of the persisted video.
video_path = st.session_state["video_path"]
if video_path:
    if st.session_state["video_info"] is None:
        try:
            st.session_state["video_info"] = get_video_info(Path(video_path))
        except Exception:
            logger.exception("Failed to read video metadata")
            st.error("Impossibile leggere il video caricato. Riprova con un altro file.")
            st.session_state["video_path"] = None

    info = st.session_state["video_info"]
    if info is not None:
        col_dur, col_res, col_fps, col_frames = st.columns(4)
        col_dur.metric("Durata", f"{info.duration_seconds:.1f} s")
        col_res.metric("Risoluzione", f"{info.width}×{info.height}")
        col_fps.metric("FPS", f"{info.fps:.1f}")
        col_frames.metric("Frame", f"{info.frame_count}")

# ---- Sezioni 2-4: disponibili solo con un video caricato ----
video_path = st.session_state["video_path"]
if video_path:
    # ---- Sezione 2: Settings (mode) ----
    st.subheader("2. Modalita' di anonimizzazione")
    mode = st.radio(
        "Cosa anonimizzare",
        options=list(MODE_OPTIONS.keys()),
        format_func=lambda key: MODE_OPTIONS[key]["label"],
        horizontal=True,
    )
    st.caption(MODE_OPTIONS[mode]["description"])

    # Resolve-once invariant: (re)build the components only on a mode change.
    resolve_for_mode(mode)

    # ---- Sezione 3: Anteprima before/after ----
    st.subheader("3. Anteprima")
    if st.button("Genera anteprima"):
        try:
            with st.spinner("Generazione anteprima... (il primo avvio carica il modello)"):
                # The first detect() inside select_representative_frames loads the
                # YOLO weights into the cached detector; the spinner covers the wait.
                selected = select_representative_frames(
                    Path(video_path), st.session_state["detector"]
                )
                frames = [frame for _, frame in selected]
                st.session_state["previews"] = run_preview(
                    frames,
                    st.session_state["detector"],
                    st.session_state["anonymizer"],
                )
        except Exception:
            logger.exception("Preview generation failed")
            st.error("Generazione dell'anteprima non riuscita. Controlla il video e riprova.")

    previews = st.session_state["previews"]
    if previews:
        # Convert BGR (OpenCV) to RGB (expected by st.image) for each pair.
        for before, after in previews:
            col_before, col_after = st.columns(2)
            col_before.image(
                cv2.cvtColor(before, cv2.COLOR_BGR2RGB),
                caption="Originale",
                use_container_width=True,
            )
            col_after.image(
                cv2.cvtColor(after, cv2.COLOR_BGR2RGB),
                caption="Anonimizzato",
                use_container_width=True,
            )

    # ---- Sezione 4: Elaborazione + Risultati ----
    st.subheader("4. Elaborazione")
    if st.button("Anonimizza video", type="primary"):
        try:
            output_path = get_work_dir() / "output.mp4"
            bar = st.progress(0, text="Elaborazione...")

            def _on_progress(current: int, total: int) -> None:
                frac = min(current / total, 1.0) if total > 0 else 0.0
                bar.progress(frac, text=f"Frame {current} di {total}")

            # Reuse the cached components (no resolve here): the model is already
            # loaded from the preview, so the full run never reloads it.
            stats, h264_path = run_full(
                Path(video_path),
                output_path,
                detector=st.session_state["detector"],
                anonymizer=st.session_state["anonymizer"],
                on_progress=_on_progress,
            )
            bar.progress(1.0, text="Completato")
            # Read the output bytes once, now: Streamlit reruns the whole script
            # on every interaction, so reading them in the render block below
            # would reload the full file from disk on each unrelated rerun.
            download_bytes = h264_path.read_bytes() if h264_path.exists() else None
            st.session_state["result"] = {
                "stats": stats,
                "h264_path": str(h264_path),
                "download_bytes": download_bytes,
            }
        except Exception:
            logger.exception("Full anonymization run failed")
            st.error(
                "Elaborazione non riuscita. Riprova; se il problema persiste usa "
                "un video piu' breve."
            )

    result = st.session_state["result"]
    if result:
        stats = result["stats"]
        h264_path = Path(result["h264_path"])
        download_bytes = result.get("download_bytes")
        st.success("Elaborazione completata")
        if h264_path.exists():
            # The H.264 file is browser-playable inline (mp4v is not reliably).
            st.video(str(h264_path))

        col_tf, col_td, col_fps, col_time = st.columns(4)
        col_tf.metric("Frame totali", f"{stats.total_frames}")
        col_td.metric("Rilevazioni totali", f"{stats.total_detections}")
        col_fps.metric("FPS elaborati", f"{stats.fps_processed:.1f}")
        col_time.metric("Tempo", f"{stats.elapsed_seconds:.1f} s")

        if download_bytes is not None:
            st.download_button(
                "Scarica il video anonimizzato",
                data=download_bytes,
                file_name="privify_anonymized.mp4",
                mime="video/mp4",
            )

        if st.button("Elabora un altro video"):
            reset_session()
            st.rerun()
else:
    st.info("Carica un video MP4 qui sopra per iniziare.")

# ---- Disclaimer + footer ----
st.divider()
st.caption(
    "Demo dimostrativa. Elaborazione su infrastruttura condivisa: per video oltre "
    "i 30-60 secondi i tempi possono allungarsi. Limite consigliato 100MB."
)
st.markdown(
    "<p style='color:#8893A8;font-size:0.8rem'>"
    "Privify — Open-source CCTV anonymization · github.com/jenz26/privify</p>",
    unsafe_allow_html=True,
)
