# Privify

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenz26/privify/blob/main/notebooks/demo_colab.ipynb)

**Live demo:** [https://privify.streamlit.app](https://privify.streamlit.app)

> GDPR-compliant anonymization of people and faces in CCTV video.

Privify detects people and faces in video surveillance footage and blurs them, so
that small and medium-sized Italian enterprises can retain, share, or publish CCTV
recordings while staying compliant with the GDPR.

## Overview

Privify was developed as the final project for the Computer Vision course at Epicode
Institute of Technology. It ships as a dual deliverable: a reusable detection-and-blur
pipeline (the `src` package) and a deployed Streamlit web application that runs the
pipeline directly from the browser.

The goal is real-world applicability for organizations that handle surveillance
footage and need a fast, repeatable way to anonymize it before it leaves the premises.

## Why this matters

Under the GDPR (Art. 4) and the guidelines of the European Data Protection Board, a
human face is personal data: it identifies an individual directly. Any organization
that retains, shares, or publishes video footage of identifiable people must either
obtain explicit consent or apply effective anonymization. Doing this by hand is slow
and error-prone, and a single missed frame is already a privacy leak. Privify automates
the process.

Typical use cases for small and medium-sized enterprises:

- Retail and hospitality CCTV review
- Logistics and warehouse surveillance archives
- Office and workplace footage shared for documentation or training

## Anonymization modes

Privify offers three selectable modes. They exist because no single detector covers
every CCTV scenario equally well.

- `face`: blurs only the detected faces, using the fine-tuned face detector. Fast, and
  well suited to close-up scenes where faces are large and roughly frontal.
- `person_upper`: blurs the head-and-shoulders region (the upper 30% of each detected
  person box), using a COCO person detector. Higher recall than face detection on
  street-level CCTV.
- `person_full`: blurs the entire person box, using a COCO person detector. Maximum
  privacy, at the cost of obscuring more of the frame.

The rationale behind these modes is the core design decision of the project. A
dedicated face detector has limited recall on street-level CCTV, where faces are small,
turned away, or partially occluded. The `person_*` modes sidestep this by detecting the
person instead of the face: a person is a much larger and more reliably detected
target, so recall is substantially higher. `person_upper` is a middle ground that still
anonymizes the identifying region (the head) while leaving most of the body visible,
and `person_full` trades visibility for the strongest privacy guarantee. The Results
section below quantifies why this matters.

## Pipeline

```
Video input
    │
    ▼
[1] Frame extraction          (OpenCV, frame by frame)
    │
    ▼
[2] Detection                 (YOLOv8: fine-tuned face model
    │                          or COCO person model, per mode)
    ▼
[3] Gaussian blur             (irreversible, applied to each
    │                          detected region)
    ▼
Anonymized video output
```

Privify decodes the input video frame by frame with OpenCV, runs detection on each
frame with a YOLOv8 model (the fine-tuned face model for `face` mode, or a COCO person
model for the `person_*` modes), and blurs every detected region with a Gaussian kernel
sized proportionally to the box. The blurred frames are re-encoded into the output
video. The pipeline is stateless and per-frame: there is no temporal tracking stage.
Gaussian blur is chosen over pixelation for visual naturalness, and over face
replacement because it is irreversible, which is the property that matters for
compliance.

## Web application

The Streamlit application (`app.py`) wraps the pipeline in a browser interface:

1. Upload a video.
2. Select an anonymization mode.
3. Preview the effect: Privify picks a few representative frames and shows them before
   and after blurring, so the result can be checked before committing.
4. Run the full anonymization, with a progress bar that reports frame-by-frame
   progress.
5. Download the anonymized video.

The output is transcoded to H.264 (via the ffmpeg binary bundled with `imageio-ffmpeg`)
so that it plays back directly in the browser, since the codec OpenCV writes by default
is not reliably playable in HTML5. The app is deployed on Streamlit Community Cloud at
[https://privify.streamlit.app](https://privify.streamlit.app).

Run it locally with:

```bash
streamlit run app.py
```

## Results

The empirical story of this project is about domain shift, the gap between the training
data and the real deployment target (street-level CCTV).

- The first face detector (`face-detector-v0.1`) reached mAP@50 = 0.937
  in-distribution, but only 17.2% recall on the real CCTV target. A generic COCO person
  detector reached 80.4% recall on the same footage: a structural gap of roughly 63
  percentage points, driven entirely by domain shift.
- Re-training on a WIDER FACE subset biased toward small faces (`face-detector-v0.2`)
  raised target recall to 59.5%, an improvement of +42.4 percentage points over v0.1,
  with all hyperparameters held constant (a single-variable comparison).
- A residual gap of -20.9 percentage points remained against the COCO person detector
  (59.5% vs 80.4%). This is an intrinsic limit of face detection on CCTV, and it is
  exactly what motivates the `person_*` modes described above.

Validation metrics for the current face model (`face-detector-v0.2`), measured on a
WIDER FACE validation subset (300 images, 6,559 face instances):

| Metric | Value |
|---|---|
| Precision | 0.811 |
| Recall | 0.527 |
| mAP@50 | 0.612 |
| mAP@50-95 | 0.321 |

The full out-of-distribution evaluation is reported in the Technical Analysis Document.

## Repository structure

```
privify/
├── app.py                      # Streamlit web app entry point
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development and CI dependencies
├── packages.txt                # System libraries for Streamlit Cloud
├── pyproject.toml              # Project metadata and ruff config
├── Makefile                    # Common development tasks
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── src/
│   ├── __init__.py
│   ├── detector.py             # YOLOv8 detection wrapper (face and COCO person)
│   ├── anonymizer.py           # Gaussian blur on detected regions
│   ├── pipeline.py             # Mode resolution and video processing
│   └── training.py             # Face detector fine-tuning entry point
├── webapp/
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── video.py            # Frame extraction and video helpers
│       └── processing.py       # Preview, full run, H.264 transcode
├── tests/
│   ├── test_detector.py
│   ├── test_anonymizer.py
│   ├── test_pipeline.py
│   ├── test_training.py
│   ├── test_smoke.py
│   ├── test_prepare_wider_face.py
│   ├── test_webapp_video.py
│   └── test_webapp_processing.py
├── notebooks/
│   ├── demo_colab.ipynb        # End-to-end demo, runnable on Colab T4
│   ├── finetune_face.ipynb     # Face detector fine-tuning
│   └── ood_evaluation.ipynb    # Out-of-distribution evaluation
├── tools/
│   └── prepare_wider_face.py   # WIDER FACE dataset preparation
└── docs/
    ├── technical_analysis.pdf           # Technical Analysis Document (extended)
    ├── technical_analysis.qmd           # Quarto source for the extended version
    ├── technical_analysis_compact.pdf   # Compact 10-page version (exam submission)
    ├── technical_analysis_compact.qmd   # Quarto source for the compact version
    ├── _compact/                        # Sober paper template + figure-width filter for the compact
    ├── decisions.md                     # Architectural Decision Records
    └── typst-template.typ               # Typst template for the PDF
```

## Setup

Privify requires Python 3.10 or newer (the live deployment runs on 3.13).

### Option A: Google Colab (quickest demo)

Open `notebooks/demo_colab.ipynb` in Colab, select a T4 GPU runtime, and run all cells.
The notebook downloads the model weights, processes a sample video, and produces an
anonymized output. Use the badge at the top of this README, or open it
[here](https://colab.research.google.com/github/jenz26/privify/blob/main/notebooks/demo_colab.ipynb).

### Option B: Local environment

```bash
git clone https://github.com/jenz26/privify.git
cd privify
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Launch the web app:

```bash
streamlit run app.py
```

Or use the pipeline programmatically:

```python
from pathlib import Path

from src.pipeline import process_video

process_video(
    Path("input.mp4"),
    Path("output.mp4"),
    mode="person_upper",
)
```

> **Model weights:** on its first detection, the `face` mode automatically downloads the
> fine-tuned `face-detector-v0.2` weights (~6 MB) from the project's GitHub Release,
> verifies them with a SHA-256 checksum, and caches them under `models/`. The `person_*`
> modes use the COCO `yolov8n` weights, which ultralytics downloads from its own
> registry on first use. An internet connection is needed only once per environment, and
> no system-wide ffmpeg is required: the ffmpeg binary is bundled via `imageio-ffmpeg`.

## Limitations and failure modes

Discussed in detail in the Technical Analysis Document. Highlights:

- Face detection recall drops on the small, turned-away, and partially occluded faces
  that are typical of street-level CCTV. This is the limitation that the `person_upper`
  and `person_full` modes are designed to mitigate.
- Gaussian blur is irreversible by design, but it does not protect against
  re-identification through gait, clothing, or contextual cues.

## Ethical considerations

Discussed in the Technical Analysis Document. Key points:

- Face detectors are known to underperform on darker skin tones (Buolamwini & Gebru,
  2018). This bias is acknowledged and reported where dataset annotations allow.
- Anonymization protects identity within the video, but it does not answer the upstream
  question of whether the recording was lawful in the first place.
- Privify is intended as a privacy-enhancing tool, not as a workaround for unlawful
  surveillance.

## Documentation

- [Technical Analysis Document (compact, 10 pages)](docs/technical_analysis_compact.pdf):
  the exam submission version, in academic paper format, covering problem statement,
  methodology, experimental results, failure analysis, and ethical considerations. This
  is the reference version for evaluation.
- [Technical Analysis Document (extended, ~18 pages)](docs/technical_analysis.pdf): an
  extended version with the same analysis in greater depth and full formatting, for
  readers who want the complete treatment.
- [Architectural Decision Records](docs/decisions.md): a versioned log of the design
  decisions made during development.

## License

This project is licensed under AGPL-3.0. Privify builds on Ultralytics YOLOv8, which is
distributed under AGPL-3.0; as a copyleft license it requires derivative works to adopt
the same terms. The Affero clause also covers network use, which is appropriate since
Privify is deployed as a web application. The redistributed fine-tuned weights inherit
AGPL-3.0 from the base model.

## Author

Marco Contin  
Epicode Institute of Technology, student ID s00006824  
Computer Vision final project
