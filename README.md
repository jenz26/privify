# Privify

> GDPR-compliant video anonymization through automatic face and license plate blurring.

Privify is a computer vision pipeline that automatically detects and blurs faces and 
vehicle license plates in video footage, enabling organizations to share, store, or 
publish CCTV and surveillance recordings while preserving the privacy of individuals 
captured in the scenes.

The project was developed as the final assignment for the Computer Vision course at 
Epicode Institute of Technology, with a focus on real-world applicability for small 
and medium-sized enterprises subject to GDPR requirements.

## Why this matters

Under the GDPR (Art. 4) and EDPB guidelines, both human faces and vehicle license 
plates qualify as personal data: they can identify individuals directly (faces) or 
indirectly via vehicle ownership records (plates). Any organization that retains, 
shares, or publishes video footage containing identifiable subjects must either 
obtain explicit consent or apply effective anonymization. Manual anonymization is 
labor-intensive and error-prone; Privify automates it.

Typical use cases:
- Retail and hospitality CCTV review
- Construction site documentation and drone footage
- Logistics and warehouse surveillance archives
- Internal training material derived from real recordings
- Insurance and legal evidence preparation

## Pipeline

```
Video input
    │
    ▼
[1] Frame extraction & preprocessing  (OpenCV)
    │
    ▼
[2] Object detection                  (YOLOv8, fine-tuned)
    │   ├── Face detector head
    │   └── License plate detector head
    ▼
[3] Multi-object tracking             (ByteTrack)
    │   (temporal coherence across frames)
    ▼
[4] Blur application                  (Gaussian, OpenCV)
    │
    ▼
Anonymized video output
```

### Stage details

1. **Data Acquisition & Preprocessing** — Video is decoded frame-by-frame via 
   OpenCV. Each frame is resized and normalized to YOLOv8's expected input format. 
   During training, standard Ultralytics augmentations are applied (HSV jitter, 
   horizontal flip, mosaic).

2. **Feature Engineering & Detection** — A YOLOv8 backbone, pretrained on COCO and 
   fine-tuned on WIDER FACE (faces) and CCPD (license plates), produces bounding 
   boxes with confidence scores. The CNN backbone learns visual features 
   automatically; no handcrafted descriptors are used.

3. **Tracking & Post-processing** — Detections are linked across consecutive frames 
   using ByteTrack to enforce temporal consistency. This eliminates flickering 
   (a frame where a face is missed creates a privacy leak) and stabilizes bounding 
   box positions. Non-Maximum Suppression is applied within YOLOv8 to remove 
   duplicate detections.

4. **Blur Application** — Detected regions are blurred using a Gaussian kernel sized 
   proportionally to the bounding box. Gaussian blur is preferred over pixelation 
   for visual naturalness and over face replacement for irreversibility 
   (a key compliance property).

## Repository structure

```
privify/
├── README.md
├── requirements.txt
├── notebooks/
│   └── demo_colab.ipynb        # End-to-end demo, runnable on Colab T4
├── src/
│   ├── detector.py             # YOLOv8 wrapper for faces + plates
│   ├── tracker.py              # ByteTrack integration
│   ├── anonymizer.py           # Blur application + main pipeline
│   └── evaluate.py             # mAP, IoU, privacy leakage rate
├── data/                       # Datasets (gitignored, see notebooks)
├── models/                     # Fine-tuned weights
├── samples/                    # Example input/output videos
└── docs/
    └── technical_analysis.pdf  # 10-page technical document
```

## Setup

### Option A — Google Colab (recommended for quick demo)

Open `notebooks/demo_colab.ipynb` in Colab, select a T4 GPU runtime, run all cells. 
The notebook downloads model weights, processes a sample video, and produces an 
anonymized output.

[*Add Colab badge link here once notebook is published*]

### Option B — Local environment

Requirements: Python 3.10+, ffmpeg installed system-wide.

```bash
git clone https://github.com/jenz26/privify.git
cd privify
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the pipeline on a video:

```bash
python -m src.anonymizer --input samples/input.mp4 --output samples/output.mp4
```

## Datasets

| Dataset | Usage | Source |
|---|---|---|
| WIDER FACE | Face detection fine-tuning | http://shuoyang1213.me/WIDERFACE/ |
| CCPD | License plate fine-tuning | https://github.com/detectRecog/CCPD |
| Italian plates (custom, ~50 samples) | Domain shift evaluation | Collected manually |

## Evaluation

The system is evaluated on three axes:

- **Detection quality** — mAP@0.5 and mAP@0.5:0.95 on held-out test sets, 
  per class (face / plate)
- **Privacy leakage rate** — fraction of frames in test videos where at least 
  one ground-truth subject (face or plate) is not blurred. This is the metric 
  that matters for compliance.
- **Throughput** — frames per second on Colab T4, used for deployment feasibility 
  discussion

Detailed results are reported in `docs/technical_analysis.pdf`.

## Results summary

*[To be filled in after experiments — tables and key numbers go here]*

## Limitations and failure modes

Documented in detail in `docs/technical_analysis.pdf`. Highlights:

- Performance degrades on faces smaller than ~24 pixels (typical for far-field 
  CCTV)
- Domain shift between Chinese (CCPD training) and European license plates 
  affects detection rate; mitigated partially by fine-tuning on a small 
  Italian test set
- Severe occlusion (e.g., faces partially covered by hands or hats) reduces 
  recall; not addressed in this version
- The Gaussian blur is non-reversible by design but does not protect against 
  re-identification through gait, clothing, or contextual cues

## Ethical considerations

Discussed in `docs/technical_analysis.pdf`. Key points:

- Face detectors are known to underperform on darker skin tones (Buolamwini & 
  Gebru, 2018). Per-subgroup performance is reported where dataset annotations 
  allow.
- Anonymization protects identity within the video but does not address upstream 
  questions about whether the recording was lawful.
- The system is intended as a privacy-enhancing tool, not as a workaround for 
  unlawful surveillance.

## License

[*To be decided — MIT or Apache-2.0 recommended*]

## Author

[*Marco — your details*]