# Architecture Decision Records

This document tracks non-obvious technical decisions made during the development
of Privify. Each entry follows a lightweight ADR format.

## 2026-05-07 — Initial stack selection

**Context:** Privify is a final exam project for a Computer Vision course. It must
run on Google Colab with a T4 GPU, detect faces and license plates in video, and
blur them for GDPR compliance. The scope is fixed and the codebase must be clear
enough to explain in an oral examination.

**Decision:** Use ultralytics YOLOv8 as the unified detector for both faces and
license plates, ByteTrack (built into ultralytics) for multi-object tracking, and
OpenCV for video I/O and blur operations.

**Alternatives considered:**
- *Separate detectors (MTCNN/RetinaFace for faces + YOLO for plates):* Rejected
  because maintaining two detection pipelines doubles complexity with no clear
  accuracy gain for this use case. A single YOLOv8 model fine-tuned on both
  classes keeps the architecture simple and explainable.
- *DeepSORT instead of ByteTrack:* Rejected because ByteTrack is already
  integrated into the ultralytics tracking API, requires no additional
  dependencies, and performs comparably on the metrics we care about (consistent
  identity assignment across frames).

## 2026-05-07 — COCO classes as placeholder until fine-tuning

**Context:** The detection module (`src/detector.py`) needs to work immediately
for integration testing and pipeline development, but the fine-tuned model on
WIDER FACE + CCPD is not available yet.

**Decision:** Use `yolov8n.pt` pre-trained on COCO as a provisional backbone.
COCO class 0 ("person") serves as a temporary proxy for "face" detection.
License-plate detection is deferred until the fine-tuned model is ready.  The
`Detector` wrapper exposes a stable API (`Detection` dataclass with `class_id`
and `class_name`) so downstream modules (tracker, anonymizer) will not need
changes when the model is swapped.

**Alternatives considered:**
- *Wait for fine-tuning before writing any detection code:* Rejected because
  the wrapper, tests, and pipeline integration can proceed independently of the
  model weights.  Delaying would serialize work unnecessarily.
- *Use a face-specific detector (RetinaFace/MTCNN) as interim:* Rejected because
  it introduces a dependency that would be thrown away after fine-tuning.  Using
  the same ultralytics interface now avoids a throwaway integration effort.

## 2026-05-07 — Gaussian blur as anonymization strategy

**Context:** The anonymizer module needs a method to redact faces and license
plates in video frames for GDPR compliance.  The three main options are
Gaussian blur, pixelation (mosaic), and face/plate replacement with synthetic
imagery.

**Decision:** Use `cv2.GaussianBlur` with a configurable kernel size (default
51).  The kernel size is validated to be a positive odd integer at construction
time.

**Alternatives considered:**
- *Pixelation (mosaic):* Rejected because recent super-resolution research
  (e.g. PULSE, GFPGAN) has shown that pixelated faces can be partially
  reconstructed, making pixelation a weaker privacy guarantee.  Gaussian blur
  destroys high-frequency information more thoroughly.
- *Face/plate replacement with generative models (GAN inpainting, diffusion):*
  Rejected because it introduces a dependency on a generative model, increases
  computational cost significantly, and adds complexity far beyond the project
  scope.  The visual result would be more natural, but GDPR compliance does not
  require naturalness — only effective de-identification.

## 2026-05-13 — Video processing as function with dependency injection

**Context:** The project needs an orchestrator that ties together `Detector`
and `Anonymizer` to process a full video end-to-end (read → detect →
anonymize → write).  This is the first module that performs real video I/O.

**Decision:** Implement a plain function `process_video` that receives
`Detector` and `Anonymizer` as arguments (dependency injection) rather than
instantiating them internally.  The function returns a frozen `ProcessingStats`
dataclass.  No class wrapper is introduced at this stage.

**Alternatives considered:**
- *Class `VideoProcessor` with constructor-injected collaborators:* Deferred.
  A class becomes worthwhile when the pipeline needs shared state across
  calls (progress callbacks, batching, multi-threaded frame reading).
  Currently a single function is simpler to test and compose.
- *Shell script + ffmpeg for frame extraction and re-muxing:* Rejected
  because it is not portable to Google Colab (ffmpeg availability and codec
  licensing vary), harder to unit-test, and introduces a process boundary
  that complicates error handling.
