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
