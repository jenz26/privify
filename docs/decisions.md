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

## 2026-05-13 — MVP demo notebook with COCO placeholder detection

**Context:** The pipeline modules (detector, anonymizer, process_video) are
functional and tested.  An end-to-end demo runnable by anyone on Google Colab
is needed to validate the full workflow before investing time in fine-tuning.

**Decision:** Ship a Colab notebook (`notebooks/demo_colab.ipynb`) that uses
`yolov8n.pt` pre-trained on COCO with class "person" as a temporary proxy for
face detection.  License plate detection is deferred to the fine-tuning
milestone.  The notebook is explicit about its MVP status in both the title
cell and the "Next steps" section.

**Alternatives considered:**
- *Wait for the fine-tuned model before publishing any demo:* Rejected because
  it blocks demonstrability of the pipeline architecture.  The notebook proves
  that detector → anonymizer → video writer compose correctly; swapping the
  model weights later requires no notebook changes.

## 2026-05-13 — Bundle test video in repository instead of runtime download

**Context:** The demo notebook originally downloaded a test video from a
public URL at runtime.  This approach proved unreliable from datacenter
environments like Google Colab: Pixabay's CDN applies bot-mitigation rules
that return HTTP 403 to requests with non-browser User-Agent strings or
originating from cloud IP ranges.

**Decision:** Commit a small test video (~8 MB, Creative Commons from
Pixabay, urban scene with frontal pedestrians) as `samples/input.mp4`
directly in the repository.  The notebook verifies the file exists instead
of downloading it.  The `.gitignore` uses a negation rule
(`!samples/input.mp4`) to track this specific file while still ignoring
other local videos.

**Alternatives considered:**
- *GitHub-hosted sample videos (e.g. intel-iot-devkit/sample-videos):*
  Rejected because available clips showed pedestrians from behind, making
  them unsuitable for demonstrating facial blur visually.
- *Hosting the video on Google Drive with gdown:* Rejected because it ties
  notebook reproducibility to the lifecycle of a personal Google account.

## 2026-05-13 — Fine-tuning strategy: WIDER FACE subset + YOLOv8n + GitHub Releases

**Context:** The detector pre-trained on COCO uses "person" (class 0) as a
temporary proxy for "face".  To anonymize only faces — not full-body
silhouettes — a specialised detector is needed.

**Decision:** Fine-tune `yolov8n.pt` on a subset of WIDER FACE (~5 000
images, ~50 epochs).  The dataset is sourced from Roboflow Universe, already
in YOLOv8 format, avoiding manual conversion work.  Trained weights are
distributed as a GitHub Release asset (not committed to the repository).
Training logic lives in a reusable module (`src/training.py`) and is
launched from a dedicated Colab notebook (`notebooks/finetune_face.ipynb`).

**Alternatives considered:**
- *YOLOv8s / m / l:* Rejected for consistency with the existing pipeline
  (nano variant targets edge deployment) and to keep training times short on
  a free Colab T4 GPU.
- *Full WIDER FACE (~32 000 images):* Rejected to allow rapid iteration.
  Scaling to the full dataset is a linear change (update the data source)
  and is documented as Future Work.
- *Manual conversion of WIDER FACE from its original annotation format:*
  Rejected in favour of Roboflow Universe pre-converted datasets, to avoid
  spending time on data-engineering work outside the project scope.
- *Committing weights to the repository:* Rejected to avoid bloating the
  repository with binary files.  GitHub Releases are the standard pattern
  for distributing model artefacts.

## 2026-05-13 — Migrate dataset from v0.1 to v0.2 (proper validation split)

**Context:** The Roboflow-exported dataset v0.1 was configured with a
train/test split only (no validation set).  This would have forced reuse
of the test set as validation during training, introducing data leakage
between hyperparameter selection (driven by validation metrics) and final
evaluation — a methodologically weak setup.

**Decision:** Re-export the dataset from the personal Roboflow workspace
with an 80/10/10 train/valid/test split.  Published as GitHub Release
`dataset-v0.2`.  The old `dataset-v0.1` release is kept for archival
purposes.

**Consequences:**
- *Positive:* Independent validation set means `best.pt` is selected
  without data leakage; test set is held out from the tuning loop,
  producing honest performance reporting.
- *Positive:* Ghost class `i` (present as a label in the upstream
  workspace but with zero annotations) removed during re-export.
- *Positive:* Images standardised to 640×640 (resize-stretch) at the
  dataset level, reducing the Colab download by ~30 MB and slightly
  speeding up training.
- *Negative:* Training set slightly smaller (1 246 vs 1 402 images in
  v0.1); difference is negligible for fine-tuning.
- *Negative:* One-shot update of URLs and paths in the notebook.

## 2026-05-13 — Use fine-tuned YOLOv8n weights instead of COCO yolov8n.pt

**Context:** The pre-trained `yolov8n.pt` detects COCO class "person" with
bounding boxes covering the entire body.  For GDPR anonymization the blur
must be limited to faces, not full-body silhouettes.  A specialised detector
is also a better starting point for future domain-specific fine-tuning
(CCTV, retail, etc.) than the generic COCO model.

**Decision:** Use the fine-tuned `face-detector-v0.1` weights (YOLOv8n,
50 epochs on dataset-v0.2; mAP@50 0.937, mAP@50-95 0.682).  Weights are
downloaded automatically from the GitHub Release on first instantiation
and cached in `models/` (gitignored).  Integrity is verified via SHA-256
before loading.

**Consequences:**
- *Positive:* Bounding boxes now cover only the face, significantly
  reducing over-blurring (less false-positive area).
- *Positive:* SHA-256 check protects against corrupted downloads or
  modified upstream assets.
- *Positive:* Clear versioning pattern — upgrading to `face-detector-v0.2`
  requires updating only two constants (``WEIGHTS_URL`` and
  ``WEIGHTS_SHA256``).
- *Negative:* Internet connection required on first run (mitigated by
  local caching; download is ~6 MB, one-time per environment).
- *Negative:* Users behind strict firewalls must pre-populate `models/`
  manually.

## 2026-06-25 — Out-of-distribution evaluation via count-based recall

**Context:** The fine-tuned `face-detector-v0.1` scores well on the WIDER
FACE test split (mAP@50 0.937) but qualitatively misses subjects on the real
CCTV deployment clip (`samples/input.mp4`), where many faces are non-frontal,
small, or occluded. An empirical, reproducible measure of this domain shift
is needed for the TAD (sections 3 *Experimental Results* and 4 *Failure
Analysis*).

**Decision:** Add `notebooks/ood_evaluation.ipynb`, which samples 10 frames
at uniform, deterministic spacing and scores two detectors against a manual
per-frame **person count** (the GDPR anonymization target): our fine-tuned
face detector vs. the stock `yolov8n.pt` COCO model filtered to class
`person`. The metric is a **count-based recall**, `min(detections, GT) / GT`,
aggregated as mean/std/min/max. Ground truth lives in versioned
`evaluation/ground_truth.json`; comparison figures in `evaluation/figures/`;
extracted frames are gitignored as regenerable artifacts.

To keep the versioned ground truth portable across environments, annotations
are keyed by **sample ordinal** (`0 … N-1`), not by the absolute frame index:
`cv2.CAP_PROP_FRAME_COUNT` is unreliable and can differ between OpenCV/codec
builds (local vs Colab), so the absolute index is stored only for reference
and a drift triggers a warning, not a hard failure. Frame counting falls back
to a full decode when the reported count is non-positive, and frame reads fall
back to sequential decoding when seeking is unsupported.

**Alternatives considered:**
- *IoU-matched recall with per-person bounding boxes:* Rejected for this
  milestone — it requires box-level manual annotation of every subject plus a
  matching scheme, disproportionate effort for a 10-frame qualitative-to-
  quantitative bridge. Count-based recall captures coverage at far lower
  annotation cost; box-level evaluation remains Future Work.
- *Uncapped `detections / GT` ratio:* Rejected because false positives or
  multiple detections per subject can push the value above 1, which is not
  interpretable as recall. Clipping with `min(·)` keeps the metric in
  `[0, 1]`; raw counts are retained in the table for transparency.
- *Curated frame selection:* Rejected to avoid cherry-picking; uniform
  deterministic sampling is reproducible and defensible at the oral exam.
- *Re-using the `Detector` wrapper for the COCO baseline:* Not possible by
  design — the wrapper auto-fetches our fine-tuned face weights; the baseline
  loads `yolov8n.pt` directly via ultralytics.

## 2026-06-26 — Hybrid detection strategy: face vs person modes

**Status:** Accepted

**Context:** The OOD evaluation (commit `999a4ee`) quantified empirically the
domain shift of `face-detector-v0.1` on the CCTV deployment target: recall
17.2% (range 5%-44%) versus 80.4% (range 65%-90%) for the COCO `yolov8n.pt`
baseline with class `person`. The original Privify v0.1 pipeline, based
exclusively on the face detector, is not adequate for the CCTV street-level
deployment target.

**Decision:** Extend the pipeline with three configurable modes:

1. `face`: fine-tuned face detector, blur of the face bbox (original mode,
   suited to close-up scenes).
2. `person_upper`: COCO person detector, blur of the upper 30% of the bbox
   (head-shoulders proxy, conservative for CCTV).
3. `person_full`: COCO person detector, blur of the whole bbox (full-body
   anonymization, maximally conservative).

`mode` is a mandatory parameter of the `process_video` API, forcing an
explicit policy decision based on the deployment context. The API also keeps
manual dependency injection (`detector` + `anonymizer`) for advanced use
cases.

**Consequences:**

- (+) Users can adapt the pipeline to the deployment scenario without
  re-training (`face` for close-up scenes, `person_upper`/`person_full` for
  CCTV street-level).
- (+) The fine-tuned `face-detector-v0.1` keeps its usefulness for its target
  domain (close-up), instead of being made obsolete by the domain shift.
- (+) The design is composable: a future `person_face_hybrid` mode (face when
  available, fallback to person otherwise) can be added without breaking
  changes.
- (-) The `process_video` API is slightly more complex for the user
  (mandatory mode, explicit policy choice).
- (-) `person_upper` introduces an `upper_fraction` parameter that needs
  tuning for specific use cases (the 0.30 default works well on CCTV
  street-level, but may be too conservative for other scenarios).
- (-) The `person_*` modes require downloading `yolov8n.pt` on first use
  (~6 MB, cached in `models/` via the `fix(eval)` commit `7f6ef67`).
