"""Prepare a small-face-biased WIDER FACE subset in YOLOv8 format (v0.2).

This script downloads the official WIDER FACE training images and annotations,
parses the custom annotation format, sub-samples ~3 000 images biased towards
*small* faces (medium/hard scale bins), converts the labels to YOLOv8 format,
produces a deterministic 80/10/10 split, writes a preview, and packages the
result for upload as a GitHub Release asset.

The motivation is documented in ``docs/decisions.md``: the v0.1 face detector
was trained on close-up portraits and under-performs on the CCTV deployment
target, where faces are small. A dataset whose scale distribution matches the
deployment target is expected to improve recall.

Design notes:
    * Pure functions (parsing, scale binning, normalisation, sub-sampling,
      splitting) are separated from I/O so they can be unit-tested without
      downloading the dataset or using a GPU.
    * Heavy third-party libraries (``cv2``, ``PIL``) are imported lazily inside
      the functions that need them, keeping module import cheap for tests.
    * All randomness flows through ``random.Random(seed)`` for reproducibility.

Typical usage::

    python tools/prepare_wider_face.py            # download + prepare + package
    python tools/prepare_wider_face.py --skip-download   # reuse local zips
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Public HuggingFace mirror of the original WIDER FACE archives. The ``resolve``
#: endpoints are plain HTTPS downloads (urllib-friendly, redirected to a CDN),
#: unlike the official Google Drive links which require a confirm-token dance.
WIDER_BASE_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data"

#: Archive file names expected in the work directory (downloaded or placed by
#: hand). Only the training images and the annotation split are needed: the
#: validation/test archives are not used to build the subset.
TRAIN_ZIP = "WIDER_train.zip"
SPLIT_ZIP = "wider_face_split.zip"

#: Relative path of the training ground-truth file inside ``wider_face_split.zip``.
TRAIN_GT_RELPATH = Path("wider_face_split") / "wider_face_train_bbx_gt.txt"

#: Relative path of the training images directory inside ``WIDER_train.zip``.
TRAIN_IMAGES_RELPATH = Path("WIDER_train") / "images"

#: Scale bins follow the WIDER FACE convention, expressed as the fraction of the
#: image area covered by a face bounding box.
EASY_MIN_AREA_FRACTION = 0.01  # area > 1 %        -> easy  (large faces)
MEDIUM_MIN_AREA_FRACTION = 0.005  # 0.5 % < area <= 1 % -> medium
#                                  area <= 0.5 %     -> hard  (small faces)

#: Per-bin weight used to compute an image "hardness" score (easy faces do not
#: contribute; harder faces weigh more).
_BIN_WEIGHT = {"easy": 0.0, "medium": 1.0, "hard": 2.0}

#: Human-readable area-fraction range per scale bin (for diagnostic reporting).
_BIN_RANGE_LABEL = {"easy": "> 1%", "medium": "0.5-1%", "hard": "<= 0.5%"}

#: YOLO class id and name (single-class detector).
FACE_CLASS_ID = 0

#: data.yaml template. The ``path`` is a Colab placeholder, patched to an
#: absolute path at training time (same strategy as the v0.1 notebook).
DATA_YAML_TEMPLATE = """\
path: /content/datasets/wider_face_subset_v0.2  # placeholder Colab path
train: images/train
val: images/val
test: images/test

nc: 1
names: ['face']
"""


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FaceBox:
    """A single WIDER FACE bounding box in pixel coordinates.

    Attributes:
        x: Top-left x coordinate in pixels.
        y: Top-left y coordinate in pixels.
        w: Box width in pixels.
        h: Box height in pixels.
    """

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class ImageAnnotation:
    """Raw annotation for one WIDER FACE image (pixel coordinates).

    Attributes:
        rel_path: Image path relative to the images root, e.g.
            ``"0--Parade/0_Parade_marchingband_1_849.jpg"``.
        boxes: Valid face boxes (zero-area / invalid boxes are dropped).
    """

    rel_path: str
    boxes: tuple[FaceBox, ...]


@dataclass(frozen=True)
class ImageRecord:
    """An image with normalised YOLO boxes and per-face scale information.

    Attributes:
        rel_path: Image path relative to the images root.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        yolo_boxes: One ``(cx, cy, w, h)`` tuple per face, normalised to [0, 1].
        scale_bins: One scale bin (``"easy"``/``"medium"``/``"hard"``) per face,
            aligned with *yolo_boxes*.
        hardness: Mean per-face bin weight (see :data:`_BIN_WEIGHT`); higher
            means the image is dominated by smaller faces.
    """

    rel_path: str
    img_w: int
    img_h: int
    yolo_boxes: tuple[tuple[float, float, float, float], ...]
    scale_bins: tuple[str, ...]
    hardness: float


# --------------------------------------------------------------------------- #
# Pure functions: parsing, binning, normalisation
# --------------------------------------------------------------------------- #


def parse_wider_annotations(gt_text: str) -> list[ImageAnnotation]:
    """Parse the WIDER FACE ``*_bbx_gt.txt`` annotation format.

    The format repeats, per image::

        <image_relative_path>
        <num_faces>
        x1 y1 w h blur expression illumination invalid occlusion pose
        ... (num_faces lines)

    Two well-known quirks are handled:
        * Images with ``num_faces == 0`` still carry a single dummy bbox line
          ``0 0 0 0 0 0 0 0 0 0`` which must be consumed and discarded.
        * Some boxes have ``w == 0`` or ``h == 0`` (invalid annotations); these
          are dropped.

    Args:
        gt_text: Full text content of the ground-truth file.

    Returns:
        One :class:`ImageAnnotation` per image, in file order.

    Raises:
        ValueError: If the file structure is malformed (e.g. a non-integer
            where a face count is expected).
    """
    lines = gt_text.splitlines()
    annotations: list[ImageAnnotation] = []
    i = 0
    n = len(lines)

    while i < n:
        rel_path = lines[i].strip()
        i += 1
        if not rel_path:
            continue  # tolerate stray blank lines

        if i >= n:
            raise ValueError(f"Missing face-count line after image path {rel_path!r}")
        try:
            count = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(
                f"Expected an integer face count after {rel_path!r}, got {lines[i]!r}"
            ) from exc
        i += 1

        # Even a 0-face image is followed by one dummy bbox line in WIDER FACE.
        n_bbox_lines = count if count > 0 else 1
        boxes: list[FaceBox] = []
        for _ in range(n_bbox_lines):
            if i >= n:
                raise ValueError(f"Truncated bbox block for image {rel_path!r}")
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                raise ValueError(f"Malformed bbox line for {rel_path!r}: {lines[i - 1]!r}")
            try:
                x, y, w, h = (float(parts[k]) for k in range(4))
            except ValueError as exc:
                raise ValueError(
                    f"Malformed bbox coordinates for {rel_path!r}: {lines[i - 1]!r}"
                ) from exc
            if count > 0 and w > 0 and h > 0:
                boxes.append(FaceBox(x=x, y=y, w=w, h=h))

        annotations.append(ImageAnnotation(rel_path=rel_path, boxes=tuple(boxes)))

    return annotations


def classify_scale(area_fraction: float) -> str:
    """Classify a face into a WIDER FACE scale bin from its area fraction.

    Args:
        area_fraction: ``face_area / image_area`` in [0, 1].

    Returns:
        ``"easy"`` if area > 1 %, ``"medium"`` if 0.5 % < area <= 1 %,
        otherwise ``"hard"``.
    """
    if area_fraction > EASY_MIN_AREA_FRACTION:
        return "easy"
    if area_fraction > MEDIUM_MIN_AREA_FRACTION:
        return "medium"
    return "hard"


def to_yolo_bbox(box: FaceBox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert a pixel-coordinate box to a normalised YOLO box.

    Args:
        box: Face box in pixel coordinates.
        img_w: Image width in pixels (must be > 0).
        img_h: Image height in pixels (must be > 0).

    Returns:
        ``(cx, cy, w, h)`` normalised to [0, 1] and clipped to the unit square.

    Raises:
        ValueError: If *img_w* or *img_h* is not positive.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got {img_w}x{img_h}")

    cx = (box.x + box.w / 2.0) / img_w
    cy = (box.y + box.h / 2.0) / img_h
    nw = box.w / img_w
    nh = box.h / img_h
    return (_clip01(cx), _clip01(cy), _clip01(nw), _clip01(nh))


def _clip01(value: float) -> float:
    """Clip *value* to the closed interval [0, 1]."""
    return max(0.0, min(1.0, value))


def compute_hardness(scale_bins: tuple[str, ...]) -> float:
    """Compute the mean per-face hardness weight for an image.

    Args:
        scale_bins: Per-face scale bins.

    Returns:
        Mean of :data:`_BIN_WEIGHT` over the faces, or 0.0 if there are no
        faces.
    """
    if not scale_bins:
        return 0.0
    return sum(_BIN_WEIGHT[b] for b in scale_bins) / len(scale_bins)


def build_image_records(annotations: list[ImageAnnotation], images_root: Path) -> list[ImageRecord]:
    """Enrich raw annotations with image size, YOLO boxes and scale bins.

    Reads only the image header (via PIL) to recover dimensions, so this scales
    to the full WIDER FACE training set without decoding pixel data.

    Args:
        annotations: Parsed annotations (pixel coordinates).
        images_root: Directory the *rel_path* values are relative to.

    Returns:
        One :class:`ImageRecord` per readable, non-empty image. Images that are
        missing on disk, unreadable, or have no valid faces are skipped (with a
        warning for the first two cases).
    """
    from PIL import Image  # lazy import (provided transitively by ultralytics)

    records: list[ImageRecord] = []
    for ann in annotations:
        if not ann.boxes:
            continue  # all-invalid or genuinely empty image
        image_path = images_root / ann.rel_path
        if not image_path.exists():
            logger.warning("Image not found, skipping: %s", image_path)
            continue
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except OSError as exc:
            logger.warning("Could not read %s (%s), skipping", image_path, exc)
            continue
        if img_w <= 0 or img_h <= 0:
            continue

        image_area = float(img_w * img_h)
        yolo_boxes: list[tuple[float, float, float, float]] = []
        scale_bins: list[str] = []
        for box in ann.boxes:
            yolo_boxes.append(to_yolo_bbox(box, img_w, img_h))
            scale_bins.append(classify_scale((box.w * box.h) / image_area))

        bins = tuple(scale_bins)
        records.append(
            ImageRecord(
                rel_path=ann.rel_path,
                img_w=img_w,
                img_h=img_h,
                yolo_boxes=tuple(yolo_boxes),
                scale_bins=bins,
                hardness=compute_hardness(bins),
            )
        )

    return records


# --------------------------------------------------------------------------- #
# Pure functions: biased sub-sampling and splitting
# --------------------------------------------------------------------------- #


def _is_eligible(record: ImageRecord) -> bool:
    """Return True if the image has at least one medium or hard face."""
    return any(b in ("medium", "hard") for b in record.scale_bins)


def filter_eligible(records: list[ImageRecord]) -> list[ImageRecord]:
    """Return images with at least one medium/hard face (drop all-easy ones).

    Args:
        records: All image records.

    Returns:
        The subset of *records* that are eligible for sub-sampling.
    """
    return [r for r in records if _is_eligible(r)]


def _weighted_sample_without_replacement(
    items: list[ImageRecord],
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[ImageRecord]:
    """Sample *k* distinct items with probability proportional to *weights*.

    Uses the Efraimidis-Spirakis A-Res scheme (key = u ** (1 / weight)), which
    is exact weighted sampling without replacement and fully deterministic for
    a seeded *rng* and a stable input order.

    Args:
        items: Candidate items (must be in a stable order).
        weights: One positive weight per item; non-positive weights are skipped.
        k: Number of items to draw (capped at the number of positive weights).
        rng: Seeded random generator.

    Returns:
        Up to *k* sampled items, highest-key first.
    """
    keyed: list[tuple[float, int, ImageRecord]] = []
    for idx, (item, weight) in enumerate(zip(items, weights, strict=True)):
        if weight <= 0:
            continue
        key = rng.random() ** (1.0 / weight)
        keyed.append((key, idx, item))
    # Sort by key desc; idx is a deterministic tie-breaker.
    keyed.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    # Clamp k: a negative slice bound would silently return the wrong items.
    return [item for _, _, item in keyed[: max(0, k)]]


def select_biased_subset(
    records: list[ImageRecord],
    target: int,
    seed: int,
    medium_fraction: float,
) -> list[ImageRecord]:
    """Select ~*target* images biased towards small faces.

    Strategy:
        1. Keep only *eligible* images (>= 1 medium/hard face); drop all-easy
           close-ups (already covered by the v0.1 portrait dataset).
        2. Draw ``round(target * medium_fraction)`` images weighted by hardness
           **capped at 1.0** (medium-leaning, avoids an extreme small-face bias).
        3. Draw the remaining images from the leftover pool weighted by raw
           hardness (hard-leaning).
        4. Top up by descending hardness if the pool is too small.

    Args:
        records: All image records.
        target: Desired subset size.
        seed: Seed for the deterministic sampler.
        medium_fraction: Fraction of *target* drawn in the medium-leaning pass.

    Returns:
        The selected records (order is sampler-defined, then deterministic).
        If fewer than *target* images are eligible, all of them are returned.

    Raises:
        ValueError: If *medium_fraction* is outside [0, 1].
    """
    if not 0.0 <= medium_fraction <= 1.0:
        raise ValueError(f"medium_fraction must be in [0, 1], got {medium_fraction}")

    eligible = sorted(filter_eligible(records), key=lambda r: r.rel_path)
    if len(eligible) <= target:
        logger.info(
            "Only %d eligible images (<= target %d); using all of them.",
            len(eligible),
            target,
        )
        return eligible

    rng = random.Random(seed)
    n_medium = round(target * medium_fraction)
    n_hard = target - n_medium

    medium_weights = [min(r.hardness, 1.0) for r in eligible]
    medium_pick = _weighted_sample_without_replacement(eligible, medium_weights, n_medium, rng)

    picked_paths = {r.rel_path for r in medium_pick}
    leftover = [r for r in eligible if r.rel_path not in picked_paths]
    hard_weights = [r.hardness for r in leftover]
    hard_pick = _weighted_sample_without_replacement(leftover, hard_weights, n_hard, rng)

    selected = medium_pick + hard_pick

    if len(selected) < target:
        # Top up from whatever remains, hardest first, for a stable result.
        chosen = {r.rel_path for r in selected}
        remainder = sorted(
            (r for r in eligible if r.rel_path not in chosen),
            key=lambda r: (-r.hardness, r.rel_path),
        )
        selected.extend(remainder[: target - len(selected)])

    return selected


def assign_splits(records: list[ImageRecord], seed: int) -> dict[str, list[ImageRecord]]:
    """Shuffle deterministically and split into 80/10/10 train/val/test.

    Args:
        records: Selected image records.
        seed: Seed for the shuffle.

    Returns:
        Mapping ``{"train": [...], "val": [...], "test": [...]}``.
    """
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * 0.8)
    n_val = round(n * 0.1)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


# --------------------------------------------------------------------------- #
# I/O: download, conversion, preview, packaging
# --------------------------------------------------------------------------- #


def download_file(url: str, dest: Path) -> None:
    """Stream a URL to *dest*, sending a browser-like User-Agent.

    Args:
        url: HTTPS URL to download.
        dest: Destination file path (parent directories are created).

    Raises:
        urllib.error.URLError: If the download fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    request = urllib.request.Request(url, headers={"User-Agent": "privify-dataset-prep"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out, length=1024 * 1024)
    logger.info("Saved %.1f MB to %s", dest.stat().st_size / 1e6, dest)


def download_wider_face(
    workdir: Path, *, skip_download: bool, base_url: str = WIDER_BASE_URL
) -> Path:
    """Ensure WIDER FACE training images and annotations are present in *workdir*.

    Downloads ``WIDER_train.zip`` and ``wider_face_split.zip`` (unless
    *skip_download*), extracts them, and returns the images root. Existing
    extracted directories are reused; existing zips are not re-downloaded.

    Args:
        workdir: Directory to download and extract into.
        skip_download: If True, expect the zips already in *workdir*.
        base_url: Base URL the archive names are appended to.

    Returns:
        Path to the extracted training images root.

    Raises:
        FileNotFoundError: If a required archive is missing and *skip_download*
            is set, or if extraction does not produce the expected layout.
        zipfile.BadZipFile: If an archive cannot be extracted.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    images_root = workdir / TRAIN_IMAGES_RELPATH
    gt_file = workdir / TRAIN_GT_RELPATH

    for archive in (TRAIN_ZIP, SPLIT_ZIP):
        zip_path = workdir / archive
        extracted_marker = images_root if archive == TRAIN_ZIP else gt_file
        if extracted_marker.exists():
            logger.info("Already extracted (%s present), skipping %s", extracted_marker, archive)
            continue
        if not zip_path.exists():
            if skip_download:
                raise FileNotFoundError(
                    f"{zip_path} not found and --skip-download was set. Download "
                    f"{archive} from {base_url}/{archive} (or the official page "
                    "http://shuoyang1213.me/WIDERFACE/) and place it there."
                )
            download_file(f"{base_url}/{archive}", zip_path)
        logger.info("Extracting %s ...", zip_path)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(workdir)
        except zipfile.BadZipFile as exc:
            raise zipfile.BadZipFile(f"{zip_path} is corrupt; re-download it.") from exc

    if not images_root.exists():
        raise FileNotFoundError(f"Expected images at {images_root} after extraction.")
    if not gt_file.exists():
        raise FileNotFoundError(f"Expected annotations at {gt_file} after extraction.")
    return images_root


def write_yolo_dataset(
    splits: dict[str, list[ImageRecord]], images_root: Path, output_dir: Path
) -> None:
    """Copy images and write YOLO label files for every split.

    Source paths contain sub-directories (``category/file.jpg``); these are
    flattened to ``category__file.jpg`` so each split directory is flat and
    image/label stems match.

    Args:
        splits: Mapping of split name to image records.
        images_root: Root the record *rel_path* values are relative to.
        output_dir: Destination dataset root.
    """
    for split, records in splits.items():
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for record in records:
            flat_stem = record.rel_path.replace("/", "__")
            src = images_root / record.rel_path
            shutil.copy2(src, img_dir / flat_stem)

            label_lines = [
                f"{FACE_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for (cx, cy, w, h) in record.yolo_boxes
            ]
            label_path = (lbl_dir / flat_stem).with_suffix(".txt")
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        logger.info("Wrote split %-5s: %d images", split, len(records))


def write_data_yaml(output_dir: Path) -> Path:
    """Write the YOLOv8 ``data.yaml`` for the subset.

    Args:
        output_dir: Dataset root.

    Returns:
        Path to the written ``data.yaml``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(DATA_YAML_TEMPLATE, encoding="utf-8")
    return yaml_path


def generate_preview(
    records: list[ImageRecord],
    images_root: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
) -> list[Path]:
    """Draw bounding boxes on a few sampled images for visual inspection.

    Args:
        records: Records to sample previews from (e.g. the train split).
        images_root: Root the record *rel_path* values are relative to.
        output_dir: Dataset root; previews go to ``output_dir / "_preview"``.
        count: Number of preview images to produce.
        seed: Seed for the deterministic sample.

    Returns:
        Paths of the written preview PNG files.
    """
    import cv2  # lazy import (opencv-python, part of the runtime stack)

    preview_dir = output_dir / "_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    sample = list(records)
    random.Random(seed).shuffle(sample)
    sample = sample[:count]

    written: list[Path] = []
    for record in sample:
        image = cv2.imread(str(images_root / record.rel_path))
        if image is None:
            logger.warning("Preview skipped (unreadable): %s", record.rel_path)
            continue
        height, width = image.shape[:2]
        for (cx, cy, nw, nh), bin_name in zip(record.yolo_boxes, record.scale_bins, strict=True):
            x1 = int((cx - nw / 2) * width)
            y1 = int((cy - nh / 2) * height)
            x2 = int((cx + nw / 2) * width)
            y2 = int((cy + nh / 2) * height)
            colour = (0, 0, 255) if bin_name == "hard" else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 1)

        out_path = preview_dir / f"preview_{record.rel_path.replace('/', '__')}.png"
        cv2.imwrite(str(out_path), image)
        written.append(out_path)

    logger.info("Wrote %d preview image(s) to %s", len(written), preview_dir)
    return written


def package_subset(output_dir: Path, zip_path: Path) -> Path:
    """Zip the prepared subset directory.

    Args:
        output_dir: Dataset root to archive.
        zip_path: Destination ``.zip`` path.

    Returns:
        The written *zip_path*.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Packaging %s -> %s ...", output_dir, zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(output_dir.parent))
    logger.info("Archive size: %.1f MB", zip_path.stat().st_size / 1e6)
    return zip_path


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def summarise_eligible_pool(eligible: list[ImageRecord]) -> str:
    """Report the natural scale-bin distribution of the eligible pool.

    Counts *every* face annotation across the eligible images (including easy
    faces that co-occur with medium/hard faces in the same image), to document
    the distribution the biased sub-sampling starts from — the empirical
    justification for biasing towards small faces.

    Args:
        eligible: Eligible image records (see :func:`filter_eligible`).

    Returns:
        A multi-line summary string (eligible image count, total face
        annotations, and the easy/medium/hard percentages).
    """
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for record in eligible:
        for bin_name in record.scale_bins:
            counts[bin_name] += 1
    total = sum(counts.values())

    lines = [
        "",
        f"Eligible pool (before sub-sampling): {len(eligible)} images, {total} face annotations",
    ]
    for bin_name in ("easy", "medium", "hard"):
        pct = 100.0 * counts[bin_name] / total if total else 0.0
        label = _BIN_RANGE_LABEL[bin_name]
        lines.append(f"  {bin_name:<6} (area {label:<7}): {counts[bin_name]:>7} ({pct:5.1f}%)")
    return "\n".join(lines)


def summarise_splits(splits: dict[str, list[ImageRecord]]) -> str:
    """Build a human-readable summary of the prepared splits.

    Args:
        splits: Mapping of split name to image records.

    Returns:
        A multi-line summary string (image counts, face counts, scale mix).
    """
    lines = ["", "Subset summary (per split):"]
    grand_faces = 0
    grand_images = 0
    bin_totals = {"easy": 0, "medium": 0, "hard": 0}
    for split in ("train", "val", "test"):
        records = splits.get(split, [])
        bins = {"easy": 0, "medium": 0, "hard": 0}
        faces = 0
        for record in records:
            for bin_name in record.scale_bins:
                bins[bin_name] += 1
                faces += 1
        for key in bin_totals:
            bin_totals[key] += bins[key]
        grand_images += len(records)
        grand_faces += faces
        pct_med = 100.0 * bins["medium"] / faces if faces else 0.0
        pct_hard = 100.0 * bins["hard"] / faces if faces else 0.0
        lines.append(
            f"  {split:<5} | images={len(records):>4} | faces={faces:>6} | "
            f"medium={pct_med:5.1f}% | hard={pct_hard:5.1f}%"
        )

    pct_med = 100.0 * bin_totals["medium"] / grand_faces if grand_faces else 0.0
    pct_hard = 100.0 * bin_totals["hard"] / grand_faces if grand_faces else 0.0
    lines.append(
        f"  TOTAL | images={grand_images:>4} | faces={grand_faces:>6} | "
        f"medium={pct_med:5.1f}% | hard={pct_hard:5.1f}%"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI orchestration
# --------------------------------------------------------------------------- #


def build_subset(
    *,
    workdir: Path,
    output_dir: Path,
    zip_path: Path | None,
    target: int,
    seed: int,
    medium_fraction: float,
    preview_count: int,
    skip_download: bool,
) -> None:
    """Run the full preparation pipeline end to end.

    Args:
        workdir: Download/extraction directory for the raw WIDER FACE archives.
        output_dir: Destination YOLO dataset root.
        zip_path: Output archive path, or None to skip packaging.
        target: Desired subset size.
        seed: Master seed for sub-sampling, splitting and preview.
        medium_fraction: Fraction of *target* drawn in the medium-leaning pass.
        preview_count: Number of preview images to render.
        skip_download: Reuse local archives instead of downloading.
    """
    images_root = download_wider_face(workdir, skip_download=skip_download)

    gt_text = (workdir / TRAIN_GT_RELPATH).read_text(encoding="utf-8")
    annotations = parse_wider_annotations(gt_text)
    logger.info("Parsed %d annotated images from the training set.", len(annotations))

    records = build_image_records(annotations, images_root)
    logger.info("Built %d image records with valid faces.", len(records))

    # Diagnostic: the natural scale distribution the sub-sampling starts from.
    logger.info("%s", summarise_eligible_pool(filter_eligible(records)))

    selected = select_biased_subset(
        records, target=target, seed=seed, medium_fraction=medium_fraction
    )
    logger.info("Selected %d images for the subset.", len(selected))

    splits = assign_splits(selected, seed=seed)
    write_yolo_dataset(splits, images_root, output_dir)
    write_data_yaml(output_dir)
    generate_preview(splits["train"], images_root, output_dir, count=preview_count, seed=seed)

    summary = summarise_splits(splits)
    logger.info("%s", summary)

    if zip_path is not None:
        package_subset(output_dir, zip_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    default_workdir = Path(tempfile.gettempdir()) / "wider_face"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=default_workdir,
        help=f"Download/extraction directory (default: {default_workdir}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/wider_face_subset_v0.2"),
        help="Destination YOLO dataset root (default: %(default)s).",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("face-detection-dataset-v0.2.zip"),
        help="Output archive path (default: %(default)s). Use --no-package to skip.",
    )
    parser.add_argument("--no-package", action="store_true", help="Do not create the .zip archive.")
    parser.add_argument(
        "--target", type=int, default=3000, help="Subset size (default: %(default)s)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: %(default)s).")
    parser.add_argument(
        "--medium-fraction",
        type=float,
        default=2.0 / 3.0,
        help="Fraction drawn in the medium-leaning pass (default: ~0.67).",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="Number of preview images (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Reuse local archives in --workdir."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    build_subset(
        workdir=args.workdir,
        output_dir=args.output,
        zip_path=None if args.no_package else args.zip,
        target=args.target,
        seed=args.seed,
        medium_fraction=args.medium_fraction,
        preview_count=args.preview_count,
        skip_download=args.skip_download,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
