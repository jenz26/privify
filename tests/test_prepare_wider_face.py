"""Unit tests for tools.prepare_wider_face — no GPU, no dataset download.

Only the pure functions (parsing, scale binning, normalisation, biased
sub-sampling, splitting, reporting) are exercised. Functions that touch the
network, the filesystem at scale, ``cv2`` or ``PIL`` are intentionally left to
manual/integration verification.
"""

from __future__ import annotations

import random

import pytest

import tools.prepare_wider_face as pwf


def _record(rel_path: str, bins: tuple[str, ...]) -> pwf.ImageRecord:
    """Build a synthetic ImageRecord with the given per-face scale bins."""
    yolo = tuple((0.5, 0.5, 0.05, 0.05) for _ in bins)
    return pwf.ImageRecord(
        rel_path=rel_path,
        img_w=100,
        img_h=100,
        yolo_boxes=yolo,
        scale_bins=bins,
        hardness=pwf.compute_hardness(bins),
    )


# --------------------------------------------------------------------------- #
# parse_wider_annotations
# --------------------------------------------------------------------------- #


class TestParseWiderAnnotations:
    """Tests for the WIDER FACE annotation parser."""

    GT = (
        "0--Parade/img1.jpg\n"
        "2\n"
        "10 20 30 40 0 0 0 0 0 0\n"
        "50 60 0 80 0 0 0 0 0 0\n"  # w == 0 -> invalid, dropped
        "0--Parade/img2.jpg\n"
        "0\n"
        "0 0 0 0 0 0 0 0 0 0\n"  # dummy line for a 0-face image
        "0--Parade/img3.jpg\n"
        "1\n"
        "5 5 10 10 0 0 0 0 0 0\n"
    )

    def test_parses_all_three_images(self) -> None:
        anns = pwf.parse_wider_annotations(self.GT)
        assert [a.rel_path for a in anns] == [
            "0--Parade/img1.jpg",
            "0--Parade/img2.jpg",
            "0--Parade/img3.jpg",
        ]

    def test_drops_invalid_zero_dimension_box(self) -> None:
        anns = pwf.parse_wider_annotations(self.GT)
        assert len(anns[0].boxes) == 1
        assert anns[0].boxes[0] == pwf.FaceBox(x=10, y=20, w=30, h=40)

    def test_zero_face_image_consumes_dummy_line(self) -> None:
        anns = pwf.parse_wider_annotations(self.GT)
        # img2 has no boxes, and its dummy line must not leak into img3.
        assert anns[1].boxes == ()
        assert len(anns[2].boxes) == 1

    def test_malformed_count_raises(self) -> None:
        with pytest.raises(ValueError, match="integer face count"):
            pwf.parse_wider_annotations("img.jpg\nNOTANUMBER\n")


# --------------------------------------------------------------------------- #
# classify_scale / to_yolo_bbox / compute_hardness
# --------------------------------------------------------------------------- #


class TestScaleAndNormalisation:
    """Tests for scale binning and YOLO normalisation."""

    @pytest.mark.parametrize(
        ("area_fraction", "expected"),
        [
            (0.02, "easy"),
            (0.01, "medium"),  # boundary: not > 1 %
            (0.0075, "medium"),
            (0.005, "hard"),  # boundary: not > 0.5 %
            (0.001, "hard"),
        ],
    )
    def test_classify_scale_boundaries(self, area_fraction: float, expected: str) -> None:
        assert pwf.classify_scale(area_fraction) == expected

    def test_to_yolo_bbox_normalises(self) -> None:
        box = pwf.FaceBox(x=10, y=20, w=30, h=40)
        cx, cy, nw, nh = pwf.to_yolo_bbox(box, img_w=100, img_h=200)
        assert (cx, cy, nw, nh) == pytest.approx((0.25, 0.20, 0.30, 0.20))

    def test_to_yolo_bbox_clips_to_unit_square(self) -> None:
        box = pwf.FaceBox(x=90, y=90, w=30, h=30)
        cx, cy, _, _ = pwf.to_yolo_bbox(box, img_w=100, img_h=100)
        assert cx == 1.0 and cy == 1.0

    def test_to_yolo_bbox_clips_width_and_height(self) -> None:
        # A box larger than the image overflows in width/height before clipping;
        # YOLO labels must stay within the unit square.
        box = pwf.FaceBox(x=0, y=0, w=150, h=150)
        cx, cy, nw, nh = pwf.to_yolo_bbox(box, img_w=100, img_h=100)
        assert all(0.0 <= v <= 1.0 for v in (cx, cy, nw, nh))
        assert nw == 1.0 and nh == 1.0

    def test_to_yolo_bbox_rejects_nonpositive_dims(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            pwf.to_yolo_bbox(pwf.FaceBox(0, 0, 1, 1), img_w=0, img_h=100)

    def test_compute_hardness(self) -> None:
        assert pwf.compute_hardness(("easy", "medium", "hard")) == pytest.approx(1.0)
        assert pwf.compute_hardness(()) == 0.0


# --------------------------------------------------------------------------- #
# select_biased_subset
# --------------------------------------------------------------------------- #


class TestSelectBiasedSubset:
    """Tests for the deterministic biased sub-sampling."""

    @staticmethod
    def _population() -> list[pwf.ImageRecord]:
        records: list[pwf.ImageRecord] = []
        # 30 all-easy close-ups (must be excluded).
        records += [_record(f"easy{i:03d}.jpg", ("easy", "easy")) for i in range(30)]
        # 40 medium-dominant, 30 hard-dominant (eligible).
        records += [_record(f"med{i:03d}.jpg", ("medium", "easy")) for i in range(40)]
        records += [_record(f"hard{i:03d}.jpg", ("hard", "hard")) for i in range(30)]
        return records

    def test_excludes_all_easy_images(self) -> None:
        selected = pwf.select_biased_subset(
            self._population(), target=40, seed=42, medium_fraction=2 / 3
        )
        assert all(not p.startswith("easy") for p in (r.rel_path for r in selected))

    def test_returns_exactly_target_when_pool_is_large(self) -> None:
        selected = pwf.select_biased_subset(
            self._population(), target=40, seed=42, medium_fraction=2 / 3
        )
        assert len(selected) == 40
        # No duplicates.
        assert len({r.rel_path for r in selected}) == 40

    def test_is_deterministic(self) -> None:
        kwargs = {"target": 40, "seed": 42, "medium_fraction": 2 / 3}
        first = pwf.select_biased_subset(self._population(), **kwargs)
        second = pwf.select_biased_subset(self._population(), **kwargs)
        assert [r.rel_path for r in first] == [r.rel_path for r in second]

    def test_returns_all_eligible_when_target_exceeds_pool(self) -> None:
        selected = pwf.select_biased_subset(
            self._population(), target=500, seed=42, medium_fraction=2 / 3
        )
        # 70 eligible (40 medium + 30 hard), all returned.
        assert len(selected) == 70

    def test_oversamples_hard_faces(self) -> None:
        # The whole point of the script: over-represent small (hard) faces.
        selected = pwf.select_biased_subset(
            self._population(), target=40, seed=42, medium_fraction=2 / 3
        )
        hard_selected = sum(1 for r in selected if r.hardness == 2.0)
        # Population hard fraction is 30/70 = 42.9 %; the biased sample skews up.
        assert hard_selected / len(selected) > 30 / 70 + 0.05
        assert hard_selected == 22  # pinned for seed=42 (mutation guard)

    def test_matches_golden_order(self) -> None:
        # Pins the exact seeded output so a change to the seeding scheme, the
        # A-Res key, or the tie-breaker is caught (self-equality cannot).
        selected = pwf.select_biased_subset(
            self._population(), target=10, seed=42, medium_fraction=2 / 3
        )
        assert [r.rel_path for r in selected] == [
            "med030.jpg",
            "hard024.jpg",
            "med003.jpg",
            "hard006.jpg",
            "med023.jpg",
            "hard028.jpg",
            "hard018.jpg",
            "hard017.jpg",
            "med000.jpg",
            "hard001.jpg",
        ]

    def test_is_order_invariant(self) -> None:
        # Output must not depend on input ordering (internal sort by rel_path).
        reference = [
            r.rel_path
            for r in pwf.select_biased_subset(
                self._population(), target=10, seed=42, medium_fraction=2 / 3
            )
        ]
        for shuffle_seed in range(5):
            population = self._population()
            random.Random(shuffle_seed).shuffle(population)
            result = [
                r.rel_path
                for r in pwf.select_biased_subset(
                    population, target=10, seed=42, medium_fraction=2 / 3
                )
            ]
            assert result == reference

    def test_rejects_out_of_range_medium_fraction(self) -> None:
        for bad in (1.5, -0.5):
            with pytest.raises(ValueError, match="medium_fraction must be in"):
                pwf.select_biased_subset(
                    self._population(), target=40, seed=42, medium_fraction=bad
                )


# --------------------------------------------------------------------------- #
# assign_splits / summarise_splits
# --------------------------------------------------------------------------- #


class TestAssignAndSummarise:
    """Tests for the 80/10/10 split and the textual summary."""

    @staticmethod
    def _records(n: int) -> list[pwf.ImageRecord]:
        return [_record(f"img{i:03d}.jpg", ("medium", "hard")) for i in range(n)]

    def test_split_sizes_are_80_10_10(self) -> None:
        splits = pwf.assign_splits(self._records(3000), seed=42)
        assert len(splits["train"]) == 2400
        assert len(splits["val"]) == 300
        assert len(splits["test"]) == 300

    def test_split_is_a_partition(self) -> None:
        records = self._records(123)
        splits = pwf.assign_splits(records, seed=42)
        recovered = {r.rel_path for split in splits.values() for r in split}
        assert recovered == {r.rel_path for r in records}

    def test_split_is_deterministic(self) -> None:
        records = self._records(100)
        first = pwf.assign_splits(records, seed=42)
        second = pwf.assign_splits(records, seed=42)
        assert [r.rel_path for r in first["train"]] == [r.rel_path for r in second["train"]]

    def test_split_train_order_matches_golden(self) -> None:
        # Golden train ordering for seed=42 (cross-run/machine reproducibility).
        splits = pwf.assign_splits(self._records(20), seed=42)
        assert [r.rel_path for r in splits["train"]] == [
            "img019.jpg",
            "img005.jpg",
            "img014.jpg",
            "img004.jpg",
            "img009.jpg",
            "img013.jpg",
            "img015.jpg",
            "img018.jpg",
            "img006.jpg",
            "img012.jpg",
            "img017.jpg",
            "img010.jpg",
            "img001.jpg",
            "img011.jpg",
            "img002.jpg",
            "img016.jpg",
        ]

    def test_summarise_splits_reports_totals(self) -> None:
        splits = pwf.assign_splits(self._records(10), seed=42)
        summary = pwf.summarise_splits(splits)
        assert "TOTAL" in summary
        assert "train" in summary


# --------------------------------------------------------------------------- #
# filter_eligible / summarise_eligible_pool
# --------------------------------------------------------------------------- #


class TestEligiblePool:
    """Tests for eligibility filtering and the natural-distribution report."""

    def test_filter_eligible_drops_all_easy_images(self) -> None:
        records = [
            _record("easy0.jpg", ("easy", "easy")),
            _record("mixed.jpg", ("easy", "medium", "hard")),
            _record("hard0.jpg", ("hard", "hard")),
        ]
        eligible = pwf.filter_eligible(records)
        assert [r.rel_path for r in eligible] == ["mixed.jpg", "hard0.jpg"]

    def test_summarise_eligible_pool_counts_all_faces_per_bin(self) -> None:
        # mixed: easy=1, medium=1, hard=1; hard0: hard=2 -> total 5 faces.
        eligible = [
            _record("mixed.jpg", ("easy", "medium", "hard")),
            _record("hard0.jpg", ("hard", "hard")),
        ]
        summary = pwf.summarise_eligible_pool(eligible)
        assert "2 images, 5 face annotations" in summary
        assert "easy" in summary and "20.0%" in summary  # 1/5 easy and 1/5 medium
        assert "60.0%" in summary  # 3/5 hard

    def test_summarise_eligible_pool_handles_empty(self) -> None:
        summary = pwf.summarise_eligible_pool([])
        assert "0 images, 0 face annotations" in summary
        assert "0.0%" in summary
