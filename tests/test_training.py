"""Unit tests for src.training — no GPU or real dataset required."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.training import TrainingConfig, TrainingResult, train_face_detector

# ------------------------------------------------------------------
# TrainingConfig validation tests
# ------------------------------------------------------------------


class TestTrainingConfigValidation:
    """Tests for :class:`TrainingConfig` __post_init__ validation."""

    def test_training_config_validation_rejects_zero_epochs(self) -> None:
        """TrainingConfig raises ValueError when epochs is 0."""
        dummy_yaml = Path(__file__).parent / "dummy_data.yaml"
        dummy_yaml.touch()
        try:
            with pytest.raises(ValueError, match="epochs must be > 0"):
                TrainingConfig(
                    dataset_yaml_path=dummy_yaml,
                    output_dir=Path("/tmp/runs"),
                    epochs=0,
                )
        finally:
            dummy_yaml.unlink()

    def test_training_config_validation_rejects_invalid_base_model(self) -> None:
        """TrainingConfig raises ValueError for unsupported base models."""
        dummy_yaml = Path(__file__).parent / "dummy_data.yaml"
        dummy_yaml.touch()
        try:
            with pytest.raises(ValueError, match="base_model must be one of"):
                TrainingConfig(
                    dataset_yaml_path=dummy_yaml,
                    output_dir=Path("/tmp/runs"),
                    base_model="yolov8x.pt",
                )
        finally:
            dummy_yaml.unlink()

    def test_training_config_validation_requires_existing_dataset_yaml(
        self, tmp_path: Path
    ) -> None:
        """TrainingConfig raises FileNotFoundError for missing data.yaml."""
        missing = tmp_path / "nonexistent" / "data.yaml"
        with pytest.raises(FileNotFoundError, match="dataset_yaml_path does not exist"):
            TrainingConfig(
                dataset_yaml_path=missing,
                output_dir=Path("/tmp/runs"),
            )


# ------------------------------------------------------------------
# TrainingResult tests
# ------------------------------------------------------------------


class TestTrainingResult:
    """Tests for the :class:`TrainingResult` frozen dataclass."""

    def test_training_result_is_frozen(self) -> None:
        """Assigning to a field of a frozen TrainingResult raises an error."""
        result = TrainingResult(
            best_weights_path=Path("best.pt"),
            final_metrics={"mAP50": 0.9},
            elapsed_seconds=120.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.elapsed_seconds = 999.0  # type: ignore[misc]


# ------------------------------------------------------------------
# train_face_detector tests (fully mocked)
# ------------------------------------------------------------------


class TestTrainFaceDetector:
    """Tests for :func:`train_face_detector` with mocked ultralytics."""

    def test_train_face_detector_invokes_yolo_with_correct_args(self, tmp_path: Path) -> None:
        """train_face_detector passes config parameters to YOLO.train()."""
        # Create a dummy data.yaml so TrainingConfig validation passes.
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("train: ./train\nval: ./val\nnc: 1\nnames: ['face']\n")

        output_dir = tmp_path / "runs"

        config = TrainingConfig(
            dataset_yaml_path=data_yaml,
            output_dir=output_dir,
            base_model="yolov8n.pt",
            epochs=10,
            image_size=320,
            batch_size=8,
            device="cpu",
        )

        # Mock YOLO model and trainer.
        mock_model = MagicMock()
        mock_model.trainer.metrics = {
            "metrics/mAP50(B)": 0.85,
            "metrics/mAP50-95(B)": 0.55,
            "metrics/precision(B)": 0.90,
            "metrics/recall(B)": 0.80,
        }
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with patch("ultralytics.YOLO", mock_yolo_cls):
            result = train_face_detector(config)

        # Verify YOLO was instantiated with the correct base model.
        mock_yolo_cls.assert_called_once_with("yolov8n.pt")

        # Verify model.train was called with the correct parameters.
        mock_model.train.assert_called_once_with(
            data=str(data_yaml),
            epochs=10,
            imgsz=320,
            batch=8,
            device="cpu",
            project=str(output_dir),
            name="train",
            exist_ok=True,
        )

        # Verify the returned TrainingResult.
        assert isinstance(result, TrainingResult)
        assert result.best_weights_path == output_dir / "train" / "weights" / "best.pt"
        assert result.final_metrics["mAP50"] == pytest.approx(0.85)
        assert result.final_metrics["mAP50-95"] == pytest.approx(0.55)
        assert result.final_metrics["precision"] == pytest.approx(0.90)
        assert result.final_metrics["recall"] == pytest.approx(0.80)
        assert result.elapsed_seconds >= 0.0
