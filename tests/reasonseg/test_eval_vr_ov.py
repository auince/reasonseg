# pyright: reportMissingImports=false, reportUnknownParameterType=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_legacy_eval_no_additive_artifact(tmp_path: Path) -> None:
    from reasonseg.runtime.eval import run_evaluation

    mock_model = MagicMock()
    mock_model.training = False
    mock_cfg = MagicMock()
    mock_cfg.DATASETS.TEST = ["refcoco_val_unc"]
    mock_cfg.OUTPUT_DIR = str(tmp_path)

    mock_output = {
        "grounding_mask": MagicMock(),
        "grounding_scores": MagicMock(),
    }
    mock_output["grounding_mask"].sigmoid.return_value = MagicMock()
    mock_output["grounding_mask"].sigmoid.return_value.__gt__ = MagicMock()
    mock_output["grounding_mask"].sigmoid.return_value.__gt__.return_value = MagicMock()
    mock_output["grounding_mask"].sigmoid.return_value.__gt__.return_value.cpu.return_value = MagicMock()
    mock_output["grounding_mask"].sigmoid.return_value.__gt__.return_value.cpu.return_value.numpy.return_value = MagicMock()

    mock_loader = [[{"groundings": {"masks": MagicMock(), "texts": []}, "image_id": 1, "prompt": ["test"]}]]
    mock_deps = {
        "comm": MagicMock(),
    }
    mock_deps["comm"].get_world_size.return_value = 1
    mock_deps["comm"].is_main_process.return_value = True

    inference_dir = tmp_path / "inference" / "iter_0"
    inference_dir.mkdir(parents=True)

    with patch("reasonseg.runtime.common._import_runtime_deps", return_value=mock_deps):
        with patch("reasonseg.runtime.common.build_refcoco_test_loader", return_value=mock_loader):
            with patch("reasonseg.evaluation.grounding.GroundingEvaluator") as MockEvaluator:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate.return_value = {"grounding": {"mIoU": 50.0}}
                mock_evaluator.progress_metrics.return_value = None
                MockEvaluator.return_value = mock_evaluator

                mock_model.return_value = [mock_output]
                run_evaluation(mock_model, mock_cfg, deps=mock_deps, output_dir=inference_dir)

    assert not (inference_dir / "vr_ov_compositional.json").exists()


def test_vr_ov_eval_writes_additive_artifact(tmp_path: Path) -> None:
    from reasonseg.runtime.eval import run_evaluation

    mock_model = MagicMock()
    mock_model.training = False
    mock_cfg = MagicMock()
    mock_cfg.DATASETS.TEST = ["refcoco_val_unc"]
    mock_cfg.OUTPUT_DIR = str(tmp_path)

    mock_mask = MagicMock()
    mock_mask.sigmoid.return_value = MagicMock()
    mock_mask.sigmoid.return_value.__gt__ = MagicMock()
    mock_mask.sigmoid.return_value.__gt__.return_value = MagicMock()
    mock_mask.sigmoid.return_value.__gt__.return_value.cpu.return_value = MagicMock()
    mock_mask.sigmoid.return_value.__gt__.return_value.cpu.return_value.numpy.return_value = MagicMock()

    mock_output = {
        "grounding_mask": mock_mask,
        "grounding_scores": MagicMock(),
        "vr_ov_compositional": {
            "has_comp_scores": True,
            "modalities": ["cat_feat", "attr_feat"],
        },
    }

    mock_loader = [[{"groundings": {"masks": MagicMock(), "texts": []}, "image_id": 1, "prompt": ["test"]}]]
    mock_deps = {
        "comm": MagicMock(),
    }
    mock_deps["comm"].get_world_size.return_value = 1
    mock_deps["comm"].is_main_process.return_value = True

    inference_dir = tmp_path / "inference" / "iter_0"
    inference_dir.mkdir(parents=True)

    with patch("reasonseg.runtime.common._import_runtime_deps", return_value=mock_deps):
        with patch("reasonseg.runtime.common.build_refcoco_test_loader", return_value=mock_loader):
            with patch("reasonseg.evaluation.grounding.GroundingEvaluator") as MockEvaluator:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate.return_value = {"grounding": {"mIoU": 50.0}}
                mock_evaluator.progress_metrics.return_value = None
                MockEvaluator.return_value = mock_evaluator

                mock_model.return_value = [mock_output]
                run_evaluation(mock_model, mock_cfg, deps=mock_deps, output_dir=inference_dir)

    artifact_path = inference_dir / "vr_ov_compositional.json"
    assert artifact_path.exists()
    data = json.loads(artifact_path.read_text())
    assert len(data) == 1
    assert data[0]["has_comp_scores"] is True
    assert "cat_feat" in data[0]["modalities"]


def test_invalid_vr_ov_config_fails_early(tmp_path: Path) -> None:
    from reasonseg.runtime.eval import _worker

    fake_checkpoint = tmp_path / "fake_checkpoint.pth"
    fake_checkpoint.write_text("")

    mock_args = MagicMock()
    mock_args.split = "refcoco_val_unc"
    mock_args.opts = []
    mock_args.checkpoint = str(fake_checkpoint)

    mock_cfg = MagicMock()
    mock_cfg.MODEL.META_ARCHITECTURE = "VR_OV"
    mock_cfg.MODEL.VR_OV = MagicMock()
    mock_cfg.MODEL.VR_OV.ENABLED = True
    mock_cfg.MODEL.VR_OV.QUERY_PARSER = MagicMock()
    mock_cfg.MODEL.VR_OV.QUERY_PARSER.HIDDEN_DIM = 128
    mock_cfg.dump.return_value = "test_cfg_dump"

    with patch("reasonseg.runtime.common._import_runtime_deps") as mock_import:
        mock_deps = MagicMock()
        mock_import.return_value = mock_deps
        with patch("reasonseg.runtime.common.setup_cfg", return_value=mock_cfg):
            with patch("reasonseg.runtime.common.setup_runtime_logging"):
                with patch("model.vr_ov_config.validate_vr_ov_config", side_effect=ValueError("Missing required field")):
                    with pytest.raises(ValueError, match="Missing required field"):
                        _worker(mock_args)
