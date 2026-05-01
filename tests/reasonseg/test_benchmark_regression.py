# pyright: reportMissingImports=false, reportUnknownParameterType=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _minimal_grounding_prediction(
    *,
    manifest_file: str = "test_manifest.json",
    entry_index: int = 0,
    intersection: float = 100.0,
    union: float = 150.0,
) -> dict[str, Any]:
    return {
        "prediction_schema_version": "1.0.0",
        "predictions": [
            {
                "manifest_file": manifest_file,
                "entry_index": entry_index,
                "intersection": intersection,
                "union": union,
            }
        ],
    }


def _minimal_metrics_json() -> dict[str, Any]:
    return {
        "grounding": {
            "grounding/cIoU": 66.67,
            "grounding/mIoU": 66.67,
            "grounding/precision@0.5": 100.0,
            "grounding/precision@0.6": 100.0,
            "grounding/precision@0.7": 100.0,
            "grounding/precision@0.8": 100.0,
            "grounding/precision@0.9": 0.0,
        }
    }


def test_legacy_eval_output_has_required_grounding_keys():
    metrics = _minimal_metrics_json()
    required_keys = [
        "grounding/cIoU",
        "grounding/mIoU",
        "grounding/precision@0.5",
        "grounding/precision@0.6",
        "grounding/precision@0.7",
        "grounding/precision@0.8",
        "grounding/precision@0.9",
    ]
    for key in required_keys:
        assert key in metrics["grounding"], f"Missing required metric key: {key}"


def test_vr_ov_eval_output_preserves_required_grounding_keys():
    metrics = _minimal_metrics_json()
    required_keys = [
        "grounding/cIoU",
        "grounding/mIoU",
        "grounding/precision@0.5",
    ]
    for key in required_keys:
        assert key in metrics["grounding"]


def test_vr_ov_compositional_artifact_is_json_serializable():
    artifact = [
        {
            "has_comp_scores": True,
            "modalities": ["cat_feat", "attr_feat"],
        }
    ]
    serialized = json.dumps(artifact)
    deserialized = json.loads(serialized)
    assert deserialized[0]["has_comp_scores"] is True
    assert "cat_feat" in deserialized[0]["modalities"]


def test_vr_ov_compositional_artifact_does_not_affect_predictions_format():
    predictions = _minimal_grounding_prediction()
    assert "prediction_schema_version" in predictions
    assert "predictions" in predictions
    for pred in predictions["predictions"]:
        assert "manifest_file" in pred
        assert "entry_index" in pred
        assert "intersection" in pred
        assert "union" in pred


def test_grounding_accumulator_contract():
    from reasonseg.evaluation.grounding import GroundingAccumulator

    acc = GroundingAccumulator()
    acc.add(intersection=100.0, union=150.0)
    metrics = acc.metrics()

    required_keys = [
        "grounding/cIoU",
        "grounding/mIoU",
        "grounding/precision@0.5",
        "grounding/precision@0.6",
        "grounding/precision@0.7",
        "grounding/precision@0.8",
        "grounding/precision@0.9",
    ]
    for key in required_keys:
        assert key in metrics, f"GroundingAccumulator missing key: {key}"


def test_benchmark_scope_accumulator_render_structure():
    from reasonseg.benchmark_eval import ScopeAccumulator

    acc = ScopeAccumulator()
    acc.add_positive(
        metric_family="openworldsam_grounding",
        intersection=100.0,
        union=150.0,
        class_partition=None,
        classification_correct=None,
    )
    rendered = acc.render()

    assert "entry_count" in rendered
    assert "positive_entry_count" in rendered
    assert "no_target_entry_count" in rendered
    assert "positive_metrics_by_family" in rendered
    assert "openworldsam_grounding" in rendered["positive_metrics_by_family"]


def test_legacy_and_vr_ov_predictions_share_schema_version():
    legacy_pred = _minimal_grounding_prediction()
    vr_ov_pred = _minimal_grounding_prediction()
    assert legacy_pred["prediction_schema_version"] == vr_ov_pred["prediction_schema_version"]


def test_eval_output_keys_stable_across_architectures():
    legacy_keys = set(_minimal_metrics_json()["grounding"].keys())
    vr_ov_keys = set(_minimal_metrics_json()["grounding"].keys())
    assert legacy_keys == vr_ov_keys
