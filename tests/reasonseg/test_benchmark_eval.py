# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest

from reasonseg.benchmark_runner import run_benchmark
from reasonseg.benchmark_eval import evaluate_predictions, render_csv
from reasonseg.evaluation import GroundingAccumulator


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SPEC = ROOT / "benchmarks/reasonseg_phase1_benchmark_spec.json"
PAPER_BENCHMARK_SPEC = ROOT / "benchmarks/refexp_paper_benchmark.json"
SMOKE_MANIFESTS = [
    ROOT / "benchmarks/smoke/reasonseg_phase1_smoke_refcoco_short_manifest.json",
    ROOT / "benchmarks/smoke/reasonseg_phase1_smoke_refcocoplus_attr_manifest.json",
    ROOT / "benchmarks/smoke/reasonseg_phase1_smoke_refcocog_long_manifest.json",
    ROOT / "benchmarks/smoke/reasonseg_phase1_smoke_noun_regression_manifest.json",
]
SMOKE_PREDICTIONS = [
    ROOT / "benchmarks/smoke/reasonseg_phase1_smoke_eval_predictions.json"
]


def _paper_prediction_payload() -> dict[str, object]:
    return {
        "prediction_schema_version": "1.0.0",
        "predictions": [
            {
                "manifest_file": "reasonseg_phase1_smoke_refcoco_short_manifest.json",
                "entry_index": 0,
                "intersection": 80,
                "union": 100,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcoco_short_manifest.json",
                "entry_index": 1,
                "intersection": 60,
                "union": 100,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcoco_short_manifest.json",
                "entry_index": 2,
                "intersection": 45,
                "union": 50,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcoco_short_manifest.json",
                "entry_index": 3,
                "intersection": 20,
                "union": 40,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcocoplus_attr_manifest.json",
                "entry_index": 0,
                "intersection": 70,
                "union": 100,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcocoplus_attr_manifest.json",
                "entry_index": 1,
                "intersection": 55,
                "union": 100,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcocoplus_attr_manifest.json",
                "entry_index": 2,
                "intersection": 72,
                "union": 90,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcocoplus_attr_manifest.json",
                "entry_index": 3,
                "intersection": 27,
                "union": 45,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcocog_long_manifest.json",
                "entry_index": 0,
                "intersection": 65,
                "union": 100,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcocog_long_manifest.json",
                "entry_index": 1,
                "intersection": 40,
                "union": 80,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcocog_long_manifest.json",
                "entry_index": 2,
                "intersection": 75,
                "union": 100,
            },
            {
                "manifest_file": "reasonseg_phase1_smoke_refcocog_long_manifest.json",
                "entry_index": 3,
                "intersection": 36,
                "union": 60,
            },
            {
                "manifest_file": "reasonseg_refexp_paper_no_target_manifest.json",
                "entry_index": 0,
                "predicted_positive_mask_count": 0,
            },
            {
                "manifest_file": "reasonseg_refexp_paper_no_target_manifest.json",
                "entry_index": 1,
                "predicted_positive_mask_count": 1,
            },
        ],
    }


def test_smoke_evaluation_report_contains_dataset_and_slice_metrics() -> None:
    report = evaluate_predictions(
        benchmark_spec_path=BENCHMARK_SPEC,
        manifest_paths=SMOKE_MANIFESTS,
        prediction_paths=SMOKE_PREDICTIONS,
    )

    slice_metrics = report["slice_metrics"]
    assert tuple(slice_metrics.keys()) == (
        "noun",
        "attribute",
        "relation_action",
        "no_target",
    )
    assert report["dataset_metrics"]["refcocog_umd"]["positive_metrics_by_family"][
        "openworldsam_grounding"
    ]["grounding/mIoU"] == pytest.approx(62.5)
    assert report["dataset_metrics"]["coco_2017_48_17_ov"][
        "positive_metrics_by_family"
    ]["ovsam_instance_mask_iou"]["instance/score"] == pytest.approx(50.0)
    assert report["slice_metrics"]["no_target"]["no_target_metrics"][
        "no_target/rejection_rate"
    ] == pytest.approx(50.0)


def test_csv_render_emits_machine_readable_rows() -> None:
    report = evaluate_predictions(
        benchmark_spec_path=BENCHMARK_SPEC,
        manifest_paths=SMOKE_MANIFESTS,
        prediction_paths=SMOKE_PREDICTIONS,
    )

    rendered = render_csv(report)

    assert "scope_type,scope_name,metric_family,metric_group,metric_key" in rendered
    assert (
        "slice,no_target,no_target,no_target_metrics,no_target/rejection_rate,50.0"
        in rendered
    )


def test_missing_canonical_slice_tag_fails_loudly(tmp_path: Path) -> None:
    manifest_path = (
        ROOT / "benchmarks/smoke/reasonseg_phase1_smoke_refcoco_short_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["slice_tags"] = []
    bad_manifest_path = tmp_path / "bad_manifest.json"
    _ = bad_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match="Each entry must include at least one slice tag"
    ):
        evaluate_predictions(
            benchmark_spec_path=BENCHMARK_SPEC,
            manifest_paths=[bad_manifest_path, *SMOKE_MANIFESTS[1:]],
            prediction_paths=SMOKE_PREDICTIONS,
        )


def test_grounding_accumulator_preserves_root_metric_namespace() -> None:
    accumulator = GroundingAccumulator()

    accumulator.add(intersection=8, union=10)
    accumulator.add(intersection=3, union=5)

    metrics = accumulator.metrics()

    assert tuple(metrics.keys()) == (
        "grounding/cIoU",
        "grounding/mIoU",
        "grounding/precision@0.5",
        "grounding/precision@0.6",
        "grounding/precision@0.7",
        "grounding/precision@0.8",
        "grounding/precision@0.9",
    )
    assert metrics["grounding/cIoU"] == pytest.approx((11 / 15) * 100.0)
    assert metrics["grounding/mIoU"] == pytest.approx(70.0)
    assert metrics["grounding/precision@0.8"] == pytest.approx(50.0)
    assert metrics["grounding/precision@0.9"] == pytest.approx(0.0)


def test_public_benchmark_runner_writes_machine_readable_json(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    prediction_payload = _paper_prediction_payload()
    _ = (pred_root / "predictions.json").write_text(
        json.dumps(prediction_payload), encoding="utf-8"
    )
    output_path = tmp_path / "report.json"

    report = run_benchmark(
        spec_path=PAPER_BENCHMARK_SPEC,
        pred_root=pred_root,
        output_path=output_path,
        output_format="json",
    )

    rendered = json.loads(output_path.read_text(encoding="utf-8"))
    assert rendered["benchmark"]["benchmark_id"] == "reasonseg_refexp_paper_benchmark"
    assert report["suite_metrics"]["positive_metrics_by_family"][
        "openworldsam_grounding"
    ]["grounding/mIoU"] == pytest.approx(
        rendered["suite_metrics"]["positive_metrics_by_family"][
            "openworldsam_grounding"
        ]["grounding/mIoU"]
    )
    assert rendered["slice_metrics"]["no_target"]["no_target_metrics"][
        "no_target/rejection_rate"
    ] == pytest.approx(50.0)


def test_public_benchmark_runner_ignores_non_prediction_json_siblings(
    tmp_path: Path,
) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    prediction_payload = _paper_prediction_payload()
    _ = (pred_root / "predictions.json").write_text(
        json.dumps(prediction_payload), encoding="utf-8"
    )
    _ = (pred_root / "metrics.json").write_text(
        json.dumps({"grounding/mIoU": 62.5}), encoding="utf-8"
    )
    nested = pred_root / "run_0" / "inference"
    nested.mkdir(parents=True)
    _ = (nested / "metrics.json").write_text(
        json.dumps({"grounding/mIoU": 70.0}), encoding="utf-8"
    )
    output_path = tmp_path / "report.json"

    report = run_benchmark(
        spec_path=PAPER_BENCHMARK_SPEC,
        pred_root=pred_root,
        output_path=output_path,
        output_format="json",
    )

    assert output_path.is_file()
    assert report["slice_metrics"]["no_target"]["no_target_metrics"][
        "no_target/rejection_rate"
    ] == pytest.approx(50.0)


def test_public_benchmark_runner_missing_prediction_root_fails_clearly(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "report.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmark/run_benchmark.py"),
        "--spec",
        str(PAPER_BENCHMARK_SPEC),
        "--pred-root",
        str(tmp_path / "missing"),
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode != 0
    assert "Prediction artifact root does not exist" in completed.stderr


def test_public_benchmark_runner_empty_prediction_root_fails_clearly(
    tmp_path: Path,
) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    output_path = tmp_path / "report.json"

    with pytest.raises(ValueError, match="No prediction artifacts found under"):
        run_benchmark(
            spec_path=PAPER_BENCHMARK_SPEC,
            pred_root=pred_root,
            output_path=output_path,
            output_format="json",
        )


def test_public_benchmark_runner_rejects_roots_without_prediction_payloads(
    tmp_path: Path,
) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    _ = (pred_root / "metrics.json").write_text(
        json.dumps({"grounding/mIoU": 62.5}), encoding="utf-8"
    )
    output_path = tmp_path / "report.json"

    with pytest.raises(ValueError, match="No prediction artifacts found under"):
        run_benchmark(
            spec_path=PAPER_BENCHMARK_SPEC,
            pred_root=pred_root,
            output_path=output_path,
            output_format="json",
        )
