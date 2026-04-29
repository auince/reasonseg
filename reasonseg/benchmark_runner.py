from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from .benchmark_eval import evaluate_predictions, render_csv


JsonDict = dict[str, object]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> JsonDict:
    data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    _require(isinstance(data, dict), f"Expected a JSON object at {path}")
    return cast(JsonDict, data)


def _as_dict(value: object, label: str) -> JsonDict:
    _require(isinstance(value, dict), f"{label} must be a JSON object.")
    return cast(JsonDict, value)


def _as_list(value: object, label: str) -> list[object]:
    _require(isinstance(value, list), f"{label} must be a JSON array.")
    return cast(list[object], value)


def _as_str(value: object, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string.")
    return cast(str, value)


def _resolve_manifest_paths(spec_path: Path) -> list[Path]:
    spec = _load_json(spec_path)
    runner = _as_dict(spec.get("runner"), "runner")
    manifest_values = _as_list(runner.get("manifests"), "runner.manifests")
    manifests = [
        (spec_path.parent / _as_str(value, "runner.manifests[]")).resolve()
        for value in manifest_values
    ]
    _require(
        bool(manifests), "Benchmark spec must declare at least one runner manifest."
    )
    missing = [str(path) for path in manifests if not path.is_file()]
    _require(
        not missing,
        "Benchmark spec references missing manifest artifacts: " + ", ".join(missing),
    )
    return manifests


def _resolve_prediction_paths(pred_root: Path) -> list[Path]:
    _require(
        pred_root.exists(),
        f"Prediction artifact root does not exist: {pred_root}",
    )
    _require(
        pred_root.is_dir(),
        f"Prediction artifact root is not a directory: {pred_root}",
    )
    prediction_paths: list[Path] = []
    for path in sorted(pred_root.rglob("*.json")):
        if not path.is_file():
            continue
        payload = _load_json(path)
        if path.name == "predictions.json" or (
            "prediction_schema_version" in payload and "predictions" in payload
        ):
            prediction_paths.append(path)
    _require(
        bool(prediction_paths),
        f"No prediction artifacts found under {pred_root}",
    )
    return prediction_paths


def run_benchmark(
    *,
    spec_path: Path,
    pred_root: Path,
    output_path: Path,
    output_format: str,
) -> JsonDict:
    report = evaluate_predictions(
        benchmark_spec_path=spec_path,
        manifest_paths=_resolve_manifest_paths(spec_path),
        prediction_paths=_resolve_prediction_paths(pred_root),
    )
    rendered = (
        render_csv(report)
        if output_format == "csv"
        else json.dumps(report, indent=2) + "\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(rendered, encoding="utf-8")
    return report
