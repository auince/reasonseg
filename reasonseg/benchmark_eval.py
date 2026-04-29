# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import cast

from reasonseg.benchmarks.manifest import (
    CANONICAL_SLICE_NAMES,
    validate_manifest,
    validate_spec,
)
from reasonseg.evaluation import GROUNDING_METRIC_FAMILY, GroundingAccumulator


JsonDict = dict[str, object]

DEFAULT_BENCHMARK_SPEC = Path("benchmarks/reasonseg_phase1_benchmark_spec.json")
PREDICTION_SCHEMA_VERSION = "1.0.0"
INSTANCE_FAMILY = "ovsam_instance_mask_iou"


def _load_json(path: Path) -> JsonDict:
    data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    _require(isinstance(data, dict), f"Expected a JSON object at {path}")
    return cast(JsonDict, data)


def _dump_json(payload: object) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_dict(value: object, label: str) -> JsonDict:
    _require(isinstance(value, dict), f"{label} must be a JSON object.")
    return cast(JsonDict, value)


def _as_list(value: object, label: str) -> list[object]:
    _require(isinstance(value, list), f"{label} must be a JSON array.")
    return cast(list[object], value)


def _as_str(value: object, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string.")
    return cast(str, value)


def _as_int(value: object, label: str) -> int:
    _require(isinstance(value, int), f"{label} must be an integer.")
    return cast(int, value)


def _as_number(value: object, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric.",
    )
    return float(cast(int | float, value))


def _mean_or_none(total: float, count: int) -> float | None:
    return None if count == 0 else total / count


def _canonical_slice(entry: JsonDict) -> str:
    slice_tags = [
        _as_str(tag, "entries[].slice_tags[]")
        for tag in _as_list(entry.get("slice_tags"), "entries[].slice_tags")
    ]
    canonical_tags = [tag for tag in slice_tags if tag in CANONICAL_SLICE_NAMES]
    _require(
        len(canonical_tags) == 1,
        "Each entry must contain exactly one canonical slice tag before evaluation.",
    )
    return canonical_tags[0]


def _dataset_metric_family_lookup(spec: JsonDict) -> dict[str, str]:
    datasets = [
        _as_dict(dataset, "datasets[]")
        for dataset in _as_list(spec.get("datasets"), "datasets")
    ]
    return {
        _as_str(dataset.get("dataset_id"), "datasets[].dataset_id"): _as_str(
            dataset.get("metric_family"), "datasets[].metric_family"
        )
        for dataset in datasets
    }


class InstanceAccumulator:
    def __init__(self) -> None:
        self.sum_iou = 0.0
        self.total = 0
        self.sum_score = 0.0
        self.score_total = 0
        self.partition_iou_sums = {"base": 0.0, "novel": 0.0}
        self.partition_iou_counts = {"base": 0, "novel": 0}
        self.partition_score_sums = {"base": 0.0, "novel": 0.0}
        self.partition_score_counts = {"base": 0, "novel": 0}

    def add(
        self,
        *,
        intersection: float,
        union: float,
        class_partition: str,
        classification_correct: bool,
    ) -> None:
        _require(union > 0, "Positive instance predictions must have union > 0.")
        _require(
            0 <= intersection <= union,
            "Instance prediction intersection must be within [0, union].",
        )
        _require(
            class_partition in {"base", "novel"},
            "Instance predictions must declare class_partition as 'base' or 'novel'.",
        )
        iou = intersection / union
        score = 100.0 if classification_correct else 0.0
        self.sum_iou += iou
        self.total += 1
        self.sum_score += score
        self.score_total += 1
        self.partition_iou_sums[class_partition] += iou
        self.partition_iou_counts[class_partition] += 1
        self.partition_score_sums[class_partition] += score
        self.partition_score_counts[class_partition] += 1

    def metrics(self) -> JsonDict:
        _require(
            self.total > 0, "Instance metrics require at least one positive entry."
        )
        results: JsonDict = {"instance/miou": self.sum_iou / self.total}
        base_iou = _mean_or_none(
            self.partition_iou_sums["base"], self.partition_iou_counts["base"]
        )
        novel_iou = _mean_or_none(
            self.partition_iou_sums["novel"], self.partition_iou_counts["novel"]
        )
        if base_iou is not None:
            results["instance/base_iou"] = base_iou
        if novel_iou is not None:
            results["instance/novel_iou"] = novel_iou
        if self.score_total > 0:
            results["instance/score"] = self.sum_score / self.score_total
        base_score = _mean_or_none(
            self.partition_score_sums["base"], self.partition_score_counts["base"]
        )
        novel_score = _mean_or_none(
            self.partition_score_sums["novel"], self.partition_score_counts["novel"]
        )
        if base_score is not None:
            results["instance/base_score"] = base_score
        if novel_score is not None:
            results["instance/novel_score"] = novel_score
        return results


class NoTargetAccumulator:
    def __init__(self) -> None:
        self.total = 0
        self.rejected = 0
        self.false_positive_mask_total = 0

    def add(self, *, predicted_positive_mask_count: int) -> None:
        _require(
            predicted_positive_mask_count >= 0,
            "No-target predictions must use a non-negative predicted_positive_mask_count.",
        )
        self.total += 1
        self.false_positive_mask_total += predicted_positive_mask_count
        if predicted_positive_mask_count == 0:
            self.rejected += 1

    def metrics(self) -> JsonDict:
        _require(
            self.total > 0, "No-target metrics require at least one no-target entry."
        )
        return {
            "no_target/rejection_rate": self.rejected * 100.0 / self.total,
            "no_target/false_positive_rate": (self.total - self.rejected)
            * 100.0
            / self.total,
            "no_target/rejected": self.rejected,
            "no_target/total": self.total,
            "no_target/false_positive_mask_total": self.false_positive_mask_total,
        }


class ScopeAccumulator:
    def __init__(self) -> None:
        self.entry_count = 0
        self.positive_entry_count = 0
        self.no_target_entry_count = 0
        self.family_accumulators: dict[
            str, GroundingAccumulator | InstanceAccumulator
        ] = {}
        self.no_target_accumulator = NoTargetAccumulator()
        self.no_target_families: set[str] = set()

    def _get_grounding_accumulator(self) -> GroundingAccumulator:
        existing = self.family_accumulators.get(GROUNDING_METRIC_FAMILY)
        if existing is None:
            accumulator = GroundingAccumulator()
            self.family_accumulators[GROUNDING_METRIC_FAMILY] = accumulator
            return accumulator
        _require(
            isinstance(existing, GroundingAccumulator),
            "Grounding accumulator type mismatch.",
        )
        return cast(GroundingAccumulator, existing)

    def _get_instance_accumulator(self) -> InstanceAccumulator:
        existing = self.family_accumulators.get(INSTANCE_FAMILY)
        if existing is None:
            accumulator = InstanceAccumulator()
            self.family_accumulators[INSTANCE_FAMILY] = accumulator
            return accumulator
        _require(
            isinstance(existing, InstanceAccumulator),
            "Instance accumulator type mismatch.",
        )
        return cast(InstanceAccumulator, existing)

    def add_positive(
        self,
        *,
        metric_family: str,
        intersection: float,
        union: float,
        class_partition: str | None,
        classification_correct: bool | None,
    ) -> None:
        self.entry_count += 1
        self.positive_entry_count += 1
        if metric_family == GROUNDING_METRIC_FAMILY:
            grounding_accumulator = self._get_grounding_accumulator()
            grounding_accumulator.add(intersection=intersection, union=union)
            return
        _require(
            metric_family == INSTANCE_FAMILY,
            f"Unsupported metric family for positive evaluation: {metric_family}",
        )
        instance_accumulator = self._get_instance_accumulator()
        _require(
            class_partition is not None, "Instance predictions require class_partition."
        )
        _require(
            classification_correct is not None,
            "Instance predictions require classification_correct.",
        )
        concrete_class_partition = cast(str, class_partition)
        concrete_classification_correct = cast(bool, classification_correct)
        instance_accumulator.add(
            intersection=intersection,
            union=union,
            class_partition=concrete_class_partition,
            classification_correct=concrete_classification_correct,
        )

    def add_no_target(
        self, *, metric_family: str, predicted_positive_mask_count: int
    ) -> None:
        self.entry_count += 1
        self.no_target_entry_count += 1
        self.no_target_families.add(metric_family)
        self.no_target_accumulator.add(
            predicted_positive_mask_count=predicted_positive_mask_count
        )

    def render(self) -> JsonDict:
        positive_metrics = {
            metric_family: accumulator.metrics()
            for metric_family, accumulator in self.family_accumulators.items()
        }
        output: JsonDict = {
            "entry_count": self.entry_count,
            "positive_entry_count": self.positive_entry_count,
            "no_target_entry_count": self.no_target_entry_count,
            "positive_metrics_by_family": positive_metrics,
        }
        if self.no_target_entry_count > 0:
            output["no_target_metrics"] = self.no_target_accumulator.metrics()
            output["no_target_metric_families"] = sorted(self.no_target_families)
        return output


def _flatten_paths(grouped_paths: list[list[str]]) -> list[Path]:
    return [Path(path) for group in grouped_paths for path in group]


def _prediction_key(manifest_file: str, entry_index: int) -> tuple[str, int]:
    return (manifest_file, entry_index)


def _load_prediction_lookup(
    prediction_paths: list[Path],
) -> dict[tuple[str, int], JsonDict]:
    lookup: dict[tuple[str, int], JsonDict] = {}
    for prediction_path in prediction_paths:
        payload = _load_json(prediction_path)
        _require(
            _as_str(
                payload.get("prediction_schema_version"),
                "prediction_schema_version",
            )
            == PREDICTION_SCHEMA_VERSION,
            "Prediction schema version mismatch.",
        )
        predictions = [
            _as_dict(item, "predictions[]")
            for item in _as_list(payload.get("predictions"), "predictions")
        ]
        for prediction in predictions:
            manifest_file = _as_str(
                prediction.get("manifest_file"), "predictions[].manifest_file"
            )
            entry_index = _as_int(
                prediction.get("entry_index"), "predictions[].entry_index"
            )
            key = _prediction_key(manifest_file, entry_index)
            _require(key not in lookup, f"Duplicate prediction entry for {key}.")
            lookup[key] = prediction
    return lookup


def _require_all_canonical_slices(slice_metrics: dict[str, ScopeAccumulator]) -> None:
    missing = [
        slice_name
        for slice_name in CANONICAL_SLICE_NAMES
        if slice_metrics[slice_name].entry_count == 0
    ]
    _require(
        not missing,
        "Evaluation input is missing canonical slices: " + ", ".join(missing),
    )


def evaluate_predictions(
    *,
    benchmark_spec_path: Path,
    manifest_paths: list[Path],
    prediction_paths: list[Path],
    allow_partial_slices: bool = False,
) -> JsonDict:
    _require(bool(manifest_paths), "At least one manifest path is required.")
    _require(bool(prediction_paths), "At least one prediction path is required.")
    spec = _load_json(benchmark_spec_path)
    benchmark_summary = validate_spec(spec)
    family_lookup = _dataset_metric_family_lookup(spec)
    prediction_lookup = _load_prediction_lookup(prediction_paths)

    suite_accumulator = ScopeAccumulator()
    dataset_accumulators: dict[str, ScopeAccumulator] = {}
    slice_accumulators = {
        slice_name: ScopeAccumulator() for slice_name in CANONICAL_SLICE_NAMES
    }
    manifest_summaries: list[JsonDict] = []
    seen_prediction_keys: set[tuple[str, int]] = set()
    manifest_names: set[str] = set()

    for manifest_path in manifest_paths:
        manifest_file = manifest_path.name
        _require(
            manifest_file not in manifest_names,
            f"Manifest file names must be unique across inputs; found duplicate {manifest_file}.",
        )
        manifest_names.add(manifest_file)
        manifest = _load_json(manifest_path)
        manifest_summary = validate_manifest(spec, manifest)
        manifest_summaries.append(
            {
                "manifest_file": manifest_file,
                "entry_count": manifest_summary["entry_count"],
                "slice_counts": manifest_summary["slice_counts"],
            }
        )
        entries = [
            _as_dict(entry, "entries[]")
            for entry in _as_list(manifest.get("entries"), "entries")
        ]
        for entry_index, entry in enumerate(entries):
            dataset_id = _as_str(entry.get("dataset_id"), "entries[].dataset_id")
            metric_family = family_lookup[dataset_id]
            canonical_slice = _canonical_slice(entry)
            prediction_key = _prediction_key(manifest_file, entry_index)
            _require(
                prediction_key in prediction_lookup,
                f"Missing prediction for {manifest_file} entry_index={entry_index}.",
            )
            prediction = prediction_lookup[prediction_key]
            seen_prediction_keys.add(prediction_key)
            dataset_accumulator = dataset_accumulators.setdefault(
                dataset_id, ScopeAccumulator()
            )
            slice_accumulator = slice_accumulators[canonical_slice]

            if canonical_slice == "no_target":
                predicted_positive_mask_count = _as_int(
                    prediction.get("predicted_positive_mask_count"),
                    "predictions[].predicted_positive_mask_count",
                )
                suite_accumulator.add_no_target(
                    metric_family=metric_family,
                    predicted_positive_mask_count=predicted_positive_mask_count,
                )
                dataset_accumulator.add_no_target(
                    metric_family=metric_family,
                    predicted_positive_mask_count=predicted_positive_mask_count,
                )
                slice_accumulator.add_no_target(
                    metric_family=metric_family,
                    predicted_positive_mask_count=predicted_positive_mask_count,
                )
                continue

            intersection = _as_number(
                prediction.get("intersection"), "predictions[].intersection"
            )
            union = _as_number(prediction.get("union"), "predictions[].union")
            class_partition = cast(str | None, prediction.get("class_partition"))
            classification_correct = cast(
                bool | None, prediction.get("classification_correct")
            )
            suite_accumulator.add_positive(
                metric_family=metric_family,
                intersection=intersection,
                union=union,
                class_partition=class_partition,
                classification_correct=classification_correct,
            )
            dataset_accumulator.add_positive(
                metric_family=metric_family,
                intersection=intersection,
                union=union,
                class_partition=class_partition,
                classification_correct=classification_correct,
            )
            slice_accumulator.add_positive(
                metric_family=metric_family,
                intersection=intersection,
                union=union,
                class_partition=class_partition,
                classification_correct=classification_correct,
            )

    unexpected_predictions = sorted(
        set(prediction_lookup).difference(seen_prediction_keys)
    )
    _require(
        not unexpected_predictions,
        "Predictions include entries not present in the supplied manifests: "
        + ", ".join(
            f"{manifest_file}:{entry_index}"
            for manifest_file, entry_index in unexpected_predictions
        ),
    )
    if not allow_partial_slices:
        _require_all_canonical_slices(slice_accumulators)

    return {
        "benchmark": benchmark_summary,
        "manifests": manifest_summaries,
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "suite_metrics": suite_accumulator.render(),
        "dataset_metrics": {
            dataset_id: accumulator.render()
            for dataset_id, accumulator in sorted(dataset_accumulators.items())
        },
        "slice_metrics": {
            slice_name: slice_accumulators[slice_name].render()
            for slice_name in CANONICAL_SLICE_NAMES
        },
    }


def render_csv(report: JsonDict) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "scope_type",
        "scope_name",
        "metric_family",
        "metric_group",
        "metric_key",
        "metric_value",
        "entry_count",
        "positive_entry_count",
        "no_target_entry_count",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    def write_scope(scope_type: str, scope_name: str, scope_payload: JsonDict) -> None:
        entry_count = _as_int(
            scope_payload.get("entry_count"), f"{scope_name}.entry_count"
        )
        positive_entry_count = _as_int(
            scope_payload.get("positive_entry_count"),
            f"{scope_name}.positive_entry_count",
        )
        no_target_entry_count = _as_int(
            scope_payload.get("no_target_entry_count"),
            f"{scope_name}.no_target_entry_count",
        )
        positive_metrics = _as_dict(
            scope_payload.get("positive_metrics_by_family"),
            f"{scope_name}.positive_metrics_by_family",
        )
        for metric_family, metrics in sorted(positive_metrics.items()):
            metric_payload = _as_dict(metrics, f"{scope_name}.{metric_family}")
            for metric_key, metric_value in sorted(metric_payload.items()):
                writer.writerow(
                    {
                        "scope_type": scope_type,
                        "scope_name": scope_name,
                        "metric_family": metric_family,
                        "metric_group": "positive_metrics",
                        "metric_key": metric_key,
                        "metric_value": metric_value,
                        "entry_count": entry_count,
                        "positive_entry_count": positive_entry_count,
                        "no_target_entry_count": no_target_entry_count,
                    }
                )
        no_target_metrics = scope_payload.get("no_target_metrics")
        if isinstance(no_target_metrics, dict):
            for metric_key, metric_value in sorted(no_target_metrics.items()):
                writer.writerow(
                    {
                        "scope_type": scope_type,
                        "scope_name": scope_name,
                        "metric_family": "no_target",
                        "metric_group": "no_target_metrics",
                        "metric_key": metric_key,
                        "metric_value": metric_value,
                        "entry_count": entry_count,
                        "positive_entry_count": positive_entry_count,
                        "no_target_entry_count": no_target_entry_count,
                    }
                )

    write_scope(
        "suite", "suite", _as_dict(report.get("suite_metrics"), "suite_metrics")
    )
    dataset_metrics = _as_dict(report.get("dataset_metrics"), "dataset_metrics")
    for dataset_id, payload in sorted(dataset_metrics.items()):
        write_scope(
            "dataset", dataset_id, _as_dict(payload, f"dataset_metrics.{dataset_id}")
        )
    slice_metrics = _as_dict(report.get("slice_metrics"), "slice_metrics")
    for slice_name, payload in sorted(slice_metrics.items()):
        write_scope(
            "slice", slice_name, _as_dict(payload, f"slice_metrics.{slice_name}")
        )
    return buffer.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate local ReasonSeg benchmark predictions against manifest-driven contracts."
    )
    parser.add_argument(
        "--benchmark-spec",
        type=Path,
        default=DEFAULT_BENCHMARK_SPEC,
        help="Path to the phase-1 benchmark spec JSON.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        nargs="+",
        required=True,
        help="One or more materialized benchmark manifest paths. Repeatable.",
    )
    parser.add_argument(
        "--predictions",
        action="append",
        nargs="+",
        required=True,
        help="One or more normalized prediction payload JSON paths. Repeatable.",
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "csv"),
        default="json",
        help="Machine-readable report format.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional file path for the rendered report. Defaults to stdout.",
    )
    parser.add_argument(
        "--allow-partial-slices",
        action="store_true",
        help="Allow reports that do not cover all four canonical slices.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = evaluate_predictions(
        benchmark_spec_path=args.benchmark_spec,
        manifest_paths=_flatten_paths(args.manifest),
        prediction_paths=_flatten_paths(args.predictions),
        allow_partial_slices=args.allow_partial_slices,
    )
    rendered = render_csv(report) if args.output_format == "csv" else _dump_json(report)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        _ = args.output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
