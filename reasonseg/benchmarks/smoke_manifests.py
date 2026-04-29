from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .manifest import validate_manifest, validate_spec


JsonDict = dict[str, object]


DEFAULT_BENCHMARK_SPEC = Path("benchmarks/reasonseg_phase1_benchmark_spec.json")
DEFAULT_SOURCE_SPEC = Path("benchmarks/reasonseg_phase1_smoke_source_records.json")
DEFAULT_OUTPUT_DIR = Path("benchmarks/smoke")


def _load_json(path: Path) -> JsonDict:
    data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return cast(JsonDict, data)


def _dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def _stringify_ids(values: list[object]) -> list[str]:
    return [str(value) for value in values]


def _dataset_lookup(benchmark_spec: JsonDict) -> dict[str, JsonDict]:
    datasets = [
        _as_dict(dataset, "datasets[]")
        for dataset in _as_list(benchmark_spec.get("datasets"), "datasets")
    ]
    return {
        _as_str(dataset.get("dataset_id"), "datasets[].dataset_id"): dataset
        for dataset in datasets
    }


def _materialized_manifest_name(subset_id: str) -> str:
    return f"reasonseg_phase1_smoke_{subset_id}_manifest.json"


def _build_manifest_entry(raw_entry: JsonDict, dataset: JsonDict) -> JsonDict:
    return {
        "dataset_id": _as_str(dataset.get("dataset_id"), "datasets[].dataset_id"),
        "dataset_role": _as_str(dataset.get("role"), "datasets[].role"),
        "split": _as_str(raw_entry.get("split"), "entries[].split"),
        "image_id": raw_entry.get("image_id"),
        "query_text": _as_str(raw_entry.get("query_text"), "entries[].query_text"),
        "query_struct": _as_dict(
            raw_entry.get("query_struct"), "entries[].query_struct"
        ),
        "requested_target": _as_str(
            raw_entry.get("requested_target"), "entries[].requested_target"
        ),
        "slice_tags": _as_list(raw_entry.get("slice_tags"), "entries[].slice_tags"),
        "image_category_names": _as_list(
            raw_entry.get("image_category_names"), "entries[].image_category_names"
        ),
        "positive_mask_count": raw_entry.get("positive_mask_count"),
        "source_annotation_ids": _as_list(
            raw_entry.get("source_annotation_ids"), "entries[].source_annotation_ids"
        ),
    }


def _build_provenance_entry(
    subset_id: str,
    manifest_name: str,
    index: int,
    raw_entry: JsonDict,
    dataset: JsonDict,
) -> JsonDict:
    return {
        "subset_id": subset_id,
        "manifest_file": manifest_name,
        "entry_index": index,
        "dataset_id": _as_str(dataset.get("dataset_id"), "datasets[].dataset_id"),
        "dataset_role": _as_str(dataset.get("role"), "datasets[].role"),
        "source_split": _as_str(raw_entry.get("split"), "entries[].split"),
        "image_id": raw_entry.get("image_id"),
        "query_text": _as_str(raw_entry.get("query_text"), "entries[].query_text"),
        "requested_target": _as_str(
            raw_entry.get("requested_target"), "entries[].requested_target"
        ),
        "source_ref_id": raw_entry.get("source_ref_id"),
        "source_query_id": raw_entry.get("source_query_id"),
        "source_query_origin": _as_str(
            raw_entry.get("source_query_origin"), "entries[].source_query_origin"
        ),
        "source_annotation_ids": _as_list(
            raw_entry.get("source_annotation_ids"), "entries[].source_annotation_ids"
        ),
        "slice_tags": _as_list(raw_entry.get("slice_tags"), "entries[].slice_tags"),
    }


def _build_exclusion_rule(
    subset: JsonDict, dataset: JsonDict, manifest_name: str
) -> JsonDict:
    train_splits = _stringify_ids(
        _as_list(subset.get("train_splits"), "subsets[].train_splits")
    )
    eval_splits = _stringify_ids(
        _as_list(subset.get("eval_splits"), "subsets[].eval_splits")
    )
    overlap = sorted(set(train_splits).intersection(eval_splits))
    _require(not overlap, f"Subset has overlapping train/eval split labels: {overlap}")
    return {
        "subset_id": _as_str(subset.get("subset_id"), "subsets[].subset_id"),
        "dataset_id": _as_str(dataset.get("dataset_id"), "datasets[].dataset_id"),
        "manifest_file": manifest_name,
        "train_splits": train_splits,
        "eval_splits": eval_splits,
        "overlap_policy": "forbid_shared_image_ids",
        "leakage_key": "image_id",
    }


def generate_manifests(
    benchmark_spec_path: Path, source_spec_path: Path, output_dir: Path
) -> JsonDict:
    benchmark_spec = _load_json(benchmark_spec_path)
    benchmark_summary = validate_spec(benchmark_spec)
    source_spec = _load_json(source_spec_path)
    _require(
        source_spec.get("benchmark_spec_version")
        == benchmark_spec.get("schema_version"),
        "Smoke source records must match the benchmark spec schema_version.",
    )

    dataset_lookup = _dataset_lookup(benchmark_spec)
    subsets = [
        _as_dict(subset, "subsets[]")
        for subset in _as_list(source_spec.get("subsets"), "subsets")
    ]
    _require(bool(subsets), "Smoke source records must define at least one subset.")

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_entries: list[JsonDict] = []
    exclusion_rules: list[JsonDict] = []
    subset_summaries: list[JsonDict] = []

    for subset in subsets:
        subset_id = _as_str(subset.get("subset_id"), "subsets[].subset_id")
        dataset_id = _as_str(subset.get("dataset_id"), "subsets[].dataset_id")
        _require(dataset_id in dataset_lookup, f"Unknown dataset_id: {dataset_id}")
        dataset = dataset_lookup[dataset_id]
        manifest_name = _materialized_manifest_name(subset_id)

        raw_entries = [
            _as_dict(entry, "subsets[].entries[]")
            for entry in _as_list(subset.get("entries"), "subsets[].entries")
        ]
        _require(
            bool(raw_entries), f"Subset '{subset_id}' must include at least one entry."
        )

        manifest_entries = [
            _build_manifest_entry(raw_entry, dataset) for raw_entry in raw_entries
        ]
        manifest = {
            "manifest_kind": "phase1_materialized_slice_manifest",
            "benchmark_spec_version": benchmark_spec["schema_version"],
            "entries": manifest_entries,
        }
        manifest_summary = validate_manifest(benchmark_spec, manifest)

        manifest_path = output_dir / manifest_name
        _dump_json(manifest_path, manifest)

        for index, raw_entry in enumerate(raw_entries):
            provenance_entries.append(
                _build_provenance_entry(
                    subset_id=subset_id,
                    manifest_name=manifest_name,
                    index=index,
                    raw_entry=raw_entry,
                    dataset=dataset,
                )
            )

        exclusion_rules.append(_build_exclusion_rule(subset, dataset, manifest_name))
        subset_summaries.append(
            {
                "subset_id": subset_id,
                "dataset_id": dataset_id,
                "manifest_file": manifest_name,
                "entry_count": manifest_summary["entry_count"],
                "slice_counts": manifest_summary["slice_counts"],
                "train_splits": _stringify_ids(
                    _as_list(subset.get("train_splits"), "subsets[].train_splits")
                ),
                "eval_splits": _stringify_ids(
                    _as_list(subset.get("eval_splits"), "subsets[].eval_splits")
                ),
            }
        )

    provenance_manifest = {
        "manifest_kind": "reasonseg_phase1_smoke_provenance",
        "benchmark_spec_version": benchmark_spec["schema_version"],
        "entries": provenance_entries,
    }
    _dump_json(
        output_dir / "reasonseg_phase1_smoke_provenance_manifest.json",
        provenance_manifest,
    )

    exclusion_manifest = {
        "manifest_kind": "reasonseg_phase1_smoke_exclusion_rules",
        "benchmark_spec_version": benchmark_spec["schema_version"],
        "rules": exclusion_rules,
    }
    exclusion_path = output_dir / "reasonseg_phase1_smoke_exclusion_rules.json"
    _dump_json(exclusion_path, exclusion_manifest)

    leakage_summary = check_leakage(
        manifest_dir=output_dir, exclusion_rules_path=exclusion_path
    )
    return {
        "benchmark": benchmark_summary,
        "generated_manifest_dir": str(output_dir),
        "subset_count": len(subset_summaries),
        "subsets": subset_summaries,
        "provenance_entries": len(provenance_entries),
        "leakage_check": leakage_summary,
    }


def _collect_manifest_image_ids(
    manifest: JsonDict, allowed_splits: set[str]
) -> set[str]:
    entries = [
        _as_dict(entry, "entries[]")
        for entry in _as_list(manifest.get("entries"), "entries")
    ]
    image_ids: set[str] = set()
    for entry in entries:
        split = _as_str(entry.get("split"), "entries[].split")
        if split in allowed_splits:
            image_ids.add(str(entry.get("image_id")))
    return image_ids


def _load_manifest(path: Path) -> JsonDict:
    manifest = _load_json(path)
    _require(
        manifest.get("manifest_kind") == "phase1_materialized_slice_manifest",
        f"Expected a materialized smoke manifest at {path}",
    )
    return manifest


def check_leakage(
    manifest_dir: Path | None,
    exclusion_rules_path: Path | None,
    probe_file: Path | None = None,
) -> JsonDict:
    if probe_file is not None:
        probe = _load_json(probe_file)
        train_ids = set(
            _stringify_ids(_as_list(probe.get("train_image_ids"), "train_image_ids"))
        )
        eval_ids = set(
            _stringify_ids(_as_list(probe.get("eval_image_ids"), "eval_image_ids"))
        )
        overlap = sorted(train_ids.intersection(eval_ids))
        _require(
            not overlap,
            "Leakage detected between smoke-train and smoke-eval image ids: "
            + ", ".join(overlap),
        )
        return {
            "checked_rule_count": 1,
            "overlap_count": 0,
            "overlap_image_ids": [],
            "mode": "probe",
        }

    _require(
        manifest_dir is not None, "manifest_dir is required when probe_file is absent."
    )
    _require(
        exclusion_rules_path is not None,
        "exclusion_rules_path is required when probe_file is absent.",
    )
    concrete_manifest_dir = cast(Path, manifest_dir)
    concrete_exclusion_rules_path = cast(Path, exclusion_rules_path)
    exclusion_rules = _load_json(concrete_exclusion_rules_path)
    _require(
        exclusion_rules.get("manifest_kind")
        == "reasonseg_phase1_smoke_exclusion_rules",
        "Expected a smoke exclusion rules manifest.",
    )
    rules = [
        _as_dict(rule, "rules[]")
        for rule in _as_list(exclusion_rules.get("rules"), "rules")
    ]
    _require(bool(rules), "Exclusion rules manifest must include at least one rule.")

    overlaps: set[str] = set()
    for rule in rules:
        manifest_path = concrete_manifest_dir / _as_str(
            rule.get("manifest_file"), "rules[].manifest_file"
        )
        manifest = _load_manifest(manifest_path)
        train_ids = _collect_manifest_image_ids(
            manifest,
            set(
                _stringify_ids(
                    _as_list(rule.get("train_splits"), "rules[].train_splits")
                )
            ),
        )
        eval_ids = _collect_manifest_image_ids(
            manifest,
            set(
                _stringify_ids(_as_list(rule.get("eval_splits"), "rules[].eval_splits"))
            ),
        )
        overlaps.update(train_ids.intersection(eval_ids))

    overlap_list = sorted(overlaps)
    _require(
        not overlap_list,
        "Leakage detected between smoke-train and smoke-eval image ids: "
        + ", ".join(overlap_list),
    )
    return {
        "checked_rule_count": len(rules),
        "overlap_count": 0,
        "overlap_image_ids": [],
        "mode": "manifest_dir",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and validate ReasonSeg smoke manifests."
    )
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--source-spec", type=Path, default=DEFAULT_SOURCE_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-file", type=Path)
    args = cast(argparse.Namespace, parser.parse_args())

    if args.probe_file is not None:
        payload = check_leakage(
            manifest_dir=None,
            exclusion_rules_path=None,
            probe_file=cast(Path, args.probe_file),
        )
    else:
        payload = generate_manifests(
            benchmark_spec_path=cast(Path, args.benchmark_spec),
            source_spec_path=cast(Path, args.source_spec),
            output_dir=cast(Path, args.output_dir),
        )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
