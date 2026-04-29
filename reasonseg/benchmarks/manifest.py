from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast


CANONICAL_QUERY_KEYS = (
    "target",
    "attributes",
    "relations",
    "actions",
    "negatives",
    "exists",
)
CANONICAL_SLICE_NAMES = ("noun", "attribute", "relation_action", "no_target")


JsonDict = dict[str, object]


def _load_json(path: Path) -> JsonDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return cast(JsonDict, data)


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


def _as_bool(value: object, label: str) -> bool:
    _require(isinstance(value, bool), f"{label} must be a boolean.")
    return cast(bool, value)


def _as_int(value: object, label: str) -> int:
    _require(isinstance(value, int), f"{label} must be an integer.")
    return cast(int, value)


def validate_spec(spec: JsonDict) -> JsonDict:
    query_contract = _as_dict(spec.get("query_contract"), "query_contract")
    ordered_keys = tuple(
        _as_str(item, "query_contract.ordered_keys[]")
        for item in _as_list(
            query_contract.get("ordered_keys"), "query_contract.ordered_keys"
        )
    )
    _require(
        ordered_keys == CANONICAL_QUERY_KEYS,
        "Spec query_contract.ordered_keys must match the Task 2 contract exactly.",
    )

    datasets = [
        _as_dict(item, "datasets[]")
        for item in _as_list(spec.get("datasets"), "datasets")
    ]
    _require(bool(datasets), "Spec must define at least one dataset.")
    dataset_ids = {
        _as_str(dataset.get("dataset_id"), "datasets[].dataset_id")
        for dataset in datasets
    }
    required_refexp_dataset_ids = {
        "refcoco_unc",
        "refcoco_plus_unc",
        "refcocog_umd",
    }
    _require(
        required_refexp_dataset_ids.issubset(dataset_ids),
        "Spec is missing one or more required RefCOCO-family dataset definitions.",
    )

    default_noun_regression = [
        _as_str(dataset.get("dataset_id"), "datasets[].dataset_id")
        for dataset in datasets
        if _as_str(dataset.get("role"), "datasets[].role") == "noun_regression"
        and bool(dataset.get("phase1_default"))
    ]
    if any(
        _as_str(dataset.get("role"), "datasets[].role") == "noun_regression"
        for dataset in datasets
    ):
        _require(
            {"coco_2017_48_17_ov", "lvis_v1_val_noun_regression_alt"}.issubset(
                dataset_ids
            ),
            "Specs with noun_regression datasets must define both COCO 48/17 and LVIS alternatives.",
        )
        _require(
            default_noun_regression == ["coco_2017_48_17_ov"],
            "Exactly one noun_regression dataset must be phase1_default, and it must be COCO 48/17.",
        )

    derived_slices = _as_dict(spec.get("derived_slices"), "derived_slices")
    _require(
        set(CANONICAL_SLICE_NAMES).issubset(derived_slices.keys()),
        "Spec must define noun, attribute, relation_action, and no_target derived slices.",
    )
    for slice_name in CANONICAL_SLICE_NAMES:
        slice_spec = _as_dict(
            derived_slices[slice_name], f"derived_slices.{slice_name}"
        )
        minimum_smoke_examples = _as_int(
            slice_spec.get("minimum_smoke_examples"),
            f"derived_slices.{slice_name}.minimum_smoke_examples",
        )
        _require(
            minimum_smoke_examples >= 50,
            f"Slice '{slice_name}' must reserve at least 50 smoke examples.",
        )

    no_target_rule = _as_dict(
        _as_dict(derived_slices["no_target"], "derived_slices.no_target").get("rule"),
        "derived_slices.no_target.rule",
    )
    _require(
        no_target_rule.get("exists") is False,
        "no_target slice must require exists=false.",
    )
    _require(
        no_target_rule.get("target") == "must_be_null",
        "no_target slice must require query_struct.target to be null.",
    )
    _require(
        "absent_object"
        in [
            _as_str(item, "derived_slices.no_target.rule.negatives_must_include[]")
            for item in _as_list(
                no_target_rule.get("negatives_must_include"),
                "derived_slices.no_target.rule.negatives_must_include",
            )
        ],
        "no_target slice must require the absent_object negative flag.",
    )

    materialized_contract = _as_dict(
        spec.get("materialized_manifest_contract"), "materialized_manifest_contract"
    )
    required_entry_fields = {
        _as_str(item, "materialized_manifest_contract.required_entry_fields[]")
        for item in _as_list(
            materialized_contract.get("required_entry_fields"),
            "materialized_manifest_contract.required_entry_fields",
        )
    }
    _require(
        "requested_target" in required_entry_fields,
        "Materialized manifest contract must preserve requested_target for no-target entries.",
    )
    _require(
        "positive_mask_count" in required_entry_fields,
        "Materialized manifest contract must include positive_mask_count for absence validation.",
    )

    return {
        "benchmark_id": _as_str(spec.get("benchmark_id"), "benchmark_id"),
        "schema_version": _as_str(spec.get("schema_version"), "schema_version"),
        "dataset_count": len(datasets),
        "default_noun_regression": (
            default_noun_regression[0] if default_noun_regression else None
        ),
        "canonical_slices": list(CANONICAL_SLICE_NAMES),
    }


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _validate_query_struct(entry: JsonDict) -> JsonDict:
    query_struct = _as_dict(entry.get("query_struct"), "query_struct")
    _require(
        tuple(query_struct.keys()) == CANONICAL_QUERY_KEYS,
        "query_struct keys must preserve Task 2 ordering exactly.",
    )
    _require(
        isinstance(query_struct["attributes"], list),
        "query_struct.attributes must be a list.",
    )
    _require(
        isinstance(query_struct["relations"], list),
        "query_struct.relations must be a list.",
    )
    _require(
        isinstance(query_struct["actions"], list),
        "query_struct.actions must be a list.",
    )
    _require(
        isinstance(query_struct["negatives"], list),
        "query_struct.negatives must be a list.",
    )
    _require(
        isinstance(query_struct["exists"], bool), "query_struct.exists must be boolean."
    )
    return query_struct


def _validate_slice_semantics(entry: JsonDict) -> None:
    query_struct = _validate_query_struct(entry)
    slice_tags = [
        _as_str(item, "slice_tags[]")
        for item in _as_list(entry.get("slice_tags"), "slice_tags")
    ]
    _require(bool(slice_tags), "Each entry must include at least one slice tag.")
    canonical_slices = [tag for tag in slice_tags if tag in CANONICAL_SLICE_NAMES]
    _require(
        len(canonical_slices) == 1,
        "Each entry must contain exactly one canonical slice tag.",
    )

    canonical_slice = canonical_slices[0]
    requested_target = _as_str(entry.get("requested_target"), "requested_target")
    image_categories = [
        _normalize_name(_as_str(name, "image_category_names[]"))
        for name in _as_list(entry.get("image_category_names"), "image_category_names")
    ]
    positive_mask_count = _as_int(
        entry.get("positive_mask_count"), "positive_mask_count"
    )

    _require(
        bool(requested_target.strip()),
        "Each entry must include a non-empty requested_target.",
    )
    _require(
        positive_mask_count >= 0,
        "Each entry must include non-negative positive_mask_count.",
    )

    if canonical_slice == "noun":
        _require(query_struct["exists"] is True, "noun entries must have exists=true.")
        _require(
            query_struct["target"] == requested_target,
            "noun requested_target must match query_struct.target.",
        )
        _require(
            not query_struct["attributes"]
            and not query_struct["relations"]
            and not query_struct["actions"],
            "noun entries cannot include attributes, relations, or actions.",
        )
        _require(
            positive_mask_count > 0,
            "noun entries must have at least one positive mask.",
        )

    elif canonical_slice == "attribute":
        _require(
            query_struct["exists"] is True, "attribute entries must have exists=true."
        )
        _require(
            query_struct["target"] == requested_target,
            "attribute requested_target must match query_struct.target.",
        )
        _require(
            len(_as_list(query_struct["attributes"], "query_struct.attributes")) >= 1,
            "attribute entries must include at least one attribute.",
        )
        _require(
            not query_struct["relations"] and not query_struct["actions"],
            "attribute entries cannot include relations or actions.",
        )
        _require(
            positive_mask_count > 0,
            "attribute entries must have at least one positive mask.",
        )

    elif canonical_slice == "relation_action":
        _require(
            query_struct["exists"] is True,
            "relation_action entries must have exists=true.",
        )
        _require(
            query_struct["target"] == requested_target,
            "relation_action requested_target must match query_struct.target.",
        )
        _require(
            bool(query_struct["relations"] or query_struct["actions"]),
            "relation_action entries must include at least one relation or action.",
        )
        _require(
            positive_mask_count > 0,
            "relation_action entries must have at least one positive mask.",
        )

    else:
        _require(
            query_struct["exists"] is False, "no_target entries must have exists=false."
        )
        _require(
            query_struct["target"] is None,
            "no_target entries must set query_struct.target to null.",
        )
        _require(
            "absent_object"
            in [
                _as_str(item, "query_struct.negatives[]")
                for item in _as_list(
                    query_struct["negatives"], "query_struct.negatives"
                )
            ],
            "no_target entries must include absent_object in negatives.",
        )
        _require(
            positive_mask_count == 0, "no_target entries must have zero positive masks."
        )
        _require(
            _normalize_name(requested_target) not in image_categories,
            "no_target requested_target must be absent from image_category_names.",
        )


def validate_manifest(spec: JsonDict, manifest: JsonDict) -> JsonDict:
    _ = validate_spec(spec)
    _require(
        manifest.get("manifest_kind") == "phase1_materialized_slice_manifest",
        "Manifest kind must be phase1_materialized_slice_manifest.",
    )
    _require(
        manifest.get("benchmark_spec_version") == spec.get("schema_version"),
        "Manifest benchmark_spec_version must match spec schema_version.",
    )

    dataset_ids = {
        _as_str(dataset.get("dataset_id"), "datasets[].dataset_id")
        for dataset in [
            _as_dict(item, "datasets[]")
            for item in _as_list(spec.get("datasets"), "datasets")
        ]
    }
    entries = [
        _as_dict(item, "entries[]")
        for item in _as_list(manifest.get("entries"), "entries")
    ]
    _require(bool(entries), "Manifest must contain at least one entry.")

    slice_counts = {slice_name: 0 for slice_name in CANONICAL_SLICE_NAMES}
    for entry in entries:
        dataset_id = _as_str(entry.get("dataset_id"), "entries[].dataset_id")
        _require(
            dataset_id in dataset_ids,
            f"Unknown dataset_id in manifest: {dataset_id}",
        )
        _validate_slice_semantics(entry)
        for tag in [
            _as_str(item, "entries[].slice_tags[]")
            for item in _as_list(entry.get("slice_tags"), "entries[].slice_tags")
        ]:
            if tag in slice_counts:
                slice_counts[tag] += 1

    return {
        "entry_count": len(entries),
        "slice_counts": slice_counts,
        "no_target_absence_checks": slice_counts["no_target"],
    }


def render_summary(spec: JsonDict) -> JsonDict:
    _ = validate_spec(spec)
    datasets = [
        _as_dict(item, "datasets[]")
        for item in _as_list(spec.get("datasets"), "datasets")
    ]
    derived_slices = _as_dict(spec.get("derived_slices"), "derived_slices")
    return {
        "benchmark_id": _as_str(spec.get("benchmark_id"), "benchmark_id"),
        "schema_version": _as_str(spec.get("schema_version"), "schema_version"),
        "datasets": [
            {
                "dataset_id": _as_str(
                    dataset.get("dataset_id"), "datasets[].dataset_id"
                ),
                "display_name": _as_str(
                    dataset.get("display_name"), "datasets[].display_name"
                ),
                "role": _as_str(dataset.get("role"), "datasets[].role"),
                "eval_splits": [
                    _as_str(item, "datasets[].phase1_default_eval_splits[]")
                    for item in _as_list(
                        dataset.get("phase1_default_eval_splits"),
                        "datasets[].phase1_default_eval_splits",
                    )
                ],
                "metric_family": _as_str(
                    dataset.get("metric_family"), "datasets[].metric_family"
                ),
                "slice_priority": [
                    _as_str(item, "datasets[].derived_slice_priority[]")
                    for item in _as_list(
                        dataset.get("derived_slice_priority"),
                        "datasets[].derived_slice_priority",
                    )
                ],
            }
            for dataset in datasets
        ],
        "derived_slices": {
            slice_name: {
                "minimum_smoke_examples": _as_int(
                    _as_dict(
                        derived_slices[slice_name], f"derived_slices.{slice_name}"
                    ).get("minimum_smoke_examples"),
                    f"derived_slices.{slice_name}.minimum_smoke_examples",
                ),
                "preferred_sources": [
                    _as_str(item, f"derived_slices.{slice_name}.preferred_sources[]")
                    for item in _as_list(
                        _as_dict(
                            derived_slices[slice_name], f"derived_slices.{slice_name}"
                        ).get("preferred_sources"),
                        f"derived_slices.{slice_name}.preferred_sources",
                    )
                ],
            }
            for slice_name in CANONICAL_SLICE_NAMES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize ReasonSeg phase-1 benchmark artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_spec_parser = subparsers.add_parser(
        "validate-spec", help="Validate the frozen benchmark spec."
    )
    validate_spec_parser.add_argument("--spec", type=Path, required=True)

    render_summary_parser = subparsers.add_parser(
        "render-summary", help="Render a normalized machine-readable benchmark summary."
    )
    render_summary_parser.add_argument("--spec", type=Path, required=True)
    render_summary_parser.add_argument("--output", type=Path)

    validate_manifest_parser = subparsers.add_parser(
        "validate-manifest",
        help="Validate a materialized slice manifest against the frozen spec.",
    )
    validate_manifest_parser.add_argument("--spec", type=Path, required=True)
    validate_manifest_parser.add_argument("--manifest", type=Path, required=True)

    args = cast(argparse.Namespace, parser.parse_args())
    spec = _load_json(cast(Path, args.spec))

    if args.command == "validate-spec":
        payload = validate_spec(spec)
    elif args.command == "render-summary":
        payload = render_summary(spec)
    else:
        manifest = _load_json(cast(Path, args.manifest))
        payload = validate_manifest(spec, manifest)

    text = json.dumps(payload, indent=2, sort_keys=True)
    output_path = getattr(args, "output", None)
    if isinstance(output_path, Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
