from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any


class RefCOCODataError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    split_by: str
    output_stems: dict[str, str]

    @property
    def raw_dir_name(self) -> str:
        return self.dataset_name

    @property
    def refs_file_name(self) -> str:
        return f"refs({self.split_by}).p"


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_name="refcoco",
        split_by="unc",
        output_stems={
            "train": "refcoco_unc_train.json",
            "val": "refcoco_unc_val.json",
            "testA": "refcoco_unc_testA.json",
            "testB": "refcoco_unc_testB.json",
        },
    ),
    DatasetSpec(
        dataset_name="refcoco+",
        split_by="unc",
        output_stems={
            "train": "refcoco_plus_unc_train.json",
            "val": "refcoco_plus_unc_val.json",
            "testA": "refcoco_plus_unc_testA.json",
            "testB": "refcoco_plus_unc_testB.json",
        },
    ),
    DatasetSpec(
        dataset_name="refcocog",
        split_by="umd",
        output_stems={
            "train": "refcocog_umd_train.json",
            "val": "refcocog_umd_val.json",
            "test": "refcocog_umd_test.json",
        },
    ),
)


def prepare_refcoco_data(
    data_root: Path, *, materialize: bool = False, verify_only: bool = False
) -> dict[str, Any]:
    if not materialize and not verify_only:
        raise RefCOCODataError(
            "Nothing to do: pass --materialize, --verify-only, or both."
        )

    data_root = data_root.resolve()
    verify_raw_assets(data_root)

    materialized_outputs: list[Path] = []
    if materialize:
        materialized_outputs = materialize_refcoco_family(data_root)

    verified_outputs = verify_materialized_outputs(data_root)
    return {
        "data_root": data_root,
        "materialized_outputs": materialized_outputs,
        "verified_outputs": verified_outputs,
    }


def verify_raw_assets(data_root: Path) -> None:
    missing_inputs: list[Path] = []
    for spec in DATASET_SPECS:
        dataset_root = data_root / spec.raw_dir_name
        for path in (
            dataset_root / "instances.json",
            dataset_root / spec.refs_file_name,
        ):
            if not path.is_file():
                missing_inputs.append(path)

    required_coco_paths = (
        data_root / "coco" / "annotations" / "instances_train2014.json",
        data_root / "coco" / "train2014",
    )
    for path in required_coco_paths:
        if not path.exists():
            missing_inputs.append(path)

    if missing_inputs:
        missing = ", ".join(str(path) for path in sorted(missing_inputs))
        raise RefCOCODataError(f"Missing RefCOCO raw input asset(s): {missing}")


def materialize_refcoco_family(data_root: Path) -> list[Path]:
    output_dir = annotation_output_dir(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    materialized_paths: list[Path] = []
    for spec in DATASET_SPECS:
        split_payloads = build_dataset_payloads(data_root, spec)
        for split_name, payload in split_payloads.items():
            output_path = output_dir / spec.output_stems[split_name]
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            materialized_paths.append(output_path)

    return materialized_paths


def verify_materialized_outputs(data_root: Path) -> list[Path]:
    output_dir = annotation_output_dir(data_root)
    missing_outputs = [
        path for path in expected_output_paths(data_root) if not path.is_file()
    ]
    if missing_outputs:
        missing = ", ".join(str(path) for path in missing_outputs)
        raise RefCOCODataError(f"Missing RefCOCO materialized output(s): {missing}")

    verified_paths: list[Path] = []
    for path in expected_output_paths(data_root):
        validate_materialized_json(path)
        verified_paths.append(path)
    return verified_paths


def annotation_output_dir(data_root: Path) -> Path:
    return data_root / "coco" / "annotations"


def expected_output_paths(data_root: Path) -> list[Path]:
    output_dir = annotation_output_dir(data_root)
    paths: list[Path] = []
    for spec in DATASET_SPECS:
        for file_name in spec.output_stems.values():
            paths.append(output_dir / file_name)
    return paths


def build_dataset_payloads(
    data_root: Path, spec: DatasetSpec
) -> dict[str, dict[str, Any]]:
    dataset_root = data_root / spec.raw_dir_name
    refs = load_refs(dataset_root / spec.refs_file_name)
    instances = json.loads(
        (dataset_root / "instances.json").read_text(encoding="utf-8")
    )

    annotations_by_id = {
        int(annotation["id"]): annotation for annotation in instances["annotations"]
    }
    images_by_id = {int(image["id"]): image for image in instances["images"]}

    refs_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        refs_by_split[str(ref["split"])].append(ref)

    payloads: dict[str, dict[str, Any]] = {}
    for split_name in spec.output_stems:
        split_refs = refs_by_split.get(split_name, [])
        if not split_refs:
            raise RefCOCODataError(
                f"No raw refs found for dataset={spec.dataset_name} split_by={spec.split_by} split={split_name}"
            )
        payloads[split_name] = build_split_payload(
            spec,
            split_name,
            split_refs,
            annotations_by_id,
            images_by_id,
        )
    return payloads


def build_split_payload(
    spec: DatasetSpec,
    split_name: str,
    refs: list[dict[str, Any]],
    annotations_by_id: dict[int, dict[str, Any]],
    images_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    refs_by_ann_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        refs_by_ann_id[int(ref["ann_id"])].append(ref)

    output_annotations: list[dict[str, Any]] = []
    image_ids: set[int] = set()

    for ann_id in sorted(refs_by_ann_id):
        ref_group = refs_by_ann_id[ann_id]
        annotation = annotations_by_id.get(ann_id)
        if annotation is None:
            raise RefCOCODataError(
                f"Missing annotation id {ann_id} for dataset={spec.dataset_name} split={split_name}"
            )
        image_id = int(annotation["image_id"])
        image = images_by_id.get(image_id)
        if image is None:
            raise RefCOCODataError(
                f"Missing image id {image_id} for dataset={spec.dataset_name} split={split_name}"
            )
        image_ids.add(image_id)
        output_annotations.append(
            merge_ref_group(spec, split_name, image, annotation, ref_group)
        )

    output_images = [deepcopy(images_by_id[image_id]) for image_id in sorted(image_ids)]
    return {"images": output_images, "annotations": output_annotations}


def merge_ref_group(
    spec: DatasetSpec,
    split_name: str,
    image: dict[str, Any],
    annotation: dict[str, Any],
    ref_group: list[dict[str, Any]],
) -> dict[str, Any]:
    merged_annotation = deepcopy(annotation)
    primary_ref = ref_group[0]
    sentences = merge_sentences(ref_group)
    ref_ids = [int(ref["ref_id"]) for ref in ref_group]

    merged_annotation.update(
        {
            "dataset_name": spec.dataset_name,
            "split_by": spec.split_by,
            "split": split_name,
            "file_name": str(image["file_name"]),
            "ref_id": ref_ids[0],
            "ref_ids": ref_ids,
            "sent_ids": [int(sentence["sent_id"]) for sentence in sentences],
            "sentences": sentences,
            "multi_ref_policy": "merge_same_ann_id_with_stable_sentence_union",
            "source_ref_count": len(ref_group),
            "category_id": int(primary_ref["category_id"]),
        }
    )
    return merged_annotation


def merge_sentences(ref_group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_sentences: list[dict[str, Any]] = []
    seen_sent_ids: set[int] = set()
    for ref in ref_group:
        for sentence in ref["sentences"]:
            sent_id = int(sentence["sent_id"])
            if sent_id in seen_sent_ids:
                continue
            seen_sent_ids.add(sent_id)
            merged_sentences.append(deepcopy(sentence))
    if not merged_sentences:
        raise RefCOCODataError(
            "Encountered a ref group without any sentences to materialize"
        )
    return merged_sentences


def load_refs(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        refs = pickle.load(handle)
    if not isinstance(refs, list):
        raise RefCOCODataError(
            f"Expected list of refs in {path}, found {type(refs).__name__}"
        )
    return refs


def validate_materialized_json(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RefCOCODataError(f"Materialized output {path} must be a JSON object")
    for key in ("images", "annotations"):
        value = payload.get(key)
        if not isinstance(value, list):
            raise RefCOCODataError(
                f"Materialized output {path} is missing top-level list '{key}'"
            )

    image_ids = set()
    for image in payload["images"]:
        if not isinstance(image, dict):
            raise RefCOCODataError(f"Output image entry in {path} must be an object")
        for key in ("id", "file_name", "height", "width"):
            if key not in image:
                raise RefCOCODataError(
                    f"Output image entry in {path} is missing '{key}'"
                )
        image_ids.add(int(image["id"]))

    for annotation in payload["annotations"]:
        if not isinstance(annotation, dict):
            raise RefCOCODataError(
                f"Output annotation entry in {path} must be an object"
            )
        for key in ("image_id", "bbox", "segmentation", "sentences"):
            if key not in annotation:
                raise RefCOCODataError(
                    f"Output annotation entry in {path} is missing '{key}'"
                )
        bbox = annotation["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RefCOCODataError(
                f"Output annotation entry in {path} must keep bbox in XYWH list form"
            )
        if int(annotation["image_id"]) not in image_ids:
            raise RefCOCODataError(
                f"Output annotation entry in {path} references unknown image_id {annotation['image_id']}"
            )
        sentences = annotation["sentences"]
        if not isinstance(sentences, list) or not sentences:
            raise RefCOCODataError(
                f"Output annotation entry in {path} must include non-empty sentences"
            )
        for sentence in sentences:
            if not isinstance(sentence, dict) or "raw" not in sentence:
                raise RefCOCODataError(
                    f"Output annotation entry in {path} must preserve sentences[*].raw"
                )
