# pyright: reportMissingImports=false
from __future__ import annotations

from pathlib import Path

import pytest

from reasonseg.data.registry import (
    RefExpRegistryError,
    get_registered_refexp_dataset,
    get_registered_refexp_dataset_metadata,
    get_registered_refexp_family,
    get_registered_refexp_family_metadata,
    list_registered_refexp_datasets,
    list_registered_refexp_families,
)


EXPECTED_DATASET_NAMES = [
    "refcoco_train_unc",
    "refcoco_val_unc",
    "refcoco_testA_unc",
    "refcoco_testB_unc",
    "refcoco_plus_train_unc",
    "refcoco_plus_val_unc",
    "refcoco_plus_testA_unc",
    "refcoco_plus_testB_unc",
    "refcocog_train_umd",
    "refcocog_val_umd",
    "refcocog_test_umd",
]
EXPECTED_FAMILY_NAMES = [
    "refcoco_plus_unc",
    "refcoco_unc",
    "refcocog_umd",
]


def _build_materialized_dataset_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "datasets"
    annotations_root = data_root / "coco" / "annotations"
    annotations_root.mkdir(parents=True)
    (data_root / "coco" / "train2014").mkdir(parents=True)

    for file_name in (
        "refcoco_unc_train.json",
        "refcoco_unc_val.json",
        "refcoco_unc_testA.json",
        "refcoco_unc_testB.json",
        "refcoco_plus_unc_train.json",
        "refcoco_plus_unc_val.json",
        "refcoco_plus_unc_testA.json",
        "refcoco_plus_unc_testB.json",
        "refcocog_umd_train.json",
        "refcocog_umd_val.json",
        "refcocog_umd_test.json",
    ):
        (annotations_root / file_name).write_text("{}", encoding="utf-8")

    return data_root


def test_registry_lists_canonical_refexp_dataset_names(tmp_path: Path) -> None:
    data_root = _build_materialized_dataset_root(tmp_path)

    assert list_registered_refexp_datasets(data_root) == EXPECTED_DATASET_NAMES
    assert list_registered_refexp_families(data_root) == EXPECTED_FAMILY_NAMES


def test_registry_resolves_split_and_family_metadata(tmp_path: Path) -> None:
    data_root = _build_materialized_dataset_root(tmp_path)

    dataset = get_registered_refexp_dataset("refcoco_plus_testA_unc", data_root)
    dataset_metadata = get_registered_refexp_dataset_metadata(
        "refcoco_plus_testA_unc", data_root
    )
    family = get_registered_refexp_family("refcocog_umd", data_root)
    family_metadata = get_registered_refexp_family_metadata("refcocog_umd", data_root)

    assert dataset.name == "refcoco_plus_testA_unc"
    assert dataset.family_name == "refcoco_plus_unc"
    assert dataset.split == "testA"
    assert dataset.split_family == "unc"

    assert dataset_metadata == {
        "name": "refcoco_plus_testA_unc",
        "family_name": "refcoco_plus_unc",
        "dataset_token": "refcoco_plus",
        "split": "testA",
        "split_family": "unc",
        "evaluator_type": "grounding_refcoco",
        "image_root": str(data_root / "coco" / "train2014"),
        "json_file": str(
            data_root / "coco" / "annotations" / "refcoco_plus_unc_testA.json"
        ),
    }

    assert [entry.name for entry in family] == [
        "refcocog_train_umd",
        "refcocog_val_umd",
        "refcocog_test_umd",
    ]
    assert [entry["split"] for entry in family_metadata] == ["train", "val", "test"]
    assert all(entry["split_family"] == "umd" for entry in family_metadata)
    assert all(
        entry["evaluator_type"] == "grounding_refcoco" for entry in family_metadata
    )


@pytest.mark.parametrize(
    "name",
    ["refcocog_google_test", "refcogog_google_test"],
)
def test_registry_rejects_unsupported_google_split_aliases(
    name: str, tmp_path: Path
) -> None:
    data_root = _build_materialized_dataset_root(tmp_path)

    with pytest.raises(RefExpRegistryError, match="Google split support"):
        get_registered_refexp_dataset(name, data_root)


@pytest.mark.parametrize(
    "name",
    ["refcoco_dev_unc", "refcoco_plus_test_unc", "refcocog_testA_umd"],
)
def test_registry_rejects_unknown_split_aliases(name: str, tmp_path: Path) -> None:
    data_root = _build_materialized_dataset_root(tmp_path)

    with pytest.raises(RefExpRegistryError, match="Unknown RefCOCO dataset alias"):
        get_registered_refexp_dataset(name, data_root)
