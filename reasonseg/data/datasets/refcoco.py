from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from ..refcoco_materializer import DATASET_SPECS, annotation_output_dir


EVALUATOR_TYPE = "grounding_refcoco"
_COCO_IMAGE_SUBDIR = Path("coco") / "train2014"
_PUBLIC_DATASET_TOKENS = {
    "refcoco": "refcoco",
    "refcoco+": "refcoco_plus",
    "refcocog": "refcocog",
}


class RefExpRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class RegisteredRefExpDataset:
    name: str
    family_name: str
    dataset_token: str
    split: str
    split_family: str
    json_file_name: str

    def metadata(self, data_root: str | Path) -> dict[str, str]:
        resolved_root = Path(data_root).expanduser().resolve()
        image_root = resolved_root / _COCO_IMAGE_SUBDIR
        json_file = annotation_output_dir(resolved_root) / self.json_file_name
        _validate_dataset_assets(self.name, image_root, json_file)
        return {
            "name": self.name,
            "family_name": self.family_name,
            "dataset_token": self.dataset_token,
            "split": self.split,
            "split_family": self.split_family,
            "evaluator_type": EVALUATOR_TYPE,
            "image_root": str(image_root),
            "json_file": str(json_file),
        }


def _build_registered_datasets() -> tuple[RegisteredRefExpDataset, ...]:
    datasets: list[RegisteredRefExpDataset] = []
    for dataset_spec in DATASET_SPECS:
        dataset_token = _PUBLIC_DATASET_TOKENS[dataset_spec.dataset_name]
        family_name = f"{dataset_token}_{dataset_spec.split_by}"
        for split_name, json_file_name in dataset_spec.output_stems.items():
            datasets.append(
                RegisteredRefExpDataset(
                    name=f"{dataset_token}_{split_name}_{dataset_spec.split_by}",
                    family_name=family_name,
                    dataset_token=dataset_token,
                    split=split_name,
                    split_family=dataset_spec.split_by,
                    json_file_name=json_file_name,
                )
            )
    return tuple(datasets)


REGISTERED_REFEXP_DATASETS = _build_registered_datasets()
REGISTERED_REFEXP_DATASET_NAMES = tuple(
    dataset.name for dataset in REGISTERED_REFEXP_DATASETS
)
REGISTERED_REFEXP_FAMILY_NAMES = tuple(
    sorted({dataset.family_name for dataset in REGISTERED_REFEXP_DATASETS})
)
_DATASETS_BY_NAME = {dataset.name: dataset for dataset in REGISTERED_REFEXP_DATASETS}
_DATASETS_BY_FAMILY = {
    family_name: tuple(
        dataset
        for dataset in REGISTERED_REFEXP_DATASETS
        if dataset.family_name == family_name
    )
    for family_name in REGISTERED_REFEXP_FAMILY_NAMES
}


def list_registered_refexp_datasets() -> list[str]:
    return list(REGISTERED_REFEXP_DATASET_NAMES)


def list_registered_refexp_families() -> list[str]:
    return list(REGISTERED_REFEXP_FAMILY_NAMES)


def resolve_registered_refexp_dataset(
    name: str, data_root: str | Path
) -> RegisteredRefExpDataset:
    dataset = _DATASETS_BY_NAME.get(name)
    if dataset is None:
        _raise_unknown_dataset_name(name)
    _ = dataset.metadata(data_root)
    return dataset


def resolve_registered_refexp_family(
    name: str, data_root: str | Path
) -> tuple[RegisteredRefExpDataset, ...]:
    datasets = _DATASETS_BY_FAMILY.get(name)
    if datasets is None:
        _raise_unknown_dataset_name(name)
    for dataset in datasets:
        _ = dataset.metadata(data_root)
    return datasets


def build_registered_refexp_dataset_metadata(
    name: str, data_root: str | Path
) -> dict[str, str]:
    return resolve_registered_refexp_dataset(name, data_root).metadata(data_root)


def build_registered_refexp_family_metadata(
    name: str, data_root: str | Path
) -> list[dict[str, str]]:
    return [
        dataset.metadata(data_root)
        for dataset in resolve_registered_refexp_family(name, data_root)
    ]


def _validate_dataset_assets(name: str, image_root: Path, json_file: Path) -> None:
    missing_paths = [path for path in (image_root, json_file) if not path.exists()]
    if not missing_paths:
        return
    missing = ", ".join(str(path) for path in missing_paths)
    raise RefExpRegistryError(
        f"Registered RefCOCO dataset '{name}' is missing required asset(s): {missing}"
    )


def _raise_unknown_dataset_name(name: str) -> NoReturn:
    if "google" in name and ("refcocog" in name or "refcogog" in name):
        raise RefExpRegistryError(
            f"Unsupported RefCOCO dataset alias '{name}': Google split support "
            "is intentionally not registered. Use the canonical UMD family "
            "alias 'refcocog_umd' or one of {'refcocog_train_umd', "
            "'refcocog_val_umd', 'refcocog_test_umd'}."
        )

    supported_names = ", ".join(REGISTERED_REFEXP_DATASET_NAMES)
    supported_families = ", ".join(REGISTERED_REFEXP_FAMILY_NAMES)
    raise RefExpRegistryError(
        f"Unknown RefCOCO dataset alias '{name}'. Supported dataset names: "
        f"{supported_names}. Supported family aliases: {supported_families}."
    )
