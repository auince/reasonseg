from __future__ import annotations

from pathlib import Path

from .datasets import (
    REGISTERED_REFEXP_DATASET_NAMES,
    REGISTERED_REFEXP_FAMILY_NAMES,
    RefExpRegistryError,
    RegisteredRefExpDataset,
    build_registered_refexp_dataset_metadata,
    build_registered_refexp_family_metadata,
    list_registered_refexp_families as _list_registered_refexp_families,
    resolve_registered_refexp_dataset,
    resolve_registered_refexp_family,
)


def list_registered_refexp_datasets(data_root: str | Path | None = None) -> list[str]:
    if data_root is None:
        return list(REGISTERED_REFEXP_DATASET_NAMES)
    _ = _resolve_data_root(data_root)
    return list(REGISTERED_REFEXP_DATASET_NAMES)


def list_registered_refexp_families(data_root: str | Path | None = None) -> list[str]:
    if data_root is None:
        return _list_registered_refexp_families()
    _ = _resolve_data_root(data_root)
    return _list_registered_refexp_families()


def get_registered_refexp_dataset(
    name: str, data_root: str | Path
) -> RegisteredRefExpDataset:
    return resolve_registered_refexp_dataset(name, data_root)


def get_registered_refexp_dataset_metadata(
    name: str, data_root: str | Path
) -> dict[str, str]:
    return build_registered_refexp_dataset_metadata(name, data_root)


def get_registered_refexp_family(
    name: str, data_root: str | Path
) -> tuple[RegisteredRefExpDataset, ...]:
    return resolve_registered_refexp_family(name, data_root)


def get_registered_refexp_family_metadata(
    name: str, data_root: str | Path
) -> list[dict[str, str]]:
    return build_registered_refexp_family_metadata(name, data_root)


def _resolve_data_root(data_root: str | Path) -> Path:
    return Path(data_root).expanduser().resolve()


__all__ = [
    "REGISTERED_REFEXP_DATASET_NAMES",
    "REGISTERED_REFEXP_FAMILY_NAMES",
    "RefExpRegistryError",
    "RegisteredRefExpDataset",
    "get_registered_refexp_dataset",
    "get_registered_refexp_dataset_metadata",
    "get_registered_refexp_family",
    "get_registered_refexp_family_metadata",
    "list_registered_refexp_datasets",
    "list_registered_refexp_families",
]
