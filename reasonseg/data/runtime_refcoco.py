# pyright: reportMissingImports=false, reportExplicitAny=false, reportAny=false
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .registry import (
    get_registered_refexp_dataset_metadata,
    list_registered_refexp_datasets,
)


def load_refcoco_json(
    *,
    image_root: str | Path,
    annot_json: str | Path,
    dataset_name: str,
) -> list[dict[str, Any]]:
    image_root_path = Path(image_root)
    payload = json.loads(Path(annot_json).read_text(encoding="utf-8"))
    grounding_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        grounding_by_image_id[int(annotation["image_id"])].append(annotation)

    dataset_dicts: list[dict[str, Any]] = []
    for image in payload["images"]:
        image_id = int(image["id"])
        dataset_dicts.append(
            {
                "file_name": str(image_root_path / str(image["file_name"])),
                "image_id": image_id,
                "height": int(image["height"]),
                "width": int(image["width"]),
                "dataset_name": dataset_name,
                "grounding_info": grounding_by_image_id[image_id],
            }
        )
    return dataset_dicts


def register_refcoco_datasets(data_root: str | Path) -> None:
    from detectron2.data import DatasetCatalog, MetadataCatalog

    registered = set(DatasetCatalog.list())
    for dataset_name in list_registered_refexp_datasets(data_root):
        metadata = get_registered_refexp_dataset_metadata(dataset_name, data_root)
        image_root = metadata["image_root"]
        annot_json = metadata["json_file"]
        if dataset_name not in registered:
            DatasetCatalog.register(
                dataset_name,
                lambda dataset_name=dataset_name, image_root=image_root, annot_json=annot_json: (
                    load_refcoco_json(
                        image_root=image_root,
                        annot_json=annot_json,
                        dataset_name=dataset_name,
                    )
                ),
            )
        MetadataCatalog.get(dataset_name).set(
            image_root=image_root,
            json_file=annot_json,
            evaluator_type=metadata["evaluator_type"],
            dataset_token=metadata["dataset_token"],
            family_name=metadata["family_name"],
            split=metadata["split"],
            split_family=metadata["split_family"],
            ignore_label=255,
            label_divisor=1000,
        )
