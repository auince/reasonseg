# pyright: reportMissingImports=false
from __future__ import annotations

import json
from pathlib import Path

from reasonseg.data.runtime_refcoco import load_refcoco_json


def test_runtime_refcoco_loader_preserves_image_and_grounding_records(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "datasets" / "coco" / "train2014"
    image_root.mkdir(parents=True)
    payload = {
        "images": [
            {
                "id": 7,
                "file_name": "COCO_train2014_000000000007.jpg",
                "height": 10,
                "width": 12,
            }
        ],
        "annotations": [
            {
                "image_id": 7,
                "bbox": [1, 2, 3, 4],
                "segmentation": [[0, 0, 1, 0, 1, 1]],
                "sentences": [{"raw": "red dog"}],
            }
        ],
    }
    annot_json = tmp_path / "datasets" / "coco" / "annotations" / "refcoco_unc_val.json"
    annot_json.parent.mkdir(parents=True)
    annot_json.write_text(json.dumps(payload), encoding="utf-8")

    dataset_dicts = load_refcoco_json(
        image_root=image_root,
        annot_json=annot_json,
        dataset_name="refcoco_val_unc",
    )

    assert dataset_dicts == [
        {
            "file_name": str(image_root / "COCO_train2014_000000000007.jpg"),
            "image_id": 7,
            "height": 10,
            "width": 12,
            "dataset_name": "refcoco_val_unc",
            "grounding_info": payload["annotations"],
        }
    ]
