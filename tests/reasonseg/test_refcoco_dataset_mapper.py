# pyright: reportMissingImports=false
from __future__ import annotations

import numpy as np
import torch

from reasonseg.data.dataset_mappers.refcoco_dataset_mapper import RefCOCODatasetMapper
from reasonseg.modeling._compat import Instances


def test_reasonseg_mapper_emits_canonical_query_contract_without_legacy_flattening() -> None:
    mapper = RefCOCODatasetMapper(
        is_train=False,
        augmentations=[],
        image_format="RGB",
        metadata=object(),
        dataset_name="refcoco_val_unc",
        reasonseg_enabled=False,
    )
    dataset_dict: dict[str, object] = {"image_id": 17}
    prompts = ["red dog", "man watering flowers", "no bicycle"]

    mapper._attach_reasonseg_fields(dataset_dict, prompts)

    assert dataset_dict["query_text"] == prompts
    assert dataset_dict["query_struct"] == [
        {
            "target": "dog",
            "attributes": ["red"],
            "relations": [],
            "actions": [],
            "negatives": [],
            "exists": True,
        },
        {
            "target": "man",
            "attributes": [],
            "relations": [],
            "actions": [{"verb": "watering", "target": "flowers"}],
            "negatives": [],
            "exists": True,
        },
        {
            "target": None,
            "attributes": [],
            "relations": [],
            "actions": [],
            "negatives": ["absent_object"],
            "exists": False,
        },
    ]
    assert dataset_dict["requested_target"] == ["dog", "man", "bicycle"]
    assert dataset_dict["slice_tags"] == ["attribute", "relation_action", "no_target"]
    assert dataset_dict["positive_mask_count"] == [1, 1, 0]
    assert dataset_dict["query_metadata"] == [
        {"image_id": 17, "prompt_index": 0, "prompt_count": 3},
        {"image_id": 17, "prompt_index": 1, "prompt_count": 3},
        {"image_id": 17, "prompt_index": 2, "prompt_count": 3},
    ]
    assert "composed_prompt" not in dataset_dict


def _build_grounding_dataset_dict() -> dict[str, object]:
    return {
        "file_name": "unused.png",
        "height": 2,
        "width": 2,
        "image_id": 11,
        "grounding_info": [
            {
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "bbox": [0, 0, 1, 1],
                "sentences": [{"raw": "Red Dog"}, {"raw": "The Dog"}],
            },
            {
                "segmentation": [[1, 1, 2, 1, 2, 2, 1, 2]],
                "bbox": [1, 1, 1, 1],
                "sentences": [{"raw": "Man watering flowers"}],
            },
        ],
    }


def _patch_mapper_runtime(monkeypatch) -> None:
    image = np.full((2, 2, 3), 127, dtype=np.uint8)
    mask_lookup = {
        ((0, 0, 1, 0, 1, 1, 0, 1),): np.array([[1, 0], [0, 0]], dtype=np.uint8),
        ((1, 1, 2, 1, 2, 2, 1, 2),): np.array([[0, 0], [0, 1]], dtype=np.uint8),
    }

    monkeypatch.setattr(
        "reasonseg.data.dataset_mappers.refcoco_dataset_mapper.utils.read_image",
        lambda file_name, format: image.copy(),
    )
    monkeypatch.setattr(
        "reasonseg.data.dataset_mappers.refcoco_dataset_mapper.utils.check_image_size",
        lambda dataset_dict, loaded_image: None,
    )
    monkeypatch.setattr(
        "reasonseg.data.dataset_mappers.refcoco_dataset_mapper.T.apply_transform_gens",
        lambda tfm_gens, loaded_image: (
            loaded_image,
            type(
                "_IdentityTransform",
                (),
                {"apply_segmentation": staticmethod(lambda padding_mask: padding_mask)},
            )(),
        ),
    )
    monkeypatch.setattr(
        "reasonseg.data.dataset_mappers.refcoco_dataset_mapper.mask.frPyObjects",
        lambda segmentation, height, width: tuple(tuple(poly) for poly in segmentation),
    )
    monkeypatch.setattr(
        "reasonseg.data.dataset_mappers.refcoco_dataset_mapper.mask.decode",
        lambda rle: np.stack([mask_lookup[rle]], axis=2),
    )


def test_reasonseg_mapper_train_call_emits_aligned_structured_fields(monkeypatch) -> None:
    _patch_mapper_runtime(monkeypatch)
    monkeypatch.setattr(
        "reasonseg.data.dataset_mappers.refcoco_dataset_mapper.random.choice",
        lambda options: options[0],
    )

    mapper = RefCOCODatasetMapper(
        is_train=True,
        augmentations=[],
        image_format="RGB",
        metadata=object(),
        dataset_name="refcoco_train_unc",
        reasonseg_enabled=False,
    )

    dataset_dict = mapper(_build_grounding_dataset_dict())

    assert dataset_dict["prompt"] == ["red dog", "man watering flowers"]
    assert dataset_dict["query_text"] == ["red dog", "man watering flowers"]
    assert dataset_dict["requested_target"] == ["dog", "man"]
    assert dataset_dict["slice_tags"] == ["attribute", "relation_action"]
    assert dataset_dict["positive_mask_count"] == [1, 1]
    assert dataset_dict["query_metadata"] == [
        {"image_id": 11, "prompt_index": 0, "prompt_count": 2},
        {"image_id": 11, "prompt_index": 1, "prompt_count": 2},
    ]
    assert "composed_prompt" not in dataset_dict
    assert dataset_dict["unique_categories"] == [0, 1]
    assert len(dataset_dict["instances"]) == 2
    assert all(isinstance(item, Instances) for item in dataset_dict["instances"])


def test_reasonseg_mapper_eval_call_keeps_legacy_flattened_prompt_compatibility(
    monkeypatch,
) -> None:
    _patch_mapper_runtime(monkeypatch)

    mapper = RefCOCODatasetMapper(
        is_train=False,
        augmentations=[],
        image_format="RGB",
        metadata=object(),
        dataset_name="refcoco_val_unc",
        reasonseg_enabled=True,
    )

    dataset_dict = mapper(_build_grounding_dataset_dict())

    assert dataset_dict["prompt"] == ["red dog", "man watering flowers"]
    assert dataset_dict["query_text"] == ["red dog", "man watering flowers"]
    assert dataset_dict["requested_target"] == ["dog", "man"]
    assert dataset_dict["slice_tags"] == ["attribute", "relation_action"]
    assert dataset_dict["positive_mask_count"] == [1, 1]
    assert dataset_dict["composed_prompt"] == [
        "red dog",
        "man watering flowers",
    ]
    assert dataset_dict["query_metadata"] == [
        {"image_id": 11, "prompt_index": 0, "prompt_count": 2},
        {"image_id": 11, "prompt_index": 1, "prompt_count": 2},
    ]
    assert isinstance(dataset_dict["instances"], Instances)
    assert torch.equal(dataset_dict["instances"].gt_classes, torch.tensor([0, 1]))
