# pyright: reportMissingImports=false
from __future__ import annotations

from reasonseg.data.dataset_mappers.refcoco_dataset_mapper import RefCOCODatasetMapper


def test_reasonseg_mapper_preserves_query_contract_fields() -> None:
    mapper = RefCOCODatasetMapper(
        is_train=False,
        augmentations=[],
        image_format="RGB",
        metadata=object(),
        dataset_name="refcoco_val_unc",
        reasonseg_enabled=True,
    )
    dataset_dict: dict[str, object] = {}
    prompts = ["red dog", "man watering flowers", "no bicycle"]

    mapper._attach_reasonseg_fields(dataset_dict, prompts)

    assert dataset_dict["query_text"] == prompts
    assert dataset_dict["requested_target"] == ["dog", "man", "bicycle"]
    assert dataset_dict["slice_tags"] == ["attribute", "relation_action", "no_target"]
    assert dataset_dict["positive_mask_count"] == [1, 1, 0]
    assert dataset_dict["composed_prompt"] == [
        "red dog",
        "man watering flowers",
        "no bicycle",
    ]
