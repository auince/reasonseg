# pyright: reportMissingTypeArgument=false, reportReturnType=false, reportMissingImports=false, reportAttributeAccessIssue=false
from __future__ import annotations

import copy
import logging
import random

import numpy as np
import torch
from pycocotools import mask
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision import transforms

from ...query import parse_query
from ...modeling._compat import (
    BitMasks,
    Boxes,
    Instances,
    MetadataCatalog,
    configurable,
    detection_utils as utils,
    transforms as T,
)
from ...modeling.prompting import (
    compose_reasonseg_prompt,
    infer_requested_target,
    infer_slice_tag,
)

_SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
_SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
_SAM_IMG_SIZE = 1024
_BEIT_IMG_SIZE = 224

__all__ = ["RefCOCODatasetMapper"]


def filter_empty_instances_by_box(
    instances,
    by_box: bool = True,
    by_mask: bool = False,
    box_threshold: float = 1e-5,
    return_mask: bool = False,
):
    assert by_box or by_mask
    results = []
    if by_box:
        results.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        results.append(instances.gt_masks.nonempty())
    if not results:
        return instances

    mask_keep = results[0]
    for item in results[1:]:
        mask_keep = mask_keep & item
    if return_mask:
        return instances[mask_keep], mask_keep
    return instances[mask_keep]


def sam_preprocess(x: np.ndarray) -> torch.Tensor:
    x_tensor = torch.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1)))
    x_tensor = F.interpolate(
        x_tensor.unsqueeze(0),
        (_SAM_IMG_SIZE, _SAM_IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (x_tensor - _SAM_PIXEL_MEAN) / _SAM_PIXEL_STD


def beit3_preprocess(x: np.ndarray) -> torch.Tensor:
    beit_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (_BEIT_IMG_SIZE, _BEIT_IMG_SIZE),
                interpolation=InterpolationMode.BICUBIC,
                antialias=False,
            ),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return beit_preprocess(np.array(x))


def build_transform_gen(cfg, is_train):
    del cfg, is_train
    return []


class RefCOCODatasetMapper:
    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations,
        image_format,
        metadata,
        tokenizer=None,
        reasonseg_enabled: bool = False,
        dataset_name,
    ):
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.reasonseg_enabled = reasonseg_enabled
        self.dataset_name = dataset_name

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}"
        )

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_transform_gen(cfg, is_train)
        dataset_name = cfg.DATASETS.TRAIN[0] if is_train else cfg.DATASETS.TEST[0]
        meta = MetadataCatalog.get(dataset_name)
        return {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "metadata": meta,
            "tokenizer": None,
            "reasonseg_enabled": cfg.MODEL.OpenWorldSAM2.REASONSEG_ENABLED,
            "dataset_name": dataset_name,
        }

    @staticmethod
    def _infer_requested_target(raw_prompt, query_struct):
        return infer_requested_target(raw_prompt, query_struct)

    @staticmethod
    def _infer_slice_tag(query_struct):
        return infer_slice_tag(query_struct)

    @staticmethod
    def _compose_prompt(query_struct, raw_prompt):
        return compose_reasonseg_prompt(query_struct, fallback_text=raw_prompt)

    def _attach_reasonseg_fields(self, dataset_dict, prompts) -> None:
        query_structs = [parse_query(prompt) for prompt in prompts]
        dataset_dict["query_text"] = prompts
        dataset_dict["query_struct"] = query_structs
        dataset_dict["requested_target"] = [
            self._infer_requested_target(prompt, query_struct)
            for prompt, query_struct in zip(prompts, query_structs)
        ]
        dataset_dict["slice_tags"] = [
            self._infer_slice_tag(query_struct) for query_struct in query_structs
        ]
        dataset_dict["positive_mask_count"] = [
            1 if query_struct["exists"] else 0 for query_struct in query_structs
        ]
        dataset_dict["composed_prompt"] = [
            self._compose_prompt(query_struct, prompt)
            for prompt, query_struct in zip(prompts, query_structs)
        ]

    def __call__(self, dataset_dict):
        dataset_dict = {**dataset_dict}
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        image_shape = image.shape[:2]

        padding_mask = np.ones(image.shape[:2])
        image, transforms_applied = T.apply_transform_gens(self.tfm_gens, image)
        padding_mask = transforms_applied.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        dataset_dict["image"] = sam_preprocess(image)
        dataset_dict["evf_image"] = beit3_preprocess(image)
        dataset_dict["padding_mask"] = torch.as_tensor(
            np.ascontiguousarray(padding_mask)
        )
        dataset_dict["prompt"] = ["object"]
        dataset_dict["unique_categories"] = [0]

        grounding_anno = dataset_dict["grounding_info"]
        assert len(grounding_anno) > 0
        masks_grd = []
        texts_grd = []
        boxes_grd = []
        for ann in grounding_anno:
            rle = mask.frPyObjects(
                ann["segmentation"],
                dataset_dict["height"],
                dataset_dict["width"],
            )
            mask_value = mask.decode(rle)
            mask_value = np.sum(mask_value, axis=2).astype(np.uint8)
            masks_grd.append(mask_value)
            texts_grd.append([item["raw"].lower() for item in ann["sentences"]])
            boxes_grd.append(ann["bbox"])

        masks_grd_tensor = torch.from_numpy(np.stack(masks_grd))
        boxes_grd_tensor = torch.tensor(boxes_grd)
        dataset_dict["groundings"] = {
            "masks": masks_grd_tensor,
            "texts": texts_grd,
            "boxes": boxes_grd_tensor,
        }

        if self.is_train:
            dataset_dict["prompt"] = [random.choice(item) for item in texts_grd]
            dummy_classes = list(range(len(texts_grd)))
            dataset_dict["unique_categories"] = dummy_classes
            grouped_instances = []
            for class_id in dummy_classes:
                category_instances = Instances(image_shape)
                category_instances.gt_masks = masks_grd_tensor[class_id : class_id + 1]
                category_instances.gt_boxes = boxes_grd_tensor[class_id : class_id + 1]
                category_instances.gt_classes = torch.tensor(
                    [class_id], dtype=torch.int64
                )
                grouped_instances.append(category_instances)
            dataset_dict["instances"] = grouped_instances
        else:
            dataset_dict["prompt"] = [item[0] for item in texts_grd]
            dummy_classes = list(range(len(texts_grd)))
            dataset_dict["unique_categories"] = dummy_classes
            instances = Instances(image_shape)
            instances.gt_masks = masks_grd_tensor
            instances.gt_boxes = boxes_grd_tensor
            instances.gt_classes = torch.tensor(dummy_classes, dtype=torch.int64)
            dataset_dict["instances"] = instances

        if self.reasonseg_enabled:
            self._attach_reasonseg_fields(dataset_dict, dataset_dict["prompt"])
        return dataset_dict
