from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torchvision
from torch import Tensor


def _max_by_axis(the_list: list[list[int]]) -> list[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    def __init__(self, tensors: Tensor, mask: Optional[Tensor]) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device | str) -> "NestedTensor":
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self) -> tuple[Tensor, Optional[Tensor]]:
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: list[Tensor]) -> NestedTensor:
    if tensor_list[0].ndim != 3:
        raise ValueError("not supported")
    if torchvision._is_tracing():
        return _onnx_nested_tensor_from_tensor_list(tensor_list)

    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    batch_size, channels, height, width = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
    for img, pad_img, pad_mask in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        pad_mask[: img.shape[1], : img.shape[2]] = False
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(
    tensor_list: list[Tensor],
) -> NestedTensor:
    max_size = []
    for index in range(tensor_list[0].dim()):
        max_size_index = torch.max(
            torch.stack(
                [
                    torch.tensor(img.shape[index], device=img.device)
                    for img in tensor_list
                ]
            ).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_index)
    max_size_tuple = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [
            current_max - current_size
            for current_max, current_size in zip(max_size_tuple, tuple(img.shape))
        ]
        padded_img = torch.nn.functional.pad(
            img,
            (0, padding[2], 0, padding[1], 0, padding[0]),
        )
        padded_imgs.append(padded_img)

        mask = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            mask,
            (0, padding[2], 0, padding[1]),
            "constant",
            1,
        )
        padded_masks.append(padded_mask.to(torch.bool))

    return NestedTensor(torch.stack(padded_imgs), mask=torch.stack(padded_masks))


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
