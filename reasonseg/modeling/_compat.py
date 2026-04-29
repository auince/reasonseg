# pyright: reportMissingImports=false, reportConstantRedefinition=false, reportMissingTypeArgument=false
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


try:
    from detectron2.config import CfgNode, configurable
    from detectron2.data import MetadataCatalog
    from detectron2.data import detection_utils, transforms
    from detectron2.modeling import (
        META_ARCH_REGISTRY,
        build_backbone,
        build_sem_seg_head,
    )
    from detectron2.structures import BitMasks, Boxes, ImageList, Instances
    from detectron2.utils.comm import get_world_size
    from detectron2.utils.memory import retry_if_cuda_oom

    DETECTRON2_AVAILABLE = True
except ModuleNotFoundError:
    DETECTRON2_AVAILABLE = False

    def configurable(init_func):
        return init_func

    class CfgNode(dict):
        def __getattr__(self, key: str) -> Any:
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

    class _Registry:
        def __init__(self) -> None:
            self._items: dict[str, type[Any]] = {}

        def register(self):
            def decorator(cls):
                self._items[cls.__name__] = cls
                return cls

            return decorator

    META_ARCH_REGISTRY = _Registry()

    def _missing_detectron2(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            "detectron2 is required for this runtime path but is not installed."
        )

    def build_backbone(*args: Any, **kwargs: Any) -> Any:
        return _missing_detectron2(*args, **kwargs)

    def build_sem_seg_head(*args: Any, **kwargs: Any) -> Any:
        return _missing_detectron2(*args, **kwargs)

    class Boxes:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
            if self.tensor.numel() == 0:
                return torch.zeros((0,), dtype=torch.bool, device=self.tensor.device)
            widths = self.tensor[:, 2] - self.tensor[:, 0]
            heights = self.tensor[:, 3] - self.tensor[:, 1]
            return (widths > threshold) & (heights > threshold)

        def __getitem__(self, item: Any) -> "Boxes":
            return Boxes(self.tensor[item])

    class BitMasks:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        def nonempty(self) -> torch.Tensor:
            if self.tensor.numel() == 0:
                return torch.zeros((0,), dtype=torch.bool, device=self.tensor.device)
            return self.tensor.flatten(1).any(dim=1)

        def __getitem__(self, item: Any) -> torch.Tensor:
            return self.tensor[item]

    class Instances:
        def __init__(self, image_size: tuple[int, int]) -> None:
            self.image_size = image_size

        def has(self, name: str) -> bool:
            return hasattr(self, name)

        def __getitem__(self, item: Any) -> "Instances":
            new_instance = Instances(self.image_size)
            for key, value in self.__dict__.items():
                if key == "image_size":
                    continue
                try:
                    setattr(new_instance, key, value[item])
                except Exception:
                    setattr(new_instance, key, value)
            return new_instance

    class ImageList:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        @staticmethod
        def from_tensors(
            tensors: list[torch.Tensor],
            size_divisibility: int = 0,
        ) -> "ImageList":
            if not tensors:
                return ImageList(torch.empty(0))
            max_channels = max(tensor.shape[0] for tensor in tensors)
            max_height = max(tensor.shape[-2] for tensor in tensors)
            max_width = max(tensor.shape[-1] for tensor in tensors)
            if size_divisibility and size_divisibility > 1:
                max_height = int(
                    ((max_height + size_divisibility - 1) // size_divisibility)
                    * size_divisibility
                )
                max_width = int(
                    ((max_width + size_divisibility - 1) // size_divisibility)
                    * size_divisibility
                )

            batch = []
            for tensor in tensors:
                padded = torch.zeros(
                    (max_channels, max_height, max_width),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                padded[: tensor.shape[0], : tensor.shape[-2], : tensor.shape[-1]] = (
                    tensor
                )
                batch.append(padded)
            return ImageList(torch.stack(batch, dim=0))

    class _MetadataCatalog:
        _items: dict[str, Any] = {}

        @classmethod
        def get(cls, name: str) -> Any:
            if name not in cls._items:
                cls._items[name] = SimpleNamespace(
                    name=name,
                    stuff_classes=[],
                    thing_dataset_id_to_contiguous_id={},
                )
            return cls._items[name]

    MetadataCatalog = _MetadataCatalog

    class _DetectionUtils:
        @staticmethod
        def read_image(file_name: str, format: str | None = None) -> np.ndarray:
            from PIL import Image

            with Image.open(file_name) as image:
                if format == "RGB":
                    image = image.convert("RGB")
                return np.asarray(image)

        @staticmethod
        def check_image_size(dataset_dict: dict[str, Any], image: np.ndarray) -> None:
            height = dataset_dict.get("height")
            width = dataset_dict.get("width")
            if height is None or width is None:
                return
            if tuple(image.shape[:2]) != (height, width):
                raise ValueError(
                    f"Image size mismatch: expected {(height, width)}, got {tuple(image.shape[:2])}."
                )

    class _IdentityTransforms:
        @staticmethod
        def apply_segmentation(segmentation: np.ndarray) -> np.ndarray:
            return segmentation

    class _TransformsModule:
        @staticmethod
        def apply_transform_gens(
            augs: list[Any], image: np.ndarray
        ) -> tuple[np.ndarray, _IdentityTransforms]:
            del augs
            return image, _IdentityTransforms()

    detection_utils = _DetectionUtils()
    transforms = _TransformsModule()

    def retry_if_cuda_oom(func):
        return func

    def get_world_size() -> int:
        return 1


def resolve_repo_path(path: str, *, repo_root: Path | None = None) -> str:
    if not path:
        return path
    if Path(path).is_absolute():
        return path

    candidate = Path(path)
    if candidate.exists():
        return str(candidate)

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / path)
