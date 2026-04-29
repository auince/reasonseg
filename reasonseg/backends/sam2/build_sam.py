from __future__ import annotations

from importlib import import_module
from .._bootstrap import ensure_root_model_package_loaded


ensure_root_model_package_loaded()

_BUILD_SAM_MODULE = import_module("model.segment_anything_2.sam2.build_sam")

build_sam2 = _BUILD_SAM_MODULE.build_sam2
build_sam2_video_predictor = _BUILD_SAM_MODULE.build_sam2_video_predictor

__all__ = ["build_sam2", "build_sam2_video_predictor"]
