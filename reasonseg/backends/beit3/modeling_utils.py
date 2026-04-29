from __future__ import annotations

from importlib import import_module
from .._bootstrap import ensure_root_model_package_loaded


ensure_root_model_package_loaded()

_MODELING_UTILS = import_module("model.unilm.beit3.modeling_utils")

BEiT3Wrapper = _MODELING_UTILS.BEiT3Wrapper
_get_base_config = _MODELING_UTILS._get_base_config
_get_large_config = _MODELING_UTILS._get_large_config

__all__ = ["BEiT3Wrapper", "_get_base_config", "_get_large_config"]
