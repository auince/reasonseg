from .configuration_evf import EvfConfig
from .criterion import SetCriterion
from .evf_sam2 import EvfSam2Model
from .matcher import HungarianMatcher
from .open_world_sam2 import OpenWorldSAM2
from .open_world_sam2_config import add_open_world_sam2_config

# Lazy import of VR_OV to register it in META_ARCH_REGISTRY
try:
    from model.vr_ov import VR_OV  # noqa: F401
except ImportError:
    pass  # VR_OV module not available (e.g., model/ not on path)

__all__ = [
    "EvfConfig",
    "EvfSam2Model",
    "HungarianMatcher",
    "OpenWorldSAM2",
    "SetCriterion",
    "add_open_world_sam2_config",
]
