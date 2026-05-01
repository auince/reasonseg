# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

from reasonseg.modeling._compat import CfgNode as CN


_VR_OV_REQUIRED_FIELDS = {
    "QUERY_PARSER": (
        "ENABLED",
        "GNN_LAYERS",
        "GNN_HEADS",
        "CHECKPOINT",
        "HIDDEN_DIM",
        "OUT_DIM",
    ),
    "SCENE_GRAPH": (
        "ENABLED",
        "HOI_TOKENS",
        "REGION_TOPK",
        "HIDDEN_DIM",
    ),
    "COMP_MATCHER": (
        "ENABLED",
        "CMF_LAYERS",
        "HIDDEN_DIM",
    ),
    "REFINE_DECODER": (
        "ENABLED",
        "MAX_ITER",
        "ATTR_THRESHOLD",
    ),
}


def _build_vr_ov_config() -> CN:
    vr_ov = CN()
    vr_ov.ENABLED = False

    vr_ov.QUERY_PARSER = CN()
    vr_ov.QUERY_PARSER.ENABLED = True
    vr_ov.QUERY_PARSER.GNN_LAYERS = 2
    vr_ov.QUERY_PARSER.GNN_HEADS = 4
    vr_ov.QUERY_PARSER.CHECKPOINT = ""
    vr_ov.QUERY_PARSER.HIDDEN_DIM = 256
    vr_ov.QUERY_PARSER.OUT_DIM = 256

    vr_ov.SCENE_GRAPH = CN()
    vr_ov.SCENE_GRAPH.ENABLED = True
    vr_ov.SCENE_GRAPH.HOI_TOKENS = 5
    vr_ov.SCENE_GRAPH.REGION_TOPK = 64
    vr_ov.SCENE_GRAPH.HIDDEN_DIM = 256

    vr_ov.COMP_MATCHER = CN()
    vr_ov.COMP_MATCHER.ENABLED = True
    vr_ov.COMP_MATCHER.CMF_LAYERS = 3
    vr_ov.COMP_MATCHER.HIDDEN_DIM = 256

    vr_ov.REFINE_DECODER = CN()
    vr_ov.REFINE_DECODER.ENABLED = True
    vr_ov.REFINE_DECODER.MAX_ITER = 3
    vr_ov.REFINE_DECODER.ATTR_THRESHOLD = 0.5

    vr_ov.LOSS = CN()
    vr_ov.LOSS.MASK_ENABLED = True
    vr_ov.LOSS.ATTR_ENABLED = False
    vr_ov.LOSS.REL_ENABLED = False
    vr_ov.LOSS.ACT_ENABLED = False
    vr_ov.LOSS.COMPOSE_ENABLED = False
    vr_ov.LOSS.LAMBDA_MASK = 5.0
    vr_ov.LOSS.LAMBDA_ATTR = 1.0
    vr_ov.LOSS.LAMBDA_REL = 0.5
    vr_ov.LOSS.LAMBDA_ACT = 0.5
    vr_ov.LOSS.LAMBDA_COMPOSE = 0.3
    return vr_ov


def add_vr_ov_config(cfg: CN) -> None:
    """Register canonical VR-OV configuration under ``cfg.MODEL.VR_OV``."""

    cfg.MODEL.VR_OV = _build_vr_ov_config()


def add_vr_ov_compat_config(cfg: CN) -> None:
    """Register the legacy OpenWorldSAM2 compatibility shim.

    This intentionally delegates to the canonical VR-OV config authority so the
    OpenWorldSAM2 and VR_OV config paths cannot drift.
    """

    add_vr_ov_config(cfg)


def validate_vr_ov_config(cfg: CN) -> None:
    """Fail loudly for malformed canonical VR-OV enablement states."""

    meta_architecture = getattr(cfg.MODEL, "META_ARCHITECTURE", "")
    vr_ov = getattr(cfg.MODEL, "VR_OV", None)
    if meta_architecture == "VR_OV" and vr_ov is None:
        raise ValueError(
            'MODEL.META_ARCHITECTURE="VR_OV" requires a MODEL.VR_OV config block.'
        )

    if vr_ov is None:
        return

    vr_ov_enabled = bool(getattr(vr_ov, "ENABLED", False))

    if meta_architecture == "VR_OV" and not vr_ov_enabled:
        raise ValueError(
            'MODEL.META_ARCHITECTURE="VR_OV" requires MODEL.VR_OV.ENABLED=True.'
        )

    if not vr_ov_enabled:
        return

    missing_fields: list[str] = []
    disabled_components: list[str] = []
    for component_name, field_names in _VR_OV_REQUIRED_FIELDS.items():
        component_cfg = getattr(vr_ov, component_name, None)
        if component_cfg is None:
            missing_fields.append(f"MODEL.VR_OV.{component_name}")
            continue
        for field_name in field_names:
            if not hasattr(component_cfg, field_name):
                missing_fields.append(f"MODEL.VR_OV.{component_name}.{field_name}")
        if hasattr(component_cfg, "ENABLED") and not bool(component_cfg.ENABLED):
            disabled_components.append(f"MODEL.VR_OV.{component_name}.ENABLED")

    if missing_fields:
        raise ValueError(
            "MODEL.VR_OV.ENABLED=True requires the canonical VR-OV config schema. "
            f"Missing: {', '.join(missing_fields)}"
        )

    if disabled_components:
        raise ValueError(
            "MODEL.VR_OV.ENABLED=True does not support partial enablement. "
            "Set MODEL.VR_OV.ENABLED=False or enable all canonical components: "
            f"{', '.join(disabled_components)}"
        )

    parser_out_dim = int(vr_ov.QUERY_PARSER.OUT_DIM)
    scene_graph_hidden = int(vr_ov.SCENE_GRAPH.HIDDEN_DIM)
    comp_matcher_hidden = int(vr_ov.COMP_MATCHER.HIDDEN_DIM)
    open_world_query_dim = int(getattr(cfg.MODEL.OpenWorldSAM2, "QUERY_DIM", scene_graph_hidden))

    if parser_out_dim != comp_matcher_hidden:
        raise ValueError(
            "MODEL.VR_OV.QUERY_PARSER.OUT_DIM must match "
            f"MODEL.VR_OV.COMP_MATCHER.HIDDEN_DIM for canonical VR-OV batching; got {parser_out_dim} vs {comp_matcher_hidden}."
        )
    if scene_graph_hidden != open_world_query_dim:
        raise ValueError(
            "MODEL.VR_OV.SCENE_GRAPH.HIDDEN_DIM must match "
            f"MODEL.OpenWorldSAM2.QUERY_DIM for canonical VR-OV prompt conditioning; got {scene_graph_hidden} vs {open_world_query_dim}."
        )
    if comp_matcher_hidden != scene_graph_hidden:
        raise ValueError(
            "MODEL.VR_OV.COMP_MATCHER.HIDDEN_DIM must match "
            f"MODEL.VR_OV.SCENE_GRAPH.HIDDEN_DIM for canonical VR-OV feature contracts; got {comp_matcher_hidden} vs {scene_graph_hidden}."
        )
