# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

from ._compat import CfgNode as CN


def add_open_world_sam2_config(cfg, *, include_vr_ov_compat: bool = True) -> None:
    from model.vr_ov_config import add_vr_ov_compat_config

    cfg.INPUT.DATASET_MAPPER_NAME = "open_world_instance"
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.MODEL.OpenWorldSAM2 = CN()
    cfg.MODEL.OpenWorldSAM2.EVF_CONFIG = "YxZhang/evf-sam2-multitask"
    cfg.MODEL.OpenWorldSAM2.TOKENIZER_CONFIG = "YxZhang/evf-sam2-multitask"
    cfg.MODEL.OpenWorldSAM2.LOCAL_EVF_CONFIG = ""
    cfg.MODEL.OpenWorldSAM2.LOCAL_TOKENIZER_CONFIG = ""
    cfg.MODEL.OpenWorldSAM2.HF_LOCAL_FILES_ONLY = False
    cfg.MODEL.OpenWorldSAM2.TORCH_DTYPE = "fp32"
    cfg.MODEL.OpenWorldSAM2.TRAIN_MASK_DECODER = False
    cfg.MODEL.OpenWorldSAM2.TRAIN_PROMPT_ENCODER = False
    cfg.MODEL.OpenWorldSAM2.TRAIN_VLM = True
    cfg.MODEL.OpenWorldSAM2.QUERY_DIM = 256
    cfg.MODEL.OpenWorldSAM2.VISION_PRETRAINED = "checkpoints/sam_vit_h_4b8939.pth"
    cfg.MODEL.OpenWorldSAM2.ENCODER_PRETRAINED = (
        "checkpoints/beit3_large_patch16_224.pth"
    )
    cfg.MODEL.OpenWorldSAM2.SAM_IOU = True

    cfg.MODEL.OpenWorldSAM2.NUM_OBJECT_QUERIES = 10
    cfg.MODEL.OpenWorldSAM2.TRAIN_TIE_BREAKER = True
    cfg.MODEL.OpenWorldSAM2.USE_VISUAL_TOKENS = True
    cfg.MODEL.OpenWorldSAM2.USE_CROSS_ATTENTION = False
    cfg.MODEL.OpenWorldSAM2.CROSS_ATTENTION_LAYERS = 1
    cfg.MODEL.OpenWorldSAM2.REASONSEG_ENABLED = False
    cfg.MODEL.OpenWorldSAM2.composition_mode = "composed_prompt"

    cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER = CN()
    cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.ENABLED = False
    cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.CHECKPOINT = ""
    cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.HIDDEN_DIM = 768
    cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.NUM_LAYERS = 2
    cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.DROPOUT = 0.1

    if include_vr_ov_compat:
        add_vr_ov_compat_config(cfg)

    cfg.MODEL.OpenWorldSAM2.TEST = CN()
    cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON = True
    cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.TOP_K_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.NMS_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.NMS_THRESHOLD = 0.0
    cfg.MODEL.OpenWorldSAM2.TEST.IOU_THRESHOLD = 0.0
    cfg.MODEL.OpenWorldSAM2.TEST.DETECTIONS_PER_IMAGE = 20
    cfg.MODEL.OpenWorldSAM2.TEST.TWO_STAGE_INFERENCE = False
    cfg.MODEL.OpenWorldSAM2.TEST.REFER_ON = False

    cfg.MODEL.OpenWorldSAM2.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.OpenWorldSAM2.DICE_WEIGHT = 1.0
    cfg.MODEL.OpenWorldSAM2.MASK_WEIGHT = 5.0
    cfg.MODEL.OpenWorldSAM2.OBJECTNESS_WEIGHT = 1.0
    cfg.MODEL.OpenWorldSAM2.NUM_OBJECT_QUERIES = 20
