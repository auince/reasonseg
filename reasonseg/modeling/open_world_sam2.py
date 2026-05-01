# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPossiblyUnboundVariable=false, reportOptionalMemberAccess=false, reportIndexIssue=false
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoTokenizer

from ._compat import (
    BitMasks,
    Boxes,
    ImageList,
    Instances,
    META_ARCH_REGISTRY,
    MetadataCatalog,
    configurable,
    resolve_repo_path,
)
from .criterion import SetCriterion
from .evf_sam2 import EvfSam2Model
from .matcher import HungarianMatcher
from .mlp import MLP
from .prompting import compose_reasonseg_prompt
# CompositionScores is imported lazily inside __init__() to avoid module-level dependency

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_local_asset_path(path: str) -> str:
    return resolve_repo_path(path, repo_root=_REPO_ROOT)


def _resolve_pretrained_source(config_source: str, local_source: str) -> str:
    if local_source:
        return _resolve_local_asset_path(local_source)
    return config_source


def _load_tokenizer(
    config_source: str,
    local_source: str,
    *,
    local_files_only: bool,
) -> AutoTokenizer:
    tokenizer_source = _resolve_pretrained_source(config_source, local_source)
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_source,
            padding_side="right",
            use_fast=False,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        raise RuntimeError(
            "Failed to load OpenWorldSAM2 tokenizer from "
            f"'{tokenizer_source}'. Set MODEL.OpenWorldSAM2.LOCAL_TOKENIZER_CONFIG "
            "to a local directory or disable MODEL.OpenWorldSAM2.HF_LOCAL_FILES_ONLY."
        ) from exc


def _resolve_torch_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.half
    return torch.float32


def _build_open_world_sam2_common_kwargs(
    cfg,
) -> tuple[dict[str, Any], int]:
    torch_dtype = _resolve_torch_dtype(cfg.MODEL.OpenWorldSAM2.TORCH_DTYPE)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "train_mask_decoder": cfg.MODEL.OpenWorldSAM2.TRAIN_MASK_DECODER,
        "train_prompt_encoder": cfg.MODEL.OpenWorldSAM2.TRAIN_PROMPT_ENCODER,
        "vision_pretrained": _resolve_local_asset_path(
            cfg.MODEL.OpenWorldSAM2.VISION_PRETRAINED
        ),
        "encoder_pretrained": _resolve_local_asset_path(
            cfg.MODEL.OpenWorldSAM2.ENCODER_PRETRAINED
        ),
    }
    local_files_only = cfg.MODEL.OpenWorldSAM2.HF_LOCAL_FILES_ONLY

    tokenizer = _load_tokenizer(
        cfg.MODEL.OpenWorldSAM2.TOKENIZER_CONFIG,
        cfg.MODEL.OpenWorldSAM2.LOCAL_TOKENIZER_CONFIG,
        local_files_only=local_files_only,
    )
    evf_model_source = _resolve_pretrained_source(
        cfg.MODEL.OpenWorldSAM2.EVF_CONFIG,
        cfg.MODEL.OpenWorldSAM2.LOCAL_EVF_CONFIG,
    )
    try:
        evf_sam2 = EvfSam2Model.from_pretrained(
            evf_model_source,
            low_cpu_mem_usage=False,
            local_files_only=local_files_only,
            **kwargs,
        )
    except OSError as exc:
        raise RuntimeError(
            "Failed to load OpenWorldSAM2 EVF-SAM2 weights from "
            f"'{evf_model_source}'. Set MODEL.OpenWorldSAM2.LOCAL_EVF_CONFIG "
            "to a local directory or disable MODEL.OpenWorldSAM2.HF_LOCAL_FILES_ONLY."
        ) from exc

    evf_sam2.config.eos_token_id = tokenizer.eos_token_id
    evf_sam2.config.bos_token_id = tokenizer.bos_token_id
    evf_sam2.config.pad_token_id = tokenizer.pad_token_id
    visual_model = evf_sam2.visual_model
    mm_extractor = evf_sam2.mm_extractor
    for param in mm_extractor.parameters():
        param.requires_grad = bool(cfg.MODEL.OpenWorldSAM2.TRAIN_VLM)
    text_hidden_fcs = evf_sam2.text_hidden_fcs
    for param in text_hidden_fcs.parameters():
        param.requires_grad = True

    query_dim = cfg.MODEL.OpenWorldSAM2.QUERY_DIM
    num_tokens = cfg.MODEL.OpenWorldSAM2.NUM_OBJECT_QUERIES
    positional_tokens = nn.Parameter(torch.randn(num_tokens, query_dim))
    positional_tokens.requires_grad = bool(cfg.MODEL.OpenWorldSAM2.TRAIN_TIE_BREAKER)

    no_object_weight = cfg.MODEL.OpenWorldSAM2.NO_OBJECT_WEIGHT
    dice_weight = cfg.MODEL.OpenWorldSAM2.DICE_WEIGHT
    mask_weight = cfg.MODEL.OpenWorldSAM2.MASK_WEIGHT
    objectness_weight = cfg.MODEL.OpenWorldSAM2.OBJECTNESS_WEIGHT
    use_cross_attention = getattr(
        cfg.MODEL.OpenWorldSAM2,
        "USE_CROSS_ATTENTION",
        False,
    )
    two_stage_inference = getattr(
        cfg.MODEL.OpenWorldSAM2.TEST,
        "TWO_STAGE_INFERENCE",
        False,
    )
    refer_on = getattr(cfg.MODEL.OpenWorldSAM2.TEST, "REFER_ON", False)

    matcher = HungarianMatcher(
        cost_class=objectness_weight,
        cost_mask=mask_weight,
        cost_dice=dice_weight,
    )
    criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": objectness_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        },
        eos_coef=no_object_weight,
        losses=["labels", "masks"],
    )

    train_datasets = getattr(cfg.DATASETS, "TRAIN", [])
    dataset_name = train_datasets[0] if train_datasets else "reasonseg_runtime"
    return {
        "evf_sam2": evf_sam2,
        "tokenizer": tokenizer,
        "visual_model": visual_model,
        "mm_extractor": mm_extractor,
        "text_hidden_fcs": text_hidden_fcs,
        "query_dim": query_dim,
        "num_tokens": num_tokens,
        "positional_tokens": positional_tokens,
        "criterion": criterion,
        "pixel_mean": cfg.MODEL.PIXEL_MEAN,
        "pixel_std": cfg.MODEL.PIXEL_STD,
        "dtype": torch_dtype,
        "semantic_on": cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON,
        "instance_on": cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON,
        "panoptic_on": cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON,
        "reasonseg_enabled": cfg.MODEL.OpenWorldSAM2.REASONSEG_ENABLED,
        "composition_mode": cfg.MODEL.OpenWorldSAM2.composition_mode,
        "top_k_on": cfg.MODEL.OpenWorldSAM2.TEST.TOP_K_ON,
        "nms_on": cfg.MODEL.OpenWorldSAM2.TEST.NMS_ON,
        "test_topk_per_image": cfg.MODEL.OpenWorldSAM2.TEST.DETECTIONS_PER_IMAGE,
        "nms_threshold": cfg.MODEL.OpenWorldSAM2.TEST.NMS_THRESHOLD,
        "iou_threshold": cfg.MODEL.OpenWorldSAM2.TEST.IOU_THRESHOLD,
        "sam_iou": cfg.MODEL.OpenWorldSAM2.SAM_IOU,
        "use_visual_tokens": cfg.MODEL.OpenWorldSAM2.USE_VISUAL_TOKENS,
        "use_cross_attention": use_cross_attention,
        "cross_attention_layers": cfg.MODEL.OpenWorldSAM2.CROSS_ATTENTION_LAYERS,
        "two_stage_inference": two_stage_inference,
        "refer_on": refer_on,
        "metadata": MetadataCatalog.get(dataset_name),
    }, int(evf_sam2.config.hidden_size)


def _build_open_world_sam2_parser_kwargs(cfg) -> dict[str, Any]:
    learned_parser_enabled = getattr(
        cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER, "ENABLED", False
    )
    parser_head: nn.Module | None = None
    if learned_parser_enabled:
        import sys
        from pathlib import Path as _Path

        _bioroot = _Path(__file__).resolve().parents[2] / "model" / "BIOtagging"
        if str(_bioroot) not in sys.path:
            sys.path.insert(0, str(_bioroot.parent.parent))
        from model.BIOtagging.query_parser_head import QueryParserHead

        parser_head = QueryParserHead(
            hidden_dim=cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.HIDDEN_DIM,
            num_tags=14,
            num_layers=cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.NUM_LAYERS,
            nhead=8,
            dim_feedforward=1024,
            dropout=cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.DROPOUT,
        )
        ckpt = cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.CHECKPOINT
        if ckpt:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            parser_head.load_state_dict(state)
            print(f"Loaded parser_head checkpoint: {ckpt}")

    return {
        "learned_parser_enabled": learned_parser_enabled,
        "parser_head": parser_head,
    }


def _build_vr_ov_module_kwargs(cfg, *, evf_hidden_size: int) -> dict[str, Any]:
    vr_ov_query_parser: nn.Module | None = None
    if cfg.MODEL.VR_OV.QUERY_PARSER.ENABLED:
        from model.query_parser import BIOQueryParser

        parser_gnn_hidden = int(cfg.MODEL.VR_OV.QUERY_PARSER.HIDDEN_DIM)
        parser_out_dim = int(cfg.MODEL.VR_OV.QUERY_PARSER.OUT_DIM)
        parser_ckpt = getattr(cfg.MODEL.VR_OV.QUERY_PARSER, "CHECKPOINT", "")
        vr_ov_query_parser = BIOQueryParser(
            parser_checkpoint=parser_ckpt if parser_ckpt else None,
            hidden_dim=evf_hidden_size,
            num_layers=int(cfg.MODEL.VR_OV.QUERY_PARSER.GNN_LAYERS),
            nhead=int(cfg.MODEL.VR_OV.QUERY_PARSER.GNN_HEADS),
            gnn_hidden=parser_gnn_hidden,
            gnn_out=parser_out_dim,
        )
        print("VR-OV: BIOQueryParser loaded")

    vr_ov_scene_graph: nn.Module | None = None
    if cfg.MODEL.VR_OV.SCENE_GRAPH.ENABLED:
        from model.scene_graph_encoder import SceneGraphVisualEncoder

        vr_ov_scene_graph = SceneGraphVisualEncoder(
            hidden_dim=int(cfg.MODEL.VR_OV.SCENE_GRAPH.HIDDEN_DIM),
            num_hoi_tokens=int(cfg.MODEL.VR_OV.SCENE_GRAPH.HOI_TOKENS),
            region_topk=int(cfg.MODEL.VR_OV.SCENE_GRAPH.REGION_TOPK),
        )
        print("VR-OV: SceneGraphVisualEncoder loaded")

    vr_ov_refine_decoder: nn.Module | None = None
    if cfg.MODEL.VR_OV.REFINE_DECODER.ENABLED:
        from model.refinement_decoder import IterativeRefinementDecoder

        max_iter = getattr(cfg.MODEL.VR_OV.REFINE_DECODER, "MAX_ITER", 3)
        vr_ov_refine_decoder = IterativeRefinementDecoder(
            hidden_dim=256,
            max_iter=max_iter,
        )
        vr_ov_refine_decoder.attr_threshold.data.fill_(
            float(cfg.MODEL.VR_OV.REFINE_DECODER.ATTR_THRESHOLD)
        )
        print("VR-OV: IterativeRefinementDecoder loaded")

    vr_ov_comp_matcher: nn.Module | None = None
    if cfg.MODEL.VR_OV.COMP_MATCHER.ENABLED:
        from model.compositional_matcher import CompositionalFeatureMatcher

        vr_ov_comp_matcher = CompositionalFeatureMatcher(
            hidden_dim=int(cfg.MODEL.VR_OV.COMP_MATCHER.HIDDEN_DIM),
            cmf_layers=int(cfg.MODEL.VR_OV.COMP_MATCHER.CMF_LAYERS),
        )
        print("VR-OV: CompositionalFeatureMatcher loaded")

    from model.vr_ov_losses import VR_OVLosses

    vr_ov_losses = None
    vr_ov_loss_config = None
    if cfg.MODEL.VR_OV.ENABLED:
        loss_cfg = getattr(cfg.MODEL.VR_OV, "LOSS", None)
        if loss_cfg is not None:
            vr_ov_loss_config = {
                "mask_enabled": bool(getattr(loss_cfg, "MASK_ENABLED", True)),
                "attr_enabled": bool(getattr(loss_cfg, "ATTR_ENABLED", False)),
                "rel_enabled": bool(getattr(loss_cfg, "REL_ENABLED", False)),
                "act_enabled": bool(getattr(loss_cfg, "ACT_ENABLED", False)),
                "compose_enabled": bool(getattr(loss_cfg, "COMPOSE_ENABLED", False)),
                "lambda_mask": float(getattr(loss_cfg, "LAMBDA_MASK", 5.0)),
                "lambda_attr": float(getattr(loss_cfg, "LAMBDA_ATTR", 1.0)),
                "lambda_rel": float(getattr(loss_cfg, "LAMBDA_REL", 0.5)),
                "lambda_act": float(getattr(loss_cfg, "LAMBDA_ACT", 0.5)),
                "lambda_compose": float(getattr(loss_cfg, "LAMBDA_COMPOSE", 0.3)),
            }
            if vr_ov_loss_config["mask_enabled"]:
                vr_ov_losses = VR_OVLosses(
                    lambda_mask=vr_ov_loss_config["lambda_mask"],
                    lambda_attr=vr_ov_loss_config["lambda_attr"],
                    lambda_rel=vr_ov_loss_config["lambda_rel"],
                    lambda_act=vr_ov_loss_config["lambda_act"],
                    lambda_compose=vr_ov_loss_config["lambda_compose"],
                )

    return {
        "vr_ov_query_parser": vr_ov_query_parser,
        "vr_ov_scene_graph": vr_ov_scene_graph,
        "vr_ov_comp_matcher": vr_ov_comp_matcher,
        "vr_ov_refine_decoder": vr_ov_refine_decoder,
        "vr_ov_losses": vr_ov_losses,
        "vr_ov_loss_config": vr_ov_loss_config,
    }


@META_ARCH_REGISTRY.register()
class OpenWorldSAM2(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        evf_sam2: EvfSam2Model,
        tokenizer: AutoTokenizer,
        visual_model: nn.Module,
        mm_extractor: nn.Module,
        text_hidden_fcs: nn.ModuleList,
        query_dim: int,
        num_tokens: int,
        positional_tokens: nn.Parameter,
        criterion: nn.Module,
        pixel_mean,
        pixel_std,
        dtype: torch.dtype,
        test_topk_per_image: int,
        top_k_on: bool,
        nms_on: bool,
        nms_threshold: float,
        iou_threshold: float,
        semantic_on: bool,
        instance_on: bool,
        panoptic_on: bool,
        sam_iou: bool,
        reasonseg_enabled: bool = False,
        learned_parser_enabled: bool = False,
        parser_head: nn.Module | None = None,
        vr_ov_query_parser: nn.Module | None = None,
        vr_ov_scene_graph: nn.Module | None = None,
        vr_ov_refine_decoder: nn.Module | None = None,
        vr_ov_comp_matcher: nn.Module | None = None,
        vr_ov_losses: nn.Module | None = None,
        vr_ov_loss_config: dict[str, Any] | None = None,
        composition_mode: str = "composed_prompt",
        use_visual_tokens: bool = True,
        use_cross_attention: bool = False,
        cross_attention_layers: int = 3,
        two_stage_inference: bool = False,
        refer_on: bool = False,
        metadata: MetadataCatalog | None = None,
    ) -> None:
        super().__init__()
        self.evf_sam2 = evf_sam2
        self.tokenizer = tokenizer
        self.visual_model = visual_model
        self.mm_extractor = mm_extractor
        self.text_hidden_fcs = text_hidden_fcs
        self.query_dim = query_dim
        self.num_tokens = num_tokens
        self.criterion = criterion
        self.positional_tokens = positional_tokens
        self.use_visual_tokens = use_visual_tokens
        self.use_cross_attention = use_cross_attention
        self.metadata = metadata
        self.two_stage_inference = two_stage_inference
        self.refer_on = refer_on
        if self.use_cross_attention:
            self.cross_attention_transformer = CrossAttentionTransformer(
                embedding_dim=256,
                num_heads=8,
                mlp_dim=query_dim * 4,
                num_layers=cross_attention_layers,
                dropout=0.1,
            )
        if not sam_iou:
            self.objectness_prediction_head = MLP(
                input_dim=evf_sam2.visual_model.memory_attention.d_model,
                hidden_dim=256,
                output_dim=1,
                num_layers=3,
                sigmoid_output=True,
            )

        self.register_buffer(
            "pixel_mean",
            torch.tensor(pixel_mean).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(pixel_std).view(-1, 1, 1),
            persistent=False,
        )
        self.dtype = dtype
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.sam_iou = sam_iou
        self.reasonseg_enabled = reasonseg_enabled
        self.learned_parser_enabled = learned_parser_enabled
        self.parser_head = parser_head
        self.vr_ov_query_parser = vr_ov_query_parser
        self.vr_ov_scene_graph = vr_ov_scene_graph
        self.vr_ov_refine_decoder = vr_ov_refine_decoder
        self.vr_ov_comp_matcher = vr_ov_comp_matcher
        self.vr_ov_losses = vr_ov_losses
        self.vr_ov_loss_config = vr_ov_loss_config
        self.composition_mode = composition_mode
        self.top_k_on = top_k_on
        self.nms_on = nms_on
        self.test_topk_per_image = test_topk_per_image
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

    @classmethod
    def from_config(cls, cfg):
        common_kwargs, _ = _build_open_world_sam2_common_kwargs(cfg)
        common_kwargs.update(_build_open_world_sam2_parser_kwargs(cfg))
        return common_kwargs

    @property
    def device(self):
        return self.pixel_mean.device

    def tokenize_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = [
            self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id)
        truncate_len = self.tokenizer.model_max_length
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
        return input_ids, attention_masks

    def _select_prompts(self, batch_input: Mapping[str, Any]) -> list[str]:
        prompts = None
        if self.learned_parser_enabled and hasattr(self, '_last_tag_logits'):
            prompts = batch_input.get("composed_prompt")
        if self.reasonseg_enabled and self.composition_mode == "composed_prompt":
            prompts = batch_input.get("composed_prompt")
        if (
            prompts is None
            and self.reasonseg_enabled
            and self.composition_mode == "composed_prompt"
            and "query_struct" in batch_input
        ):
            query_texts = batch_input.get("query_text", batch_input["prompt"])
            prompts = [
                compose_reasonseg_prompt(query_struct, fallback_text=query_text)
                for query_struct, query_text in zip(
                    batch_input["query_struct"],
                    query_texts,
                )
            ]
        if prompts is None:
            prompts = batch_input["prompt"]
        return list(prompts)

    def build_prompt_batch(
        self, batched_inputs: list[Mapping[str, Any]]
    ) -> tuple[list[str], list[int]]:
        offset = [0]
        all_prompts: list[str] = []
        for batch_input in batched_inputs:
            prompts = self._select_prompts(batch_input)
            all_prompts.extend(prompts)
            offset.append(offset[-1] + len(prompts))
        return all_prompts, offset

    @staticmethod
    def _apply_vr_ov_scene_graph_prompt_context(
        batch_feat_with_tokens: torch.Tensor,
        hoi_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if hoi_tokens is None:
            return batch_feat_with_tokens
        hoi_context = hoi_tokens.mean(dim=0).to(
            device=batch_feat_with_tokens.device,
            dtype=batch_feat_with_tokens.dtype,
        )
        return batch_feat_with_tokens + hoi_context.view(1, 1, -1)

    def _assert_forward_backend_available(self) -> None:
        if not hasattr(self.visual_model, "forward_image"):
            raise RuntimeError(
                "OpenWorldSAM2 imported successfully, but its heavyweight vision backend is not "
                "available for execution in this Task 6 slice."
            )

    def _prepare_input_tensors(
        self, batched_inputs: list[Mapping[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        images = [
            x["image"].to(dtype=self.dtype, device=self.device) for x in batched_inputs
        ]
        original_size_list = [(x["height"], x["width"]) for x in batched_inputs]
        images_evf = [
            x["evf_image"].to(dtype=self.dtype, device=self.device)
            for x in batched_inputs
        ]
        return (
            ImageList.from_tensors(images, 1024).tensor,
            ImageList.from_tensors(images_evf, 224).tensor,
            original_size_list,
        )

    def _encode_backbone_features(
        self, images: torch.Tensor, batch_size: int
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        backbone_out = self.visual_model.forward_image(images)
        _, image_embeddings, _, _ = self.visual_model._prepare_backbone_features(
            backbone_out
        )
        image_embeddings = [feature.to(images.dtype) for feature in image_embeddings]
        if self.visual_model.directly_add_no_mem_embed:
            image_embeddings[-1] = image_embeddings[-1] + self.visual_model.no_mem_embed

        feats = [
            feature.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feature, feat_size in zip(
                image_embeddings[::-1],
                self._bb_feat_sizes[::-1],
            )
        ][::-1]
        return backbone_out, {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def _encode_text_prompts(
        self,
        batched_inputs: list[Mapping[str, Any]],
        images_evf: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        list[int],
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        all_prompts, offset = self.build_prompt_batch(batched_inputs)
        input_ids, attention_masks = self.tokenize_prompts(all_prompts)
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        if self.use_visual_tokens:
            images_evf_list = []
            for index in range(len(offset) - 1):
                start_idx, end_idx = offset[index], offset[index + 1]
                images_evf_list.append(
                    images_evf[index]
                    .unsqueeze(0)
                    .expand(end_idx - start_idx, -1, -1, -1)
                    .contiguous()
                )
            visual_tokens = torch.cat(images_evf_list, dim=0)
        else:
            visual_tokens = None

        output = self.mm_extractor.beit3(
            visual_tokens=visual_tokens,
            textual_tokens=input_ids,
            text_padding_position=~attention_masks,
        )
        feat = self.text_hidden_fcs[0](output["encoder_out"][:, :1, ...])
        feat = torch.split(
            feat,
            [offset[index + 1] - offset[index] for index in range(len(offset) - 1)],
        )
        return feat, offset, input_ids, attention_masks, output

    def _update_learned_parser_logits(self, output: dict[str, Any]) -> None:
        if self.learned_parser_enabled and self.parser_head is not None:
            self._last_tag_logits = self.parser_head(output["encoder_out"])

    def _new_composition_scores(self):
        from model.vr_ov_types import CompositionScores

        return CompositionScores()

    @staticmethod
    def _validate_query_graph(query_graph: Any, *, context: str) -> None:
        from model.vr_ov_types import QueryGraph

        if not isinstance(query_graph, QueryGraph):
            raise TypeError(f"{context} must be a QueryGraph, got {type(query_graph).__name__}.")
        if len(query_graph.nodes) != 4:
            raise ValueError(
                f"{context} must contain exactly 4 semantic nodes; got {len(query_graph.nodes)}."
            )
        if len(query_graph.node_types) != 4:
            raise ValueError(
                f"{context} must declare 4 node types; got {len(query_graph.node_types)}."
            )
        if not isinstance(query_graph.edges, torch.Tensor):
            raise TypeError(f"{context}.edges must be a torch.Tensor.")
        if query_graph.edges.ndim != 2 or query_graph.edges.shape[0] != 2:
            raise ValueError(
                f"{context}.edges must have shape [2, num_edges]; got {tuple(query_graph.edges.shape)}."
            )

        node_dim: int | None = None
        for index, node in enumerate(query_graph.nodes):
            if not isinstance(node, torch.Tensor):
                raise TypeError(
                    f"{context}.nodes[{index}] must be a torch.Tensor, got {type(node).__name__}."
                )
            if node.ndim != 1:
                raise ValueError(
                    f"{context}.nodes[{index}] must be 1-D; got shape {tuple(node.shape)}."
                )
            if node_dim is None:
                node_dim = int(node.shape[0])
            elif int(node.shape[0]) != node_dim:
                raise ValueError(
                    f"{context} node feature dims must match; expected {node_dim}, got {int(node.shape[0])} at index {index}."
                )

    @staticmethod
    def _validate_scene_graph_outputs(
        scene_graph_outputs: Any,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(scene_graph_outputs, tuple) or len(scene_graph_outputs) != 3:
            raise ValueError(
                "VR-OV scene graph encoder must return a 3-tuple of (hoi_tokens, regions, relation_logits)."
            )
        hoi_tokens, regions, relation_logits = scene_graph_outputs
        for name, tensor in (
            ("hoi_tokens", hoi_tokens),
            ("regions", regions),
            ("relation_logits", relation_logits),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"VR-OV scene graph output '{name}' must be a torch.Tensor.")
            if tensor.ndim != 3:
                raise ValueError(
                    f"VR-OV scene graph output '{name}' must be rank-3 [B, N, C]; got {tuple(tensor.shape)}."
                )
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"VR-OV scene graph output '{name}' batch size must be {batch_size}; got {tensor.shape[0]}."
                )
        if hoi_tokens.shape[-1] != regions.shape[-1]:
            raise ValueError(
                "VR-OV scene graph outputs must share the same hidden dim for hoi_tokens and regions."
            )
        return hoi_tokens, regions, relation_logits

    @staticmethod
    def _validate_composition_scores(
        comp_scores: Any,
        *,
        expected_batch: int,
        expected_hw: tuple[int, int],
        require_all_modalities: bool,
        context: str,
    ):
        from model.vr_ov_types import CompositionScores

        if not isinstance(comp_scores, CompositionScores):
            raise TypeError(
                f"{context} must return CompositionScores, got {type(comp_scores).__name__}."
            )

        height, width = expected_hw
        for field_name in ("cat_feat", "attr_feat", "rel_feat", "act_feat"):
            value = getattr(comp_scores, field_name)
            if value is None:
                if require_all_modalities:
                    raise ValueError(
                        f"{context} must populate '{field_name}' for the canonical VR-OV path."
                    )
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{context}.{field_name} must be a torch.Tensor.")
            if value.ndim != 4:
                raise ValueError(
                    f"{context}.{field_name} must have shape [B, 1, H, W]; got {tuple(value.shape)}."
                )
            if tuple(value.shape) != (expected_batch, 1, height, width):
                raise ValueError(
                    f"{context}.{field_name} must have shape {(expected_batch, 1, height, width)}; got {tuple(value.shape)}."
                )
        for sub_field in ("attr_color", "attr_material", "attr_size"):
            value = getattr(comp_scores, sub_field, None)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"{context}.{sub_field} must be a torch.Tensor.")
        return comp_scores

    @staticmethod
    def _expand_composition_scores_for_masks(
        comp_scores,
        *,
        prompt_count: int,
        mask_count: int,
    ):
        from model.vr_ov_types import CompositionScores

        if prompt_count <= 0:
            raise ValueError("VR-OV refinement requires at least one prompt-level composition score.")
        if mask_count % prompt_count != 0:
            raise ValueError(
                f"VR-OV refinement expects mask_count ({mask_count}) to be divisible by prompt_count ({prompt_count})."
            )
        repeats = mask_count // prompt_count
        if repeats == 1:
            return comp_scores
        return CompositionScores(
            cat_feat=comp_scores.cat_feat.repeat_interleave(repeats, dim=0) if comp_scores.cat_feat is not None else None,
            attr_feat=comp_scores.attr_feat.repeat_interleave(repeats, dim=0) if comp_scores.attr_feat is not None else None,
            attr_color=comp_scores.attr_color.repeat_interleave(repeats, dim=0) if comp_scores.attr_color is not None else None,
            attr_material=comp_scores.attr_material.repeat_interleave(repeats, dim=0) if comp_scores.attr_material is not None else None,
            attr_size=comp_scores.attr_size.repeat_interleave(repeats, dim=0) if comp_scores.attr_size is not None else None,
            rel_feat=comp_scores.rel_feat.repeat_interleave(repeats, dim=0) if comp_scores.rel_feat is not None else None,
            act_feat=comp_scores.act_feat.repeat_interleave(repeats, dim=0) if comp_scores.act_feat is not None else None,
        )

    @staticmethod
    def _resize_composition_scores(
        comp_scores,
        *,
        target_hw: tuple[int, int],
    ):
        from model.vr_ov_types import CompositionScores

        target_height, target_width = target_hw

        def _resize(score: torch.Tensor | None) -> torch.Tensor | None:
            if score is None or tuple(score.shape[-2:]) == (target_height, target_width):
                return score
            return F.interpolate(
                score,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )

        return CompositionScores(
            cat_feat=_resize(comp_scores.cat_feat),
            attr_feat=_resize(comp_scores.attr_feat),
            attr_color=_resize(comp_scores.attr_color),
            attr_material=_resize(comp_scores.attr_material),
            attr_size=_resize(comp_scores.attr_size),
            rel_feat=_resize(comp_scores.rel_feat),
            act_feat=_resize(comp_scores.act_feat),
        )

    def _run_vr_ov_scene_graph(
        self, backbone_out: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.vr_ov_scene_graph is None:
            return None
        features = {
            "high_res_feats": backbone_out["backbone_fpn"][:-1],
            "image_embed": backbone_out["backbone_fpn"][-1],
        }
        return self._validate_scene_graph_outputs(
            self.vr_ov_scene_graph(features),
            batch_size=int(features["image_embed"].shape[0]),
        )

    @staticmethod
    def _prompt_payloads_for_image(batch_input: Mapping[str, Any]) -> tuple[list[Any] | None, list[Any] | None]:
        query_graphs = batch_input.get("query_graphs")
        if isinstance(query_graphs, list):
            return query_graphs, None

        single_query_graph = batch_input.get("query_graph")
        if single_query_graph is not None:
            return [single_query_graph], None

        query_structs = batch_input.get("query_struct")
        if isinstance(query_structs, list):
            return None, query_structs
        return None, None

    def _run_vr_ov_query_parser(
        self,
        *,
        batched_inputs: list[Mapping[str, Any]],
        offset: Sequence[int],
        output: dict[str, Any],
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
    ) -> list[list[Any]] | None:
        if self.vr_ov_query_parser is None:
            return None
        if len(offset) != len(batched_inputs) + 1:
            raise ValueError(
                f"VR-OV query parser offsets must align with the image batch; got {len(offset)} offsets for {len(batched_inputs)} images."
            )
        encoder_out = output["encoder_out"]
        if not isinstance(encoder_out, torch.Tensor) or encoder_out.ndim != 3:
            raise ValueError("VR-OV query parser expects encoder_out with shape [num_prompts, seq_len, hidden_dim].")
        parsed_query_graphs: list[list[Any]] = []

        for image_index, batch_input in enumerate(batched_inputs):
            prompt_query_graphs, prompt_query_structs = self._prompt_payloads_for_image(
                batch_input
            )
            start_index = int(offset[image_index])
            end_index = int(offset[image_index + 1])
            expected_prompt_count = end_index - start_index

            if prompt_query_graphs is not None:
                if len(prompt_query_graphs) != expected_prompt_count:
                    raise ValueError(
                        f"VR-OV query_graphs count mismatch for image {image_index}: expected {expected_prompt_count}, got {len(prompt_query_graphs)}."
                    )
                for prompt_index, query_graph in enumerate(prompt_query_graphs):
                    self._validate_query_graph(
                        query_graph,
                        context=f"VR-OV query_graphs[{image_index}][{prompt_index}]",
                    )
                parsed_query_graphs.append(list(prompt_query_graphs))
                continue

            if (
                prompt_query_structs is not None
                and len(prompt_query_structs) != expected_prompt_count
            ):
                raise ValueError(
                    f"VR-OV query_struct count mismatch for image {image_index}: expected {expected_prompt_count}, got {len(prompt_query_structs)}."
                )

            image_query_graphs: list[Any] = []
            for prompt_index, flat_index in enumerate(range(start_index, end_index)):
                prompt_length = int(attention_masks[flat_index].sum().item())
                prompt_tokens = self.tokenizer.convert_ids_to_tokens(
                    input_ids[flat_index, :prompt_length].tolist()
                )
                prompt_query_struct = None
                if prompt_query_structs is not None and prompt_index < len(prompt_query_structs):
                    prompt_query_struct = prompt_query_structs[prompt_index]
                query_graph = self.vr_ov_query_parser(
                        beit3_hidden=encoder_out[flat_index : flat_index + 1, :prompt_length],
                        attention_mask=attention_masks[flat_index, :prompt_length].unsqueeze(0),
                        tokens_list=prompt_tokens,
                        query_struct=prompt_query_struct,
                    )
                self._validate_query_graph(
                    query_graph,
                    context=f"VR-OV parser output[{image_index}][{prompt_index}]",
                )
                image_query_graphs.append(query_graph)
            parsed_query_graphs.append(image_query_graphs)

        return parsed_query_graphs

    def _run_vr_ov_comp_matcher(
        self,
        *,
        query_graphs: Sequence[Any] | None,
        image_embed: torch.Tensor,
        return_intermediates: bool = False,
    ):
        comp_scores = self._new_composition_scores()
        if self.vr_ov_comp_matcher is None:
            return (comp_scores, None) if return_intermediates else comp_scores
        if not query_graphs:
            raise ValueError("VR-OV composition matcher requires at least one query graph.")

        for prompt_index, query_graph in enumerate(query_graphs):
            self._validate_query_graph(
                query_graph,
                context=f"VR-OV composition input[{prompt_index}]",
            )
        if image_embed.ndim != 3:
            raise ValueError(
                f"VR-OV composition matcher expects image_embed with shape [C, H, W]; got {tuple(image_embed.shape)}."
            )

        query_nodes = torch.stack(
            [torch.stack(query_graph.nodes, dim=0) for query_graph in query_graphs],
            dim=0,
        )
        prompt_count = query_nodes.shape[0]
        vis_feat = (
            image_embed.flatten(1)
            .transpose(0, 1)
            .unsqueeze(0)
            .expand(prompt_count, -1, -1)
            .contiguous()
        )
        image_embed_batch = image_embed.unsqueeze(0).expand(prompt_count, -1, -1, -1).contiguous()
        result = self.vr_ov_comp_matcher(
            query_nodes,
            vis_feat,
            image_embed_batch,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            comp_scores, _, intermediates = result
        else:
            comp_scores, _ = result
            intermediates = None
        comp_scores = self._validate_composition_scores(
            comp_scores,
            expected_batch=prompt_count,
            expected_hw=(int(image_embed.shape[-2]), int(image_embed.shape[-1])),
            require_all_modalities=True,
            context="VR-OV composition matcher output",
        )
        return (comp_scores, intermediates) if return_intermediates else comp_scores

    def _build_prompt_tokens(self, img_feat: torch.Tensor) -> torch.Tensor:
        batch_feat_with_tokens = []
        for prompt_feat in img_feat:
            feat_repeated = prompt_feat.expand(self.num_tokens, -1, -1)
            batch_feat_with_tokens.append(
                feat_repeated + self.positional_tokens.unsqueeze(1)
            )
        return torch.cat(batch_feat_with_tokens, dim=0)

    def _apply_cross_attention_prompt_context(
        self,
        batch_feat_with_tokens: torch.Tensor,
        image_embed: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_cross_attention:
            return batch_feat_with_tokens
        flat_image_embed = image_embed.flatten(1).transpose(0, 1).unsqueeze(0)
        original_batch_feat = batch_feat_with_tokens
        reshaped_batch_feat = (
            batch_feat_with_tokens.squeeze(1)
            if batch_feat_with_tokens.dim() == 3
            else batch_feat_with_tokens
        )
        enhanced_batch_feat = self.cross_attention_transformer(
            reshaped_batch_feat.unsqueeze(0),
            flat_image_embed,
        ).squeeze(0)
        if batch_feat_with_tokens.dim() == 2:
            enhanced_batch_feat = enhanced_batch_feat.unsqueeze(1)
        return original_batch_feat + enhanced_batch_feat

    def _decode_masks_for_image(
        self,
        *,
        features: dict[str, torch.Tensor],
        batch_feat_with_tokens: torch.Tensor,
        img_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        sparse_embeddings, dense_embeddings = self.visual_model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=batch_feat_with_tokens,
        )
        sparse_embeddings = sparse_embeddings.to(batch_feat_with_tokens.dtype)
        high_res_features = [
            feat_level[img_idx].unsqueeze(0) for feat_level in features["high_res_feats"]
        ]
        low_res_masks, iou_pred, sam_tokens_out, _ = self.visual_model.sam_mask_decoder(
            image_embeddings=features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.visual_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        pred_logits = (
            self.objectness_prediction_head(sam_tokens_out.squeeze(1))
            if not self.sam_iou
            else iou_pred
        )
        return low_res_masks, pred_logits, high_res_features

    def _class_labels_for_image(
        self,
        *,
        pred_masks: torch.Tensor,
        unique_categories: list[int],
    ) -> torch.Tensor:
        num_total_masks = len(pred_masks)
        class_indices = torch.div(
            torch.arange(num_total_masks, device=self.device),
            self.num_tokens,
            rounding_mode="floor",
        )
        return torch.tensor(
            [unique_categories[index] for index in class_indices],
            dtype=torch.int64,
            device=self.device,
        )

    def _run_two_stage_inference(
        self,
        *,
        low_res_masks: torch.Tensor,
        pred_logits: torch.Tensor,
        class_labels: torch.Tensor,
        features: dict[str, torch.Tensor],
        high_res_features: list[torch.Tensor],
        img_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        iou_scores = pred_logits.squeeze(1) if pred_logits.dim() > 1 else pred_logits
        keep_indices = iou_scores >= self.iou_threshold
        if keep_indices.sum() <= 0:
            return low_res_masks, pred_logits, class_labels

        filtered_masks = low_res_masks[keep_indices]
        filtered_class_labels = class_labels[keep_indices]
        sparse_embeddings, dense_embeddings = self.visual_model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=filtered_masks,
            text_embeds=None,
        )
        low_res_masks, pred_logits, _, _ = self.visual_model.sam_mask_decoder(
            image_embeddings=features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.visual_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        return low_res_masks, pred_logits, filtered_class_labels

    def _build_processed_result(
        self,
        *,
        low_res_masks: torch.Tensor,
        pred_logits: torch.Tensor,
        class_labels: torch.Tensor,
        features: dict[str, torch.Tensor],
        high_res_features: list[torch.Tensor],
        original_hw: tuple[int, int],
        img_idx: int,
        comp_scores: Any | None = None,
        use_refine_decoder: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if use_refine_decoder and self.vr_ov_refine_decoder is not None:
            if comp_scores is None:
                raise ValueError(
                    "VR-OV refinement requires composition scores; received None."
                )
            visual_feat = features["image_embed"][img_idx].unsqueeze(0)
            if tuple(visual_feat.shape[-2:]) != tuple(low_res_masks.shape[-2:]):
                visual_feat = F.interpolate(
                    visual_feat,
                    size=(int(low_res_masks.shape[-2]), int(low_res_masks.shape[-1])),
                    mode="bilinear",
                    align_corners=False,
                )
            num_masks = low_res_masks.shape[0]
            if num_masks % self.num_tokens != 0:
                raise ValueError(
                    f"VR-OV refinement expects num_masks ({num_masks}) to be divisible by num_tokens ({self.num_tokens})."
                )
            prompt_count = num_masks // self.num_tokens
            aligned_scores = self._resize_composition_scores(
                comp_scores,
                target_hw=(int(low_res_masks.shape[-2]), int(low_res_masks.shape[-1])),
            )
            validated_scores = self._validate_composition_scores(
                aligned_scores,
                expected_batch=prompt_count,
                expected_hw=(int(low_res_masks.shape[-2]), int(low_res_masks.shape[-1])),
                require_all_modalities=True,
                context="VR-OV refinement input",
            )
            expanded_scores = self._expand_composition_scores_for_masks(
                validated_scores,
                prompt_count=prompt_count,
                mask_count=num_masks,
            )
            if num_masks > 1:
                visual_feat = visual_feat.expand(num_masks, -1, -1, -1)
            low_res_masks, refine_history = self.vr_ov_refine_decoder(
                low_res_masks,
                expanded_scores,
                visual_feat,
            )
            result["refine_history"] = [
                {
                    "stage": state.stage,
                    "iou": state.iou,
                    "converged": state.converged,
                }
                for state in refine_history
            ]
        elif self.two_stage_inference:
            low_res_masks, pred_logits, class_labels = self._run_two_stage_inference(
                low_res_masks=low_res_masks,
                pred_logits=pred_logits,
                class_labels=class_labels,
                features=features,
                high_res_features=high_res_features,
                img_idx=img_idx,
            )

        pred_masks = self.postprocess_masks(low_res_masks, orig_hw=original_hw)
        if self.refer_on:
            refer_masks, refer_scores = self.refer_inference(
                pred_masks,
                pred_logits,
                class_labels,
            )
            result["grounding_mask"] = refer_masks
            result["grounding_scores"] = refer_scores

        if self.instance_on:
            result["instances"] = self.instance_inference(
                pred_masks,
                pred_logits,
                class_labels,
            )

        if self.panoptic_on:
            result["panoptic_seg"] = self.panoptic_inference(
                pred_logits,
                pred_masks,
                class_labels,
            )

        if self.semantic_on:
            num_classes = len(getattr(self.metadata, "stuff_classes", []))
            mask_cls = torch.zeros(
                (pred_masks.shape[0], num_classes + 1),
                device=self.device,
            )
            for index, (cls_id, score) in enumerate(
                zip(class_labels, pred_logits.squeeze(1))
            ):
                mask_cls[index, cls_id] = score
            result["sem_seg"] = self.semantic_inference(
                mask_cls,
                pred_masks,
                keep_sem_bgd=False,
            )
        if comp_scores is not None:
            result["vr_ov_compositional"] = {
                "has_comp_scores": True,
                "modalities": [
                    field_name
                    for field_name in ("cat_feat", "attr_feat", "rel_feat", "act_feat")
                    if getattr(comp_scores, field_name, None) is not None
                ],
            }
        return result

    def _accumulate_prompt_losses(
        self,
        *,
        pred_masks: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_instances: Any,
        all_losses: dict[str, list[torch.Tensor]],
        return_intermediate: bool,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]] | None:
        if not isinstance(gt_instances, list):
            gt_instances = [gt_instances]

        pred_masks_list = torch.split(pred_masks, [self.num_tokens] * len(gt_instances))
        pred_logits_list = torch.split(pred_logits, [self.num_tokens] * len(gt_instances))
        for prompt_idx, prompt_target in enumerate(gt_instances):
            prompt_outputs = {
                "pred_masks": pred_masks_list[prompt_idx].unsqueeze(0),
                "pred_logits": pred_logits_list[prompt_idx].unsqueeze(0),
            }
            prompt_targets = self.prepare_targets([prompt_target])
            if return_intermediate and prompt_idx == 0:
                return prompt_outputs, prompt_targets
            prompt_losses = self.criterion(
                prompt_outputs,
                prompt_targets,
                reduce_num_masks=False,
            )
            criterion_weight_dict = self.criterion.weight_dict
            for key, value in prompt_losses.items():
                if key in criterion_weight_dict:
                    all_losses[key].append(value * criterion_weight_dict[key])
        return None

    def forward(self, batched_inputs, return_intermediate: bool = False):
        self._assert_forward_backend_available()
        images, images_evf, original_size_list = self._prepare_input_tensors(
            batched_inputs
        )
        batch_size = len(batched_inputs)
        _, features = self._encode_backbone_features(images, batch_size)
        feat, _, _, _, output = self._encode_text_prompts(batched_inputs, images_evf)
        self._update_learned_parser_logits(output)
        all_losses: dict[str, list[torch.Tensor]] = defaultdict(list)
        processed_results: list[dict[str, Any]] = []

        for img_idx in range(batch_size):
            batch_feat_with_tokens = self._build_prompt_tokens(feat[img_idx])
            batch_feat_with_tokens = self._apply_cross_attention_prompt_context(
                batch_feat_with_tokens,
                features["image_embed"][img_idx],
            )
            low_res_masks, pred_logits, high_res_features = self._decode_masks_for_image(
                features=features,
                batch_feat_with_tokens=batch_feat_with_tokens,
                img_idx=img_idx,
            )
            pred_masks = low_res_masks.squeeze(1)

            if not self.training:
                class_labels = self._class_labels_for_image(
                    pred_masks=pred_masks,
                    unique_categories=batched_inputs[img_idx]["unique_categories"],
                )
                processed_results.append(
                    self._build_processed_result(
                        low_res_masks=low_res_masks,
                        pred_logits=pred_logits,
                        class_labels=class_labels,
                        features=features,
                        high_res_features=high_res_features,
                        original_hw=original_size_list[img_idx],
                        img_idx=img_idx,
                    )
                )
                continue

            intermediate = self._accumulate_prompt_losses(
                pred_masks=pred_masks,
                pred_logits=pred_logits,
                gt_instances=batched_inputs[img_idx]["instances"],
                all_losses=all_losses,
                return_intermediate=return_intermediate,
            )
            if intermediate is not None:
                return intermediate

        if self.training:
            return {
                key: torch.stack(values).mean() for key, values in all_losses.items()
            }
        return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks.to(
                dtype=self.dtype, device=self.device
            )
            labels = torch.zeros_like(targets_per_image.gt_classes).to(
                device=self.device
            )
            new_targets.append({"labels": labels, "masks": gt_masks})
        return new_targets

    def instance_inference(self, pred_masks, iou_scores, class_labels):
        image_size = pred_masks.shape[-2:]
        iou_scores = iou_scores.squeeze(1)
        pred_masks = pred_masks.squeeze(1)

        if self.panoptic_on:
            thing_map = getattr(self.metadata, "thing_dataset_id_to_contiguous_id", {})
            keep = torch.zeros_like(iou_scores).bool()
            for index, label in enumerate(class_labels):
                keep[index] = label in thing_map.values()
            pred_masks = pred_masks[keep]
            iou_scores = iou_scores[keep]
            class_labels = class_labels[keep]

        if self.top_k_on:
            top_k = min(self.test_topk_per_image, pred_masks.shape[0])
            top_k_indices = torch.argsort(iou_scores, descending=True)[:top_k]
            pred_masks = pred_masks[top_k_indices]
            iou_scores = iou_scores[top_k_indices]
            class_labels = class_labels[top_k_indices]

        keep_indices = iou_scores >= self.iou_threshold
        pred_masks = pred_masks[keep_indices]
        iou_scores = iou_scores[keep_indices]
        class_labels = class_labels[keep_indices]

        if pred_masks.shape[0] == 0:
            result = Instances(image_size)
            result.pred_masks = torch.empty(
                (0, image_size[0], image_size[1]), device=self.device
            )
            result.pred_boxes = Boxes(torch.empty((0, 4), device=self.device))
            result.scores = torch.empty((0,), device=self.device)
            result.pred_classes = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            return result

        bit_masks = BitMasks(pred_masks > 0)
        pred_boxes = bit_masks.get_bounding_boxes().to(device=self.device)

        if self.nms_on:
            nms_keep = torchvision.ops.nms(
                pred_boxes.tensor, iou_scores, self.nms_threshold
            )
            pred_masks = pred_masks[nms_keep]
            pred_boxes = pred_boxes[nms_keep]
            iou_scores = iou_scores[nms_keep]
            class_labels = class_labels[nms_keep]

        result = Instances(image_size)
        result.pred_masks = (pred_masks > 0).float()
        result.pred_boxes = pred_boxes
        result.scores = iou_scores
        result.pred_classes = class_labels
        return result

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        return F.interpolate(
            masks.float(), orig_hw, mode="bilinear", align_corners=False
        )

    def semantic_inference(self, mask_cls, mask_pred, keep_sem_bgd: bool = False):
        if keep_sem_bgd:
            mask_cls = F.softmax(mask_cls, dim=-1)
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid().squeeze(1)
        return torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

    def mask_nms(self, masks, scores, iou_threshold: float = 0.5):
        n = masks.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=masks.device)
        if n == 1:
            return torch.ones(1, dtype=torch.bool, device=masks.device)
        binary_masks = masks >= 0.5
        areas = binary_masks.sum(dim=(1, 2))
        order = torch.argsort(scores, descending=True)
        keep = torch.ones(n, dtype=torch.bool, device=masks.device)
        for i in range(n):
            if not keep[order[i]]:
                continue
            mask_i = binary_masks[order[i]]
            area_i = areas[order[i]]
            for j in range(i + 1, n):
                if not keep[order[j]]:
                    continue
                mask_j = binary_masks[order[j]]
                area_j = areas[order[j]]
                intersection = (mask_i & mask_j).sum()
                union = area_i + area_j - intersection
                iou = intersection / union if union > 0 else 0
                if iou > iou_threshold:
                    keep[order[j]] = False
        return keep

    def panoptic_inference(self, mask_cls, mask_pred, class_labels):
        scores = mask_cls.squeeze(1)
        mask_pred = mask_pred.squeeze(1).sigmoid()
        keep = scores > self.iou_threshold
        cur_scores = scores[keep]
        cur_classes = class_labels[keep]
        cur_masks = mask_pred[keep]
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=self.device)
        segments_info = []
        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info

        class_ids = torch.unique(cur_classes)
        nms_keep = torch.zeros_like(cur_scores, dtype=torch.bool)
        for cls_id in class_ids:
            cls_mask = cur_classes == cls_id
            if cls_mask.sum() <= 1:
                nms_keep[cls_mask] = True
                continue
            cls_keep = self.mask_nms(
                cur_masks[cls_mask],
                cur_scores[cls_mask],
                iou_threshold=self.nms_threshold,
            )
            nms_keep[torch.where(cls_mask)[0][cls_keep]] = True

        cur_scores = cur_scores[nms_keep]
        cur_classes = cur_classes[nms_keep]
        cur_masks = cur_masks[nms_keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        thing_map = getattr(self.metadata, "thing_dataset_id_to_contiguous_id", {})
        current_segment_id = 0
        for index in range(cur_classes.shape[0]):
            pred_class = cur_classes[index].item()
            isthing = pred_class in thing_map.values()
            mask_area = (cur_mask_ids == index).sum().item()
            original_area = (cur_masks[index] >= 0.5).sum().item()
            mask = (cur_mask_ids == index) & (cur_masks[index] >= 0.5)
            if mask_area <= 0 or original_area <= 0 or mask.sum().item() <= 0:
                continue
            if mask_area / original_area < 0.5:
                continue
            if not isthing:
                if int(pred_class) in stuff_memory_list:
                    panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                    continue
                stuff_memory_list[int(pred_class)] = current_segment_id + 1
            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": bool(isthing),
                    "category_id": int(pred_class),
                }
            )
        return panoptic_seg, segments_info

    def refer_inference(self, pred_masks, pred_logits, class_labels):
        unique_classes = torch.unique(class_labels)
        num_classes = len(unique_classes)
        h, w = pred_masks.shape[-2:]
        class_masks = torch.zeros((num_classes, h, w), device=self.device)
        class_scores = torch.zeros((num_classes,), device=self.device)
        for index, cls in enumerate(unique_classes):
            cls_indices = class_labels == cls
            if cls_indices.sum() <= 0:
                continue
            cls_masks = pred_masks[cls_indices]
            cls_scores = pred_logits[cls_indices].squeeze(-1)
            best_idx = torch.argmax(cls_scores)
            class_masks[index] = cls_masks[best_idx]
            class_scores[index] = cls_scores[best_idx]
        return class_masks, class_scores


class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.input_projection = None
        self.image_projection = None
        self.output_projection = None

    def forward(
        self, vlm_features: torch.Tensor, image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        assert vlm_features.dim() == 3, (
            f"vlm_features should be 3D, got shape {vlm_features.shape}"
        )
        assert image_embeddings.dim() == 3, (
            f"image_embeddings should be 3D, got shape {image_embeddings.shape}"
        )
        input_dim = vlm_features.size(-1)
        image_dim = image_embeddings.size(-1)

        if input_dim != self.embedding_dim and self.input_projection is None:
            self.input_projection = nn.Linear(input_dim, self.embedding_dim).to(
                vlm_features.device
            )
        if image_dim != self.embedding_dim and self.image_projection is None:
            self.image_projection = nn.Linear(image_dim, self.embedding_dim).to(
                image_embeddings.device
            )
        if self.input_projection is not None:
            vlm_features = self.input_projection(vlm_features)
        if self.image_projection is not None:
            image_embeddings = self.image_projection(image_embeddings)

        output = vlm_features
        for layer in self.layers:
            output = layer(output, image_embeddings)

        if self.input_projection is not None:
            if not hasattr(self, "output_projection") or self.output_projection is None:
                self.output_projection = nn.Linear(self.embedding_dim, input_dim).to(
                    output.device
                )
            output = self.output_projection(output)
        return output


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(embedding_dim)
        self.self_attn = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.cross_attn_norm = nn.LayerNorm(embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, vlm_features: torch.Tensor, image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        residual = vlm_features
        x = self.self_attn_norm(vlm_features)
        x, _ = self.self_attn(x, x, x)
        x = self.self_attn_dropout(x)
        x = residual + x

        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(query=x, key=image_embeddings, value=image_embeddings)
        x = self.cross_attn_dropout(x)
        x = residual + x

        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        return residual + x
