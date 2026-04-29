# pyright: reportMissingImports=false
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_evf import EvfConfig


_BEIT_PRETRAIN_UNEXPECTED_KEYS = frozenset(
    {
        "mlm_head.weight",
        "mlm_head.bias",
        "mim_head.weight",
        "mim_head.bias",
    }
)


def _load_state_dict_or_raise(
    module: nn.Module,
    state_dict: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor],
    *,
    context: str,
    allowed_unexpected_keys: frozenset[str] = frozenset(),
) -> None:
    module_state = module.state_dict()
    module_keys = set(module_state)
    checkpoint_keys = set(state_dict)

    missing_keys = sorted(module_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - module_keys)
    disallowed_unexpected_keys = [
        key for key in unexpected_keys if key not in allowed_unexpected_keys
    ]

    if missing_keys or disallowed_unexpected_keys:
        problems: list[str] = []
        if missing_keys:
            problems.append(f"missing keys: {missing_keys}")
        if disallowed_unexpected_keys:
            problems.append(f"unexpected keys: {disallowed_unexpected_keys}")
        raise RuntimeError(
            f"{context} checkpoint is incompatible ({'; '.join(problems)})"
        )

    filtered_state_dict = OrderedDict(
        (key, value) for key, value in state_dict.items() if key in module_keys
    )
    try:
        module.load_state_dict(filtered_state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"{context} checkpoint is incompatible ({exc})") from exc


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale: int = 1000,
    eps: float = 1e-6,
) -> torch.Tensor:
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)


def _load_backend_symbols() -> tuple[Any, Any, Any, Any]:
    try:
        from reasonseg.backends.beit3.modeling_utils import (  # type: ignore[import-not-found]
            BEiT3Wrapper,
            _get_base_config,
            _get_large_config,
        )
        from reasonseg.backends.sam2.build_sam import build_sam2  # type: ignore[import-not-found]

        return build_sam2, BEiT3Wrapper, _get_base_config, _get_large_config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The first-party OpenWorldSAM2 wrapper imported successfully, but the SAM2/BEiT3 "
            "backend modules are not yet available under reasonseg.backends. Task 6 keeps this "
            "root-owned boundary importable without reviving vendored-path imports."
        ) from exc


class EvfSam2Model(PreTrainedModel):
    config_class = EvfConfig

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.encoder_pretrained = kwargs.get("encoder_pretrained", None)
        self.train_mask_decoder = kwargs.get("train_mask_decoder", False)
        self.train_prompt_encoder = kwargs.get("train_prompt_encoder", False)
        self.initialize_evf_modules(config)
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

    def initialize_evf_modules(self, config) -> None:
        build_sam2, BEiT3Wrapper, get_base_config, get_large_config = (
            _load_backend_symbols()
        )

        if config.sam_scale == "large":
            self.visual_model = build_sam2(
                "sam2_hiera_l.yaml",
                self.vision_pretrained,
                device=None,
            )
        elif config.sam_scale == "tiny":
            self.visual_model = build_sam2(
                "sam2_hiera_t.yaml",
                self.vision_pretrained,
                device=None,
            )
        else:
            raise NotImplementedError

        for param in self.visual_model.parameters():
            param.requires_grad = False
        if self.train_mask_decoder:
            self.visual_model.sam_mask_decoder.train()
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True
        if self.train_prompt_encoder:
            self.visual_model.sam_prompt_encoder.no_mask_embed.requires_grad_(True)

        if self.config.mm_extractor_scale == "base":
            beit_config = get_base_config()
        elif self.config.mm_extractor_scale == "large":
            beit_config = get_large_config()
        else:
            raise AttributeError(
                "model config should contain key 'mm_extractor_scale', with value 'base' or 'large'."
            )

        self.mm_extractor = BEiT3Wrapper(beit_config)
        if self.encoder_pretrained is not None:
            beit_state_dict = torch.load(self.encoder_pretrained, map_location="cpu")[
                "model"
            ]
            _load_state_dict_or_raise(
                self.mm_extractor,
                beit_state_dict,
                context="BEiT encoder preload",
                allowed_unexpected_keys=_BEIT_PRETRAIN_UNEXPECTED_KEYS,
            )

        for param in self.mm_extractor.parameters():
            param.requires_grad = True

        in_dim = config.hidden_size
        assert in_dim == beit_config.encoder_embed_dim, (
            f"projection layer dim {in_dim} mismatch with mm_extractor dim {beit_config.encoder_embed_dim}"
        )
        out_dim = config.out_dim
        text_fc = [nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

    def print_trainable_parameters(self) -> None:
        total_params = 0
        trainable_params = 0
        print(f"{'Parameter Name':<40}{'Trainable':<10}{'Shape':<20}{'Num Params':<15}")
        print("=" * 85)
        for _, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {total_params - trainable_params}")

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        return F.interpolate(
            masks.float(), orig_hw, mode="bilinear", align_corners=False
        )

    def load_weights(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        _load_state_dict_or_raise(self, state_dict, context="EVF-SAM2 model load")
