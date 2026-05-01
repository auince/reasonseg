# pyright: reportMissingImports=false
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from reasonseg.modeling._compat import META_ARCH_REGISTRY
from reasonseg.modeling.open_world_sam2 import (
    OpenWorldSAM2,
    _build_open_world_sam2_common_kwargs,
    _build_open_world_sam2_parser_kwargs,
    _build_vr_ov_module_kwargs,
)


@META_ARCH_REGISTRY.register()
class VR_OV(OpenWorldSAM2):
    """VR-OV main model with dedicated build and forward ownership."""

    @classmethod
    def from_config(cls, cfg):
        common_kwargs, evf_hidden_size = _build_open_world_sam2_common_kwargs(cfg)
        common_kwargs.update(_build_open_world_sam2_parser_kwargs(cfg))
        common_kwargs.update(
            _build_vr_ov_module_kwargs(cfg, evf_hidden_size=evf_hidden_size)
        )
        return common_kwargs

    def _accumulate_vr_ov_losses(
        self,
        *,
        pred_masks: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_instances: Any,
        comp_scores: Any,
        all_losses: dict[str, list[torch.Tensor]],
    ) -> None:
        if not isinstance(gt_instances, list):
            gt_instances = [gt_instances]

        pred_masks_list = torch.split(pred_masks, [self.num_tokens] * len(gt_instances))
        pred_logits_list = torch.split(pred_logits, [self.num_tokens] * len(gt_instances))

        prompt_preds = []
        prompt_targets = []
        for prompt_idx, prompt_target in enumerate(gt_instances):
            prompt_pred_mask = pred_masks_list[prompt_idx][0:1]
            prompt_preds.append(prompt_pred_mask)
            gt_masks = prompt_target.gt_masks.to(dtype=self.dtype, device=self.device)
            if gt_masks.dim() == 3 and gt_masks.shape[0] == 1:
                gt_masks = gt_masks.squeeze(0)
            prompt_targets.append({"masks": gt_masks})

        pred = {"pred_masks": torch.cat(prompt_preds), "pred_logits": pred_logits}
        gt = {"targets": prompt_targets}
        _, loss_dict = self.vr_ov_losses(pred, gt, comp_scores)
        gating = self.vr_ov_loss_config
        key_enabled_map = {
            "loss_mask": gating.get("mask_enabled", True),
            "loss_attr": gating.get("attr_enabled", False),
            "loss_rel": gating.get("rel_enabled", False),
            "loss_act": gating.get("act_enabled", False),
            "loss_compose": gating.get("compose_enabled", False),
        }
        for key, value in loss_dict.items():
            if key == "loss_total":
                continue
            if key_enabled_map.get(key, False):
                all_losses[f"vr_ov_{key}"].append(value)

    def forward(self, batched_inputs, return_intermediate: bool = False):
        self._assert_forward_backend_available()
        images, images_evf, original_size_list = self._prepare_input_tensors(
            batched_inputs
        )
        batch_size = len(batched_inputs)
        backbone_out, features = self._encode_backbone_features(images, batch_size)
        vr_ov_sg_features = self._run_vr_ov_scene_graph(backbone_out)
        feat, offset, input_ids, attention_masks, output = self._encode_text_prompts(
            batched_inputs,
            images_evf,
        )
        query_graphs = self._run_vr_ov_query_parser(
            batched_inputs=batched_inputs,
            offset=offset,
            output=output,
            input_ids=input_ids,
            attention_masks=attention_masks,
        )
        self._update_learned_parser_logits(output)

        all_losses: dict[str, list[torch.Tensor]] = defaultdict(list)
        processed_results: list[dict[str, Any]] = []

        for img_idx in range(batch_size):
            batch_feat_with_tokens = self._build_prompt_tokens(feat[img_idx])
            if vr_ov_sg_features is not None:
                sg_hoi_tokens, _, _ = vr_ov_sg_features
                batch_feat_with_tokens = self._apply_vr_ov_scene_graph_prompt_context(
                    batch_feat_with_tokens,
                    sg_hoi_tokens[img_idx],
                )

            comp_scores = self._run_vr_ov_comp_matcher(
                query_graphs=None if query_graphs is None else query_graphs[img_idx],
                image_embed=features["image_embed"][img_idx],
            )
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
                        comp_scores=comp_scores,
                        use_refine_decoder=True,
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

            if self.vr_ov_losses is not None and self.vr_ov_loss_config is not None:
                self._accumulate_vr_ov_losses(
                    pred_masks=pred_masks,
                    pred_logits=pred_logits,
                    gt_instances=batched_inputs[img_idx]["instances"],
                    comp_scores=comp_scores,
                    all_losses=all_losses,
                )

        if self.training:
            return {
                key: torch.stack(values).mean() for key, values in all_losses.items()
            }
        return processed_results
