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

    @staticmethod
    def _build_query_info(batch_input: dict[str, Any]) -> dict[str, bool]:
        query_struct = batch_input.get("query_struct")
        if query_struct is None:
            return {"has_attrs": True, "has_relations": True, "has_actions": True}
        if isinstance(query_struct, list):
            q = query_struct[0] if query_struct else {}
        else:
            q = query_struct
        return {
            "has_attrs": bool(q.get("attributes")) if isinstance(q, dict) else True,
            "has_relations": bool(q.get("relations")) if isinstance(q, dict) else True,
            "has_actions": bool(q.get("actions")) if isinstance(q, dict) else True,
        }

    @staticmethod
    def _slice_comp_scores_for_refinement(comp_scores: Any) -> Any:
        from model.vr_ov_types import CompositionScores
        fields = {}
        for name in ("cat_feat", "attr_feat", "attr_color", "attr_material",
                     "attr_size", "rel_feat", "act_feat"):
            val = getattr(comp_scores, name)
            if val is not None and val.shape[0] > 1:
                val = val[:1]
            fields[name] = val
        return CompositionScores(**fields)

    @staticmethod
    def _select_coarse_mask(
        low_res_masks: torch.Tensor,
        pred_logits: torch.Tensor,
    ) -> torch.Tensor:
        iou_scores = pred_logits.squeeze(-1)
        best_idx = iou_scores.argmax(dim=0)
        return low_res_masks[best_idx].unsqueeze(0)

    def _accumulate_vr_ov_losses(
        self,
        *,
        pred_masks: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_instances: Any,
        comp_scores: Any,
        all_losses: dict[str, list[torch.Tensor]],
        refined_mask: torch.Tensor | None = None,
        query_info: dict[str, bool] | None = None,
        intermediates: dict[str, torch.Tensor] | None = None,
        visual_feat: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(gt_instances, list):
            gt_instances = [gt_instances]

        pred_masks_list = torch.split(pred_masks, [self.num_tokens] * len(gt_instances))

        prompt_preds = []
        prompt_targets = []
        for prompt_idx, prompt_target in enumerate(gt_instances):
            gt_masks = prompt_target.gt_masks.to(dtype=self.dtype, device=self.device)
            if gt_masks.dim() == 3 and gt_masks.shape[0] == 1:
                gt_masks = gt_masks.squeeze(0)
            prompt_targets.append({"masks": gt_masks})

            if refined_mask is not None and prompt_idx == 0:
                if refined_mask.shape[-2:] != gt_masks.shape[-2:]:
                    ref = refined_mask if refined_mask.ndim == 4 else refined_mask.unsqueeze(0)
                    ref = torch.nn.functional.interpolate(
                        ref, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                    prompt_preds.append(ref.squeeze(0))
                else:
                    prompt_preds.append(refined_mask)
                break

        if not prompt_preds:
            prompt_preds.append(pred_masks_list[0][0:1])

        pred = {"pred_masks": torch.cat(prompt_preds)}
        gt = {"targets": prompt_targets[: len(prompt_preds)]}
        _, loss_dict = self.vr_ov_losses(
            pred, gt,
            self._slice_comp_scores_for_refinement(comp_scores),
            query_info=query_info,
            intermediates=intermediates,
            visual_feat=visual_feat,
        )
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

            comp_scores_result = self._run_vr_ov_comp_matcher(
                query_graphs=None if query_graphs is None else query_graphs[img_idx],
                image_embed=features["image_embed"][img_idx],
                return_intermediates=True,
            )
            intermediates = None
            if isinstance(comp_scores_result, tuple):
                comp_scores, intermediates = comp_scores_result
            else:
                comp_scores = comp_scores_result
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

            refined_mask = None
            if self.training and self.vr_ov_refine_decoder is not None and comp_scores is not None:
                visual_feat = features["image_embed"][img_idx].unsqueeze(0)
                single_comp_scores = self._slice_comp_scores_for_refinement(comp_scores)
                image_embed_1 = features["image_embed"][img_idx].unsqueeze(0)
                image_pe = self.visual_model.sam_prompt_encoder.get_dense_pe()
                refined_mask, _ = self.vr_ov_refine_decoder(
                    coarse_mask=None,
                    comp_scores=single_comp_scores,
                    visual_feat=visual_feat,
                    mask_decoder=self.visual_model.sam_mask_decoder,
                    image_embed=image_embed_1,
                    image_pe=image_pe,
                    high_res_features=high_res_features,
                    prompt_encoder=self.visual_model.sam_prompt_encoder,
                    positional_tokens=self.positional_tokens,
                )

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
                query_info = self._build_query_info(batched_inputs[img_idx])
                visual_feat_for_loss = features["image_embed"][img_idx].unsqueeze(0)
                self._accumulate_vr_ov_losses(
                    pred_masks=pred_masks,
                    pred_logits=pred_logits,
                    gt_instances=batched_inputs[img_idx]["instances"],
                    comp_scores=comp_scores,
                    all_losses=all_losses,
                    refined_mask=refined_mask,
                    query_info=query_info,
                    intermediates=intermediates,
                    visual_feat=visual_feat_for_loss,
                )

        if self.training:
            return {
                key: torch.stack(values).mean() for key, values in all_losses.items()
            }
        return processed_results
