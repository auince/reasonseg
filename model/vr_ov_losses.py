# pyright: reportMissingImports=false
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .vr_ov_types import CompositionScores


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.sigmoid()
        inputs_flat = inputs.flatten(1)
        targets_flat = targets.flatten(1)
        numerator = 2.0 * (inputs_flat * targets_flat).sum(-1)
        denominator = inputs_flat.sum(-1) + targets_flat.sum(-1)
        loss = 1.0 - (numerator + self.smooth) / (denominator + self.smooth)
        return loss.mean()


class VR_OVLosses(nn.Module):
    def __init__(
        self,
        lambda_mask: float = 5.0,
        lambda_attr: float = 1.0,
        lambda_rel: float = 0.5,
        lambda_act: float = 0.5,
        lambda_compose: float = 0.3,
    ) -> None:
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_attr = lambda_attr
        self.lambda_rel = lambda_rel
        self.lambda_act = lambda_act
        self.lambda_compose = lambda_compose

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.dice = DiceLoss()
        self.margin_rank = nn.MarginRankingLoss(margin=0.1)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        gt: dict[str, list],
        comp_scores: CompositionScores | None = None,
        query_info: dict[str, bool] | None = None,
        intermediates: dict[str, torch.Tensor] | None = None,
        visual_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}

        losses["loss_mask"] = self._compute_mask_loss(pred, gt) * self.lambda_mask

        if self.lambda_attr > 0 and comp_scores is not None and comp_scores.attr_feat is not None:
            losses["loss_attr"] = (
                self._compute_attr_loss(comp_scores, gt, query_info) * self.lambda_attr
            )

        if self.lambda_rel > 0 and comp_scores is not None and comp_scores.rel_feat is not None:
            losses["loss_rel"] = (
                self._compute_rel_loss(comp_scores, gt, query_info) * self.lambda_rel
            )

        if self.lambda_act > 0 and comp_scores is not None and comp_scores.act_feat is not None:
            losses["loss_act"] = (
                self._compute_act_loss(comp_scores, gt, query_info) * self.lambda_act
            )

        if self.lambda_compose > 0 and comp_scores is not None:
            losses["loss_compose"] = (
                self._compute_compose_loss(comp_scores, gt, visual_feat) * self.lambda_compose
            )

        total = sum(losses.values())
        losses["loss_total"] = total
        return total, losses

    # ------------------------------------------------------------------
    # mask loss (unchanged)
    # ------------------------------------------------------------------
    def _compute_mask_loss(
        self,
        pred: dict[str, torch.Tensor],
        gt: dict[str, list],
    ) -> torch.Tensor:
        pred_masks = pred["pred_masks"]
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)

        targets_list = [t["masks"] for t in gt["targets"]]
        target_masks = torch.stack(targets_list).to(pred_masks.device, dtype=torch.float32)

        if pred_masks.shape[-2:] != target_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks[:, None],
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        bce_loss = self.bce(pred_masks, target_masks)
        dice_loss = self.dice(pred_masks, target_masks)
        return bce_loss + dice_loss

    # ------------------------------------------------------------------
    # attr loss — BIOtagger-aware + per-sub-field
    # ------------------------------------------------------------------
    def _compute_attr_loss(
        self,
        comp_scores: CompositionScores,
        gt: dict[str, list],
        query_info: dict[str, bool] | None,
    ) -> torch.Tensor:
        has_attrs = query_info.get("has_attrs", True) if query_info else True

        if has_attrs:
            loss = self._mask_region_contrast_loss(comp_scores.attr_feat, gt)  # type: ignore[arg-type]
            if comp_scores.attr_color is not None:
                loss = loss + self._mask_region_contrast_loss(comp_scores.attr_color, gt)  # type: ignore[arg-type]
            if comp_scores.attr_material is not None:
                loss = loss + self._mask_region_contrast_loss(comp_scores.attr_material, gt)  # type: ignore[arg-type]
            if comp_scores.attr_size is not None:
                loss = loss + self._mask_region_contrast_loss(comp_scores.attr_size, gt)  # type: ignore[arg-type]
            return loss
        else:
            score = comp_scores.attr_feat
            return score.abs().mean() * 0.01  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # rel loss — BIOtagger-aware
    # ------------------------------------------------------------------
    def _compute_rel_loss(
        self,
        comp_scores: CompositionScores,
        gt: dict[str, list],
        query_info: dict[str, bool] | None,
    ) -> torch.Tensor:
        has_rels = query_info.get("has_relations", True) if query_info else True

        if has_rels:
            return self._mask_region_contrast_loss(comp_scores.rel_feat, gt)  # type: ignore[arg-type]
        else:
            score = comp_scores.rel_feat
            return score.abs().mean() * 0.01  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # act loss — BIOtagger-aware
    # ------------------------------------------------------------------
    def _compute_act_loss(
        self,
        comp_scores: CompositionScores,
        gt: dict[str, list],
        query_info: dict[str, bool] | None,
    ) -> torch.Tensor:
        has_acts = query_info.get("has_actions", True) if query_info else True

        if has_acts:
            return self._mask_region_contrast_loss(comp_scores.act_feat, gt)  # type: ignore[arg-type]
        else:
            score = comp_scores.act_feat
            return score.abs().mean() * 0.01  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # compose loss — triplet ranking
    # ------------------------------------------------------------------
    def _compute_compose_loss(
        self,
        comp_scores: CompositionScores,
        gt: dict[str, list],
        visual_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        if visual_feat is None:
            return self._mask_region_contrast_loss(comp_scores.cat_feat, gt)  # type: ignore[arg-type]

        targets_list = [t["masks"] for t in gt["targets"]]
        target_tensor = torch.stack(targets_list).to(visual_feat.device, dtype=torch.bool)

        if visual_feat.shape[-2:] != target_tensor.shape[-2:]:
            visual_feat = F.interpolate(
                visual_feat, size=target_tensor.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        B, C, H, W = visual_feat.shape
        pos_feats = []
        neg_feats = []
        for b in range(B):
            mask = target_tensor[b]
            feat = visual_feat[b]
            pos_region = feat[:, mask].mean(dim=-1) if mask.any() else feat.mean(dim=[-2, -1])
            neg_region = feat[:, ~mask].mean(dim=-1) if (~mask).any() else torch.zeros(C, device=feat.device)
            pos_feats.append(pos_region)
            neg_feats.append(neg_region)

        anchor = torch.stack(pos_feats)
        positive = anchor
        negative = torch.stack(neg_feats)

        dist_pos = (anchor - positive).pow(2).sum(-1)
        dist_neg = (anchor - negative).pow(2).sum(-1)
        target = torch.ones(B, device=visual_feat.device)
        return self.margin_rank(dist_neg, dist_pos, target)

    # ------------------------------------------------------------------
    # shared helper
    # ------------------------------------------------------------------
    def _mask_region_contrast_loss(
        self,
        score_map: torch.Tensor,
        gt: dict[str, list],
    ) -> torch.Tensor:
        targets_list = [t["masks"] for t in gt["targets"]]
        target_tensor = torch.stack(targets_list).to(
            score_map.device, dtype=torch.float32
        )
        if score_map.shape[-2:] != target_tensor.shape[-2:]:
            score_map = F.interpolate(
                score_map,
                size=target_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if score_map.ndim == 4 and score_map.shape[1] == 1:
            score_map = score_map.squeeze(1)
        if target_tensor.ndim == 4 and target_tensor.shape[1] == 1:
            target_tensor = target_tensor.squeeze(1)

        score_map = score_map.clamp(-10.0, 10.0)
        return self.bce(score_map, target_tensor)
