# pyright: reportMissingImports=false
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .vr_ov_types import CompositionScores


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation masks.

    Computes the Sørensen-Dice coefficient loss between sigmoid-activated
    logits and binary targets.  Smoothing factor prevents division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            inputs:  Sigmoid logits, shape ``(B, *)``.
            targets: Binary targets same shape as *inputs*.
        Returns:
            Scalar loss averaged over the batch.
        """
        inputs = inputs.sigmoid()
        inputs_flat = inputs.flatten(1)
        targets_flat = targets.flatten(1)
        numerator = 2.0 * (inputs_flat * targets_flat).sum(-1)
        denominator = inputs_flat.sum(-1) + targets_flat.sum(-1)
        loss = 1.0 - (numerator + self.smooth) / (denominator + self.smooth)
        return loss.mean()


class VR_OVLosses(nn.Module):
    """Multi-task joint loss for VR-OV referring-expression segmentation.

    Total loss::

        L_total = λ_mask * L_mask + λ_attr * L_attr + λ_rel * L_rel
                + λ_act * L_act + λ_compose * L_compose

    where:

    - ``L_mask``:    BCE + Dice on predicted binary masks.
    - ``L_attr``:    Attribute-matching loss（placeholder).
    - ``L_rel``:     Relation-matching loss（placeholder).
    - ``L_act``:     Action-matching loss（placeholder).
    - ``L_compose``: Composition ranking loss（placeholder).

    Default weights:  λ_mask=5.0, λ_attr=1.0, λ_rel=0.5, λ_act=0.5,
    λ_compose=0.3.
    """

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

        # ------------------------------------------------------------------
        # Loss modules
        # ------------------------------------------------------------------
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(reduction="mean")
        self.margin_rank = nn.MarginRankingLoss(margin=0.1)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(
        self,
        pred: dict[str, torch.Tensor],
        gt: dict[str, list],
        comp_scores: CompositionScores | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute all applicable loss terms.

        Args:
            pred:           Must contain ``"pred_masks"`` (``[B, H, W]``
                            or ``[B, 1, H, W]``).  ``"pred_logits"`` is
                            optional.
            gt:             Must contain ``"targets"``: a list of
                            per-image ground-truth dicts, each holding a
                            ``"masks"`` tensor ``[H, W]``.
            comp_scores:    Optional ``CompositionScores`` with per-modality
                            feature maps.

        Returns:
            ``(total_loss, loss_dict)`` where *total_loss* is a scalar
            tensor and *loss_dict* contains the weighted individual terms.
        """
        losses: dict[str, torch.Tensor] = {}

        # ── L_mask：BCE + Dice ────────────────────────────────────────
        losses["loss_mask"] = self._compute_mask_loss(pred, gt) * self.lambda_mask

        # ── L_attr：attribute matching ────────────────────────────────
        if self.lambda_attr > 0 and comp_scores is not None and comp_scores.attr_feat is not None:
            losses["loss_attr"] = (
                self._compute_attr_loss(comp_scores, gt) * self.lambda_attr
            )

        # ── L_rel：relation matching ──────────────────────────────────
        if self.lambda_rel > 0 and comp_scores is not None and comp_scores.rel_feat is not None:
            losses["loss_rel"] = (
                self._compute_rel_loss(comp_scores, gt) * self.lambda_rel
            )

        # ── L_act：action matching ────────────────────────────────────
        if self.lambda_act > 0 and comp_scores is not None and comp_scores.act_feat is not None:
            losses["loss_act"] = (
                self._compute_act_loss(comp_scores, gt) * self.lambda_act
            )

        # ── L_compose：composition ranking ────────────────────────────
        if self.lambda_compose > 0 and comp_scores is not None:
            losses["loss_compose"] = (
                self._compute_compose_loss(comp_scores, gt) * self.lambda_compose
            )

        total = sum(losses.values())
        losses["loss_total"] = total
        return total, losses

    # ------------------------------------------------------------------
    # Private helpers — mask loss
    # ------------------------------------------------------------------
    def _compute_mask_loss(
        self,
        pred: dict[str, torch.Tensor],
        gt: dict[str, list],
    ) -> torch.Tensor:
        """BCE + Dice loss on predicted binary masks.

        Matches ``pred["pred_masks"]`` against per-image ground-truth
        masks from ``gt["targets"][i]["masks"]``.
        """
        pred_masks = pred["pred_masks"]
        # Squeeze a leading singleton channel dimension if present.
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)

        targets_list = [t["masks"] for t in gt["targets"]]
        target_masks = torch.stack(targets_list).to(pred_masks.device, dtype=torch.float32)

        # Interpolate predictions to match target spatial size if needed.
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
    # Private helpers — composition losses (placeholders)
    # ------------------------------------------------------------------
    def _compute_attr_loss(
        self,
        comp_scores: CompositionScores,
        gt: dict[str, list],  # noqa: ARG002
    ) -> torch.Tensor:
        """Attribute-matching loss（placeholder).

        Returns zero while the ground-truth format for attribute
        existence is being finalised.
        """
        return comp_scores.attr_feat.mean() * 0.0  # type: ignore[union-attr]

    def _compute_rel_loss(
        self,
        comp_scores: CompositionScores,
        gt: dict[str, list],  # noqa: ARG002
    ) -> torch.Tensor:
        """Relation-matching loss（placeholder)."""
        return comp_scores.rel_feat.mean() * 0.0  # type: ignore[union-attr]

    def _compute_act_loss(
        self,
        comp_scores: CompositionScores,
        gt: dict[str, list],  # noqa: ARG002
    ) -> torch.Tensor:
        """Action-matching loss（placeholder)."""
        return comp_scores.act_feat.mean() * 0.0  # type: ignore[union-attr]

    def _compute_compose_loss(
        self,
        comp_scores: CompositionScores,
        gt: dict[str, list],  # noqa: ARG002
    ) -> torch.Tensor:
        """Composition-ranking loss（placeholder)."""
        return comp_scores.cat_feat.mean() * 0.0  # type: ignore[union-attr]
