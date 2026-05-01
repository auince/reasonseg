# pyright: reportMissingImports=false
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from model.vr_ov_types import CompositionScores, RefineState


class IterativeRefinementDecoder(nn.Module):
    """Three-stage iterative refinement decoder for referring expression segmentation.

    Stage 1 / Stage 0 (Coarse Localization):
        Receives CMF-fused features and a coarse mask.  Optionally invokes a
        provided SAM2 ``mask_decoder`` to produce the initial mask M_0.

    Stage 2 (Attribute Verification) – iteration 0 of the refinement loop:
        Extracts features inside the current mask region, compares them against
        the attribute-matching score (``attr_feat``) from ``CompositionScores``,
        and applies a learnable threshold gate::

            M_1 = M_0 * (attr_score > threshold)

    Stage 3 (Relational Refinement) – iteration 2 of the refinement loop:
        Incorporates relation and action scores via learned score weights::

            refinement = w_rel * rel_score + w_act * act_score
            M_final  = clamp(M * (1 + 0.1 * refinement), 0, 1)

    The decoder runs for at most ``max_iter`` iterations and stops early
    when the per-iteration mask IoU exceeds 0.95.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the visual feature channels (default 256).
    max_iter : int
        Maximum number of refinement iterations (default 3).
    dropout : float
        Dropout rate (reserved for future use).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        max_iter: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ---- learnable parameters ------------------------------------------------
        # attr_threshold: after sigmoid ≈ 0.6225 at initialisation
        self.attr_threshold: nn.Parameter = nn.Parameter(
            torch.tensor([0.5])
        )
        # score_weights: [cat, attr, rel, act] — all 0.25 initially
        self.score_weights: nn.Parameter = nn.Parameter(
            torch.tensor([0.25, 0.25, 0.25, 0.25])
        )

        # ---- hyper-parameters ----------------------------------------------------
        self.max_iter: int = max_iter
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        # ---- refinement modules ---------------------------------------------------
        self.mask_refine_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        coarse_mask: torch.Tensor,
        comp_scores: CompositionScores,
        visual_feat: torch.Tensor,
        mask_decoder: Optional[object] = None,
    ) -> tuple[torch.Tensor, List[RefineState]]:
        """Run the iterative refinement pipeline.

        Parameters
        ----------
        coarse_mask : Tensor [B, 1, H, W]
            Initial coarse mask (probabilities in [0, 1]).
        comp_scores : CompositionScores
            Parsed composition-matching scores (cat, attr, rel, act).
        visual_feat : Tensor [B, C, H, W]
            Visual feature map (C == ``hidden_dim``).
        mask_decoder : object, optional
            An optional SAM2-style mask decoder.  Currently unused in the
            default three-stage loop, but reserved for Stage‑1 integration.

        Returns
        -------
        final_mask : Tensor [B, 1, H, W]
        refine_history : List[RefineState]
            One entry per executed iteration.
        """
        self._validate_inputs(
            coarse_mask=coarse_mask,
            comp_scores=comp_scores,
            visual_feat=visual_feat,
        )
        history: List[RefineState] = []
        mask: torch.Tensor = coarse_mask

        for stage in range(self.max_iter):
            prev_mask = mask.clone()

            # ------ Stage 0: attribute verification -------------------------------
            if stage == 0:
                attr_score: Optional[torch.Tensor] = comp_scores.attr_feat
                if attr_score is not None:
                    attr_thresh = self.attr_threshold.sigmoid()
                    attr_gate = (attr_score > attr_thresh).float()
                    mask = mask * attr_gate

            # ------ Stage 1: visual-feature refinement ----------------------------
            elif stage == 1:
                # Concatenate visual features with the current mask and refine.
                mask_feat = torch.cat([visual_feat, mask], dim=1)
                mask = self.mask_refine_conv(mask_feat).sigmoid()

            # ------ Stage 2: relational refinement --------------------------------
            elif stage == 2:
                rel_score = comp_scores.rel_feat
                act_score = comp_scores.act_feat

                rel_weight = self.score_weights[2].sigmoid()
                act_weight = self.score_weights[3].sigmoid()

                refinement: torch.Tensor = torch.zeros_like(mask)
                if rel_score is not None:
                    refinement = refinement + rel_weight * rel_score
                if act_score is not None:
                    refinement = refinement + act_weight * act_score

                mask = mask * (1.0 + 0.1 * refinement)
                mask = torch.clamp(mask, 0.0, 1.0)

            # ------ IoU convergence check ------------------------------------------
            inter = (mask * prev_mask).sum()
            union = (mask + prev_mask - mask * prev_mask).sum().clamp(min=1e-6)
            iou_val = (inter / union).item()

            converged = iou_val > 0.95

            history.append(
                RefineState(
                    mask=mask.clone().detach(),
                    iou=iou_val,
                    stage=stage,
                    converged=converged,
                )
            )

            if converged and stage > 0:
                break

        return mask, history

    def _validate_inputs(
        self,
        *,
        coarse_mask: torch.Tensor,
        comp_scores: CompositionScores,
        visual_feat: torch.Tensor,
    ) -> None:
        if coarse_mask.ndim != 4 or coarse_mask.shape[1] != 1:
            raise ValueError(
                f"IterativeRefinementDecoder expects coarse_mask with shape [B, 1, H, W]; got {tuple(coarse_mask.shape)}."
            )
        expected_channels = self.mask_refine_conv[0].in_channels - 1
        if visual_feat.ndim != 4:
            raise ValueError(
                f"IterativeRefinementDecoder expects visual_feat with shape [B, C, H, W]; got {tuple(visual_feat.shape)}."
            )
        if tuple(visual_feat.shape[:1] + visual_feat.shape[2:]) != tuple(
            coarse_mask.shape[:1] + coarse_mask.shape[2:]
        ):
            raise ValueError(
                "IterativeRefinementDecoder visual_feat batch/spatial dimensions must match coarse_mask."
            )
        if int(visual_feat.shape[1]) != expected_channels:
            raise ValueError(
                f"IterativeRefinementDecoder visual feature dim mismatch: expected {expected_channels}, got {int(visual_feat.shape[1])}."
            )

        for field_name in ("cat_feat", "attr_feat", "rel_feat", "act_feat"):
            score = getattr(comp_scores, field_name)
            if score is None:
                raise ValueError(
                    f"IterativeRefinementDecoder requires '{field_name}' in CompositionScores for canonical VR-OV refinement."
                )
            if score.ndim != 4 or tuple(score.shape) != tuple(coarse_mask.shape):
                raise ValueError(
                    f"IterativeRefinementDecoder expects {field_name} with shape {tuple(coarse_mask.shape)}; got {tuple(score.shape)}."
                )
