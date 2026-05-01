from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.vr_ov_types import CompositionScores
from reasonseg.modeling.open_world_sam2 import CrossAttentionLayer


class CompositionalFeatureMatcher(nn.Module):
    """Five-path compositional feature matching.

    Paths
    -----
    * **BCM**  — Basic Category Matching (dot-product similarity).
    * **ATTM** — Attribute Matching (3-way projection + weighted average).
    * **RSM**  — Relation Spatial Matching (MLP verification).
    * **ACMM** — Action Matching (cross-attention).
    * **CMF**  — Cross-Modal Fusion (3-layer CrossAttention).

    Parameters
    ----------
    hidden_dim : int
        Feature dimension for all projections (default 256).
    num_heads : int
        Number of attention heads (default 8).
    cmf_layers : int
        Number of CrossAttention layers in the CMF path (default 3).
    dropout : float
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        cmf_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # ------------------------------------------------------------------ #
        # ATTM — three projection layers (colour / material / size)
        # ------------------------------------------------------------------ #
        self.color_proj = nn.Linear(hidden_dim, hidden_dim)
        self.material_proj = nn.Linear(hidden_dim, hidden_dim)
        self.size_proj = nn.Linear(hidden_dim, hidden_dim)

        # ------------------------------------------------------------------ #
        # RSM — MLP for relation-spatial verification
        # ------------------------------------------------------------------ #
        self.rsm_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # ------------------------------------------------------------------ #
        # ACMM — cross-attention based action matching
        # ------------------------------------------------------------------ #
        self.acmm_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )

        # ------------------------------------------------------------------ #
        # CMF — 3-layer cross-attention fusion
        # ------------------------------------------------------------------ #
        self.cmf_layers = nn.ModuleList([
            CrossAttentionLayer(
                embedding_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=hidden_dim * 4,
                dropout=dropout,
            )
            for _ in range(cmf_layers)
        ])

    # ---------------------------------------------------------------------- #
    # forward
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        query_nodes: Tensor,
        visual_features: Tensor,
        img_feat: Tensor,
    ) -> tuple[CompositionScores, Tensor]:
        """Run all five matching paths.

        Parameters
        ----------
        query_nodes : Tensor
            ``[B, 4, hidden_dim]`` — four GNN-enhanced query nodes
            (0=category, 1=attribute, 2=relation, 3=action).

        visual_features : Tensor
            ``[B, N, hidden_dim]`` — flattened image patch features.

        img_feat : Tensor
            ``[B, hidden_dim, H, W]`` — 2-D feature map (same semantics
            as ``visual_features`` reshaped to spatial format).

        Returns
        -------
        scores : CompositionScores
            Four score maps ``[B, 1, H, W]`` (one per modality).
        cmf_feat : Tensor
            Cross-modal fused feature map ``[B, hidden_dim, H, W]``.
        """
        self._validate_inputs(query_nodes, visual_features, img_feat)
        B, _C, H, W = img_feat.shape
        N = visual_features.shape[1]

        # ── BCM — dot-product category matching ──
        cat_node = query_nodes[:, 0:1, :]                     # [B, 1, C]
        bcm_score = torch.matmul(
            cat_node, visual_features.transpose(-2, -1)
        )                                                      # [B, 1, N]
        bcm_score = bcm_score.sigmoid()
        bcm_2d = bcm_score.view(B, 1, H, W)                   # [B, 1, H, W]

        # ── ATTM — 3-way projected attribute matching ──
        attr_node = query_nodes[:, 1:2, :]                     # [B, 1, C]
        attr_color = self.color_proj(attr_node)                # [B, 1, C]
        attr_material = self.material_proj(attr_node)          # [B, 1, C]
        attr_size = self.size_proj(attr_node)                  # [B, 1, C]
        attr_combined = (attr_color + attr_material + attr_size) / 3.0
        attm_score = torch.matmul(
            attr_combined, visual_features.transpose(-2, -1)
        ).sigmoid()                                             # [B, 1, N]
        attm_2d = attm_score.view(B, 1, H, W)                  # [B, 1, H, W]

        # ── RSM — relation-spatial MLP verification ──
        rel_node = query_nodes[:, 2:3, :].expand(-1, N, -1)   # [B, N, C]
        rsm_input = torch.cat([rel_node, visual_features], dim=-1)  # [B, N, 2C]
        rsm_score = self.rsm_mlp(rsm_input).squeeze(-1).sigmoid()    # [B, N]
        rsm_2d = rsm_score.view(B, 1, H, W)                    # [B, 1, H, W]

        # ── ACMM — cross-attention action matching ──
        act_node = query_nodes[:, 3:4, :]                      # [B, 1, C]
        acmm_out, _ = self.acmm_attn(
            act_node, visual_features, visual_features,
        )                                                       # [B, 1, C]
        acmm_score = torch.matmul(
            acmm_out, visual_features.transpose(-2, -1)
        ).sigmoid()                                             # [B, 1, N]
        acmm_2d = acmm_score.view(B, 1, H, W)                  # [B, 1, H, W]

        # ── CMF — cross-modal fusion (visual features attend to query nodes) ──
        x = visual_features                                     # [B, N, C]
        for layer in self.cmf_layers:
            x = layer(x, query_nodes)                           # [B, N, C]
        cmf_feat = x.transpose(1, 2).view(B, self.hidden_dim, H, W)  # [B, C, H, W]

        # ── Pack scores ──
        scores = CompositionScores(
            cat_feat=bcm_2d,
            attr_feat=attm_2d,
            rel_feat=rsm_2d,
            act_feat=acmm_2d,
        )

        return scores, cmf_feat

    def _validate_inputs(
        self,
        query_nodes: Tensor,
        visual_features: Tensor,
        img_feat: Tensor,
    ) -> None:
        if query_nodes.ndim != 3:
            raise ValueError(
                f"CompositionalFeatureMatcher expects query_nodes with shape [B, 4, C]; got {tuple(query_nodes.shape)}."
            )
        if visual_features.ndim != 3:
            raise ValueError(
                f"CompositionalFeatureMatcher expects visual_features with shape [B, N, C]; got {tuple(visual_features.shape)}."
            )
        if img_feat.ndim != 4:
            raise ValueError(
                f"CompositionalFeatureMatcher expects img_feat with shape [B, C, H, W]; got {tuple(img_feat.shape)}."
            )
        if query_nodes.shape[1] != 4:
            raise ValueError(
                f"CompositionalFeatureMatcher expects 4 semantic query nodes; got {int(query_nodes.shape[1])}."
            )
        if query_nodes.shape[0] != visual_features.shape[0] or query_nodes.shape[0] != img_feat.shape[0]:
            raise ValueError(
                "CompositionalFeatureMatcher batch dimensions must match across query_nodes, visual_features, and img_feat."
            )
        if query_nodes.shape[2] != self.hidden_dim:
            raise ValueError(
                f"CompositionalFeatureMatcher query node dim mismatch: expected {self.hidden_dim}, got {int(query_nodes.shape[2])}."
            )
        if visual_features.shape[2] != self.hidden_dim:
            raise ValueError(
                f"CompositionalFeatureMatcher visual feature dim mismatch: expected {self.hidden_dim}, got {int(visual_features.shape[2])}."
            )
        if img_feat.shape[1] != self.hidden_dim:
            raise ValueError(
                f"CompositionalFeatureMatcher image feature dim mismatch: expected {self.hidden_dim}, got {int(img_feat.shape[1])}."
            )
        expected_tokens = int(img_feat.shape[2] * img_feat.shape[3])
        if int(visual_features.shape[1]) != expected_tokens:
            raise ValueError(
                f"CompositionalFeatureMatcher visual token count must equal H*W ({expected_tokens}); got {int(visual_features.shape[1])}."
            )
