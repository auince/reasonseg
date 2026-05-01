from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SceneGraphVisualEncoder(nn.Module):
    """Scene-graph-aware visual encoder for VR-OV.

    Takes SAM2 multi-scale features and produces:
    - HOI tokens (human-object-interaction query tokens)
    - Top-K region proposals
    - Relation logits per region

    Sub-modules:
        input_proj      — 1×1 conv projections per scale → unified hidden_dim
        hoi_tokens      — learnable HOI query parameters  [num_hoi_tokens, hidden_dim]
        hoi_cross_attn  — single-layer MHA (HOI queries attend to visual keys)
        region_head     — conv head: objectness + 4 bbox offsets
        relation_head   — MLP: [region_feat; hoi_pooled] → num_relations logits
        gate            — learnable scalar gating original vs enhanced HOI
    """

    def __init__(
        self,
        in_channels: list[int] | None = None,
        hidden_dim: int = 256,
        num_hoi_tokens: int = 5,
        num_relations: int = 50,
        num_heads: int = 8,
        region_topk: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_hoi_tokens = num_hoi_tokens
        self.num_relations = num_relations
        self.region_topk = region_topk

        # ── 1. Multi-scale input projections ──
        #    Built lazily in forward() using actual input channel counts.
        #    If *in_channels* is explicitly provided, pre-build to those counts;
        #    forward() will still verify and rebuild if input differs.
        self.input_proj: nn.ModuleDict = nn.ModuleDict()
        self._cfg_in_channels: list[int] | None = in_channels
        if in_channels is not None:
            self._build_input_proj(in_channels)

        # ── 2. HOI tokens (learnable query embeddings) ──
        scale = hidden_dim ** -0.5
        self.hoi_tokens = nn.Parameter(scale * torch.randn(num_hoi_tokens, hidden_dim))

        # ── 3. HOI cross-attention (single-layer with pre-norm + residual) ──
        self.hoi_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.hoi_ln1 = nn.LayerNorm(hidden_dim)
        self.hoi_ln2 = nn.LayerNorm(hidden_dim)
        self.hoi_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # ── 4. Region prediction head ──
        # 5 output channels: objectness + dx, dy, dw, dh
        self.region_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 5, kernel_size=1),
        )

        # ── 5. Relation prediction head ──
        #     Input: [region_feat (256) + hoi_pooled (256)] = 512
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_relations),
        )

        # ── 6. Learnable gate ──
        self.gate = nn.Parameter(torch.tensor([1.0]))

    # ------------------------------------------------------------------ #
    # _build_input_proj
    # ------------------------------------------------------------------ #
    def _build_input_proj(self, channels: list[int]) -> None:
        """Create or recreate 1×1 conv projections for each scale."""
        self.input_proj.clear()
        for i, c in enumerate(channels):
            conv = nn.Conv2d(c, self.hidden_dim, kernel_size=1)
            # Place on same device as existing parameters (if any)
            try:
                device = next(self.parameters()).device
                conv = conv.to(device)
            except StopIteration:
                pass
            self.input_proj[str(i)] = conv

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, features: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            features: dict with
                ``high_res_feats`` — list of ``[B, C_i, H_i, W_i]`` tensors
                ``image_embed``   — ``[B, C, H, W]`` tensor

        Returns:
            hoi_tokens:      ``[B, num_hoi_tokens, hidden_dim]``
            regions:         ``[B, K, hidden_dim]``   (K = min(region_topk, H*W))
            relation_logits: ``[B, K, num_relations]``
        """
        self._validate_features(features)
        batch_size = features["image_embed"].shape[0]

        # ── collect all scales ──
        all_scales: list[Tensor] = list(features["high_res_feats"]) + [
            features["image_embed"]
        ]

        # ── auto-detect and build projections on first pass ──
        actual_channels = [int(feat.shape[1]) for feat in all_scales]
        num_scales = len(actual_channels)
        if num_scales > len(self.input_proj) or any(
            i not in self.input_proj
            or self.input_proj[str(i)].in_channels != actual_channels[i]
            for i in range(num_scales)
        ):
            self._build_input_proj(actual_channels)

        # ── 1. Project each scale to hidden_dim ──
        proj: list[Tensor] = []
        for i, feat in enumerate(all_scales):
            proj.append(self.input_proj[str(i)](feat))
        # proj[i]: [B, hidden_dim, H_i, W_i]

        # ── 2. Flatten spatial dims → single visual sequence ──
        flat: list[Tensor] = []
        for p in proj:
            flat.append(p.flatten(2).transpose(1, 2))  # [B, H_i*W_i, hidden_dim]
        vis_seq = torch.cat(flat, dim=1)  # [B, N_total, hidden_dim]

        # ── 3. HOI tokens cross-attend to visual features ──
        hoi_raw = self.hoi_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        # pre-norm (shared LN for query & key/value — follows SGC-Net pattern)
        attn_out, _ = self.hoi_cross_attn(
            self.hoi_ln1(hoi_raw),
            self.hoi_ln1(vis_seq),
            self.hoi_ln1(vis_seq),
        )
        hoi = hoi_raw + attn_out                   # residual
        hoi = hoi + self.hoi_mlp(self.hoi_ln2(hoi))  # MLP residual

        # ── gate fusion: original vs enhanced ──
        gate_val = torch.sigmoid(self.gate)
        hoi_tokens = gate_val * hoi_raw + (1.0 - gate_val) * hoi

        # ── 4. Region proposals from the deepest scale ──
        feat_last = proj[-1]  # [B, hidden_dim, H, W]
        region_raw = self.region_head(feat_last)  # [B, 5, H, W]

        _b, _c, H, W = region_raw.shape
        K = min(self.region_topk, H * W)

        objectness = region_raw[:, 0:1, :, :]          # [B, 1, H, W]
        _, top_idx = objectness.view(batch_size, -1).topk(K, dim=1)  # [B, K]

        # gather region features
        feat_last_flat = feat_last.flatten(2).transpose(1, 2)        # [B, H*W, hidden_dim]
        idx_feat = top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        top_feat = torch.gather(feat_last_flat, dim=1, index=idx_feat)  # [B, K, hidden_dim]

        # gather bbox offsets (kept internally for downstream loss / decoding)
        bbox_flat = region_raw[:, 1:5].flatten(2).transpose(1, 2)  # [B, H*W, 4]
        idx_bbox = top_idx.unsqueeze(-1).expand(-1, -1, 4)
        top_bbox = torch.gather(bbox_flat, dim=1, index=idx_bbox)   # [B, K, 4]
        _ = top_bbox  # available for downstream use; not returned here

        # ── 5. Relation logits ──
        hoi_pooled = hoi_tokens.mean(dim=1)                        # [B, hidden_dim]
        hoi_exp = hoi_pooled.unsqueeze(1).expand(-1, K, -1)        # [B, K, hidden_dim]
        region_hoi = torch.cat([top_feat, hoi_exp], dim=-1)        # [B, K, 2*hidden_dim]
        relation_logits = self.relation_head(region_hoi)           # [B, K, num_relations]

        # ── 6. Return ──
        regions = top_feat
        return hoi_tokens, regions, relation_logits

    def _validate_features(self, features: dict) -> None:
        if not isinstance(features, dict):
            raise TypeError(
                f"SceneGraphVisualEncoder expects a feature dict, got {type(features).__name__}."
            )
        if "high_res_feats" not in features or "image_embed" not in features:
            raise KeyError(
                "SceneGraphVisualEncoder requires 'high_res_feats' and 'image_embed' entries."
            )
        high_res_feats = features["high_res_feats"]
        image_embed = features["image_embed"]
        if not isinstance(high_res_feats, (list, tuple)) or not high_res_feats:
            raise ValueError(
                "SceneGraphVisualEncoder expects a non-empty high_res_feats sequence."
            )
        if not isinstance(image_embed, torch.Tensor) or image_embed.ndim != 4:
            raise ValueError(
                f"SceneGraphVisualEncoder image_embed must have shape [B, C, H, W]; got {getattr(image_embed, 'shape', None)}."
            )
        batch_size = int(image_embed.shape[0])
        for index, feat in enumerate([*high_res_feats, image_embed]):
            if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
                raise ValueError(
                    f"SceneGraphVisualEncoder feature {index} must have shape [B, C, H, W]; got {getattr(feat, 'shape', None)}."
                )
            if int(feat.shape[0]) != batch_size:
                raise ValueError(
                    f"SceneGraphVisualEncoder feature {index} batch size must be {batch_size}; got {int(feat.shape[0])}."
                )
