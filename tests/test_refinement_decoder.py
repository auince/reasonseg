from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.refinement_decoder import IterativeRefinementDecoder
from model.vr_ov_types import CompositionScores, RefineState


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_scores(
    B: int = 2,
    H: int = 64,
    W: int = 64,
    *,
    include_attr: bool = True,
    include_rel: bool = True,
    include_act: bool = True,
) -> CompositionScores:
    return CompositionScores(
        cat_feat=torch.rand(B, 1, H, W),
        attr_feat=torch.rand(B, 1, H, W) if include_attr else None,
        rel_feat=torch.rand(B, 1, H, W) if include_rel else None,
        act_feat=torch.rand(B, 1, H, W) if include_act else None,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestInitParams:
    def test_attr_threshold_value(self) -> None:
        decoder = IterativeRefinementDecoder()
        raw = decoder.attr_threshold
        assert raw.item() == pytest.approx(0.5)
        assert torch.sigmoid(raw).item() == pytest.approx(0.622459, abs=1e-4)

    def test_score_weights_initial_value(self) -> None:
        decoder = IterativeRefinementDecoder()
        expected = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert torch.allclose(decoder.score_weights, expected, atol=1e-6)

    def test_max_iter_positive(self) -> None:
        with pytest.raises(ValueError, match="max_iter"):
            IterativeRefinementDecoder(max_iter=0)


class TestForward:
    def test_forward_3stages_does_not_crash(self) -> None:
        B, C, H, W = 2, 256, 64, 64
        decoder = IterativeRefinementDecoder(hidden_dim=C, max_iter=3)

        coarse = torch.rand(B, 1, H, W)
        scores = _make_scores(B, H, W)
        visual = torch.rand(B, C, H, W)

        final_mask, history = decoder(coarse, scores, visual)

        assert final_mask.shape == (B, 1, H, W)
        assert len(history) == 3
        assert all(isinstance(s, RefineState) for s in history)

    def test_early_stopping_above_threshold(self) -> None:
        """
        Force very high IoU after stage 1 by replacing the refinement
        conv with a near-identity that outputs ~1.0 everywhere so the
        mask barely changes and the early-stopping gate fires.
        """
        B, C, H, W = 2, 256, 64, 64
        decoder = IterativeRefinementDecoder(hidden_dim=C, max_iter=3)

        with torch.no_grad():
            nn.init.constant_(decoder.mask_refine_conv[2].weight, 1.0)
            nn.init.constant_(decoder.mask_refine_conv[2].bias, 10.0)

        coarse = torch.ones(B, 1, H, W)
        scores = CompositionScores(
            cat_feat=torch.ones(B, 1, H, W),
            attr_feat=torch.ones(B, 1, H, W),
            rel_feat=torch.zeros(B, 1, H, W),
            act_feat=torch.zeros(B, 1, H, W),
        )
        visual = torch.rand(B, C, H, W)

        _, history = decoder(coarse, scores, visual)
        assert len(history) == 2
        assert history[-1].converged

    def test_mask_range_zero_to_one(self) -> None:
        B, C, H, W = 2, 256, 64, 64
        decoder = IterativeRefinementDecoder(hidden_dim=C, max_iter=3)

        coarse = torch.rand(B, 1, H, W)
        scores = _make_scores(B, H, W)
        visual = torch.rand(B, C, H, W)

        final_mask, _ = decoder(coarse, scores, visual)
        assert final_mask.min() >= 0.0
        assert final_mask.max() <= 1.0

    def test_refine_state_history_structure(self) -> None:
        B, C, H, W = 2, 256, 64, 64
        decoder = IterativeRefinementDecoder(hidden_dim=C, max_iter=3)

        coarse = torch.rand(B, 1, H, W)
        scores = _make_scores(B, H, W)
        visual = torch.rand(B, C, H, W)

        _, history = decoder(coarse, scores, visual)

        assert len(history) == 3
        for s in history:
            assert s.mask.shape == (B, 1, H, W)
            assert 0.0 <= s.iou <= 1.0
            assert s.stage in {0, 1, 2}
            assert isinstance(s.converged, bool)

    def test_gradient_flows_to_params(self) -> None:
        B, C, H, W = 2, 256, 64, 64
        decoder = IterativeRefinementDecoder(hidden_dim=C, max_iter=3)

        coarse = torch.rand(B, 1, H, W)
        scores = _make_scores(B, H, W)
        visual = torch.rand(B, C, H, W)

        final_mask, _ = decoder(coarse, scores, visual)
        loss = final_mask.sum()
        loss.backward()

        # attr_threshold used via hard-threshold gate; no grad expected.
        assert decoder.attr_threshold.grad is None
        assert decoder.score_weights.grad is not None
        for name, p in decoder.mask_refine_conv.named_parameters():
            assert p.grad is not None, f"{name} has no grad"

    def test_missing_attr_score_fails_loudly(self) -> None:
        B, C, H, W = 2, 256, 64, 64
        decoder = IterativeRefinementDecoder(hidden_dim=C, max_iter=3)

        coarse = torch.rand(B, 1, H, W)
        scores = _make_scores(B, H, W, include_attr=False, include_rel=False, include_act=False)
        visual = torch.rand(B, C, H, W)

        with pytest.raises(ValueError, match="requires 'attr_feat'"):
            decoder(coarse, scores, visual)

    def test_mismatched_score_shape_fails_loudly(self) -> None:
        B, C, H, W = 2, 256, 64, 64
        decoder = IterativeRefinementDecoder(hidden_dim=C, max_iter=3)

        coarse = torch.rand(B, 1, H, W)
        scores = _make_scores(B, H, W)
        scores.rel_feat = torch.rand(B, 1, H // 2, W)
        visual = torch.rand(B, C, H, W)

        with pytest.raises(ValueError, match="expects rel_feat"):
            decoder(coarse, scores, visual)

    def test_no_nan_in_output(self) -> None:
        B, C, H, W = 2, 256, 64, 64
        decoder = IterativeRefinementDecoder(hidden_dim=C, max_iter=3)

        coarse = torch.rand(B, 1, H, W)
        scores = _make_scores(B, H, W)
        visual = torch.rand(B, C, H, W)

        final_mask, _ = decoder(coarse, scores, visual)
        assert not torch.isnan(final_mask).any()
        assert not torch.isinf(final_mask).any()
