from __future__ import annotations

import pytest
import torch

from model.compositional_matcher import CompositionalFeatureMatcher
from model.vr_ov_types import CompositionScores


def _make_inputs(
    batch_size: int = 2, hidden_dim: int = 256, H: int = 8, W: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create standardised (query_nodes, visual_features, img_feat)."""
    N = H * W
    query_nodes = torch.randn(batch_size, 4, hidden_dim)
    visual_features = torch.randn(batch_size, N, hidden_dim)
    img_feat = torch.randn(batch_size, hidden_dim, H, W)
    return query_nodes, visual_features, img_feat


def _make_model(**kwargs: int | float) -> CompositionalFeatureMatcher:
    return CompositionalFeatureMatcher(**kwargs)  # type: ignore[arg-type]


class TestCompositionalFeatureMatcher:
    """Unit tests for the CompositionalFeatureMatcher."""

    # ------------------------------------------------------------------ #
    # test_bcm_shape
    # ------------------------------------------------------------------ #
    def test_bcm_shape(self) -> None:
        """BCM output has correct shape [B, 1, H, W]."""
        model = _make_model()
        model.eval()

        qn, vf, imf = _make_inputs(batch_size=2, H=8, W=8)
        with torch.no_grad():
            scores, _ = model(qn, vf, imf)

        assert scores.cat_feat is not None
        assert scores.cat_feat.shape == (2, 1, 8, 8)
        assert not torch.isnan(scores.cat_feat).any()
        # sigmoid outputs should be in [0, 1]
        assert (scores.cat_feat >= 0).all()
        assert (scores.cat_feat <= 1).all()

    # ------------------------------------------------------------------ #
    # test_attm_shape
    # ------------------------------------------------------------------ #
    def test_attm_shape(self) -> None:
        """ATTM three-way projection produces correct shape, no NaN."""
        model = _make_model()
        model.eval()

        qn, vf, imf = _make_inputs(batch_size=2, H=16, W=16)
        with torch.no_grad():
            scores, _ = model(qn, vf, imf)

        assert scores.attr_feat is not None
        assert scores.attr_feat.shape == (2, 1, 16, 16)
        assert not torch.isnan(scores.attr_feat).any()

    # ------------------------------------------------------------------ #
    # test_full_forward
    # ------------------------------------------------------------------ #
    def test_full_forward(self) -> None:
        """Full forward pass with all five paths does not crash."""
        model = _make_model()
        model.eval()

        qn, vf, imf = _make_inputs(batch_size=2, H=12, W=12)
        with torch.no_grad():
            scores, cmf = model(qn, vf, imf)

        # All score heads must be populated
        assert scores.cat_feat is not None
        assert scores.attr_feat is not None
        assert scores.rel_feat is not None
        assert scores.act_feat is not None
        # CMF feature map shape
        assert cmf.shape == (2, 256, 12, 12)
        # No NaN anywhere
        for t in [scores.cat_feat, scores.attr_feat, scores.rel_feat, scores.act_feat, cmf]:
            assert t is not None
            assert not torch.isnan(t).any()

    # ------------------------------------------------------------------ #
    # test_compositionscores_output
    # ------------------------------------------------------------------ #
    def test_compositionscores_output(self) -> None:
        """Return value is a CompositionScores dataclass instance."""
        model = _make_model()
        model.eval()

        qn, vf, imf = _make_inputs(batch_size=1, H=8, W=8)
        with torch.no_grad():
            scores, cmf = model(qn, vf, imf)

        assert isinstance(scores, CompositionScores)
        assert isinstance(cmf, torch.Tensor)

    # ------------------------------------------------------------------ #
    # test_cmf_fusion_shape
    # ------------------------------------------------------------------ #
    def test_cmf_fusion_shape(self) -> None:
        """CMF fusion output is a 2-D feature map [B, C, H, W]."""
        model = _make_model()
        model.eval()

        qn, vf, imf = _make_inputs(batch_size=2, H=16, W=32)
        with torch.no_grad():
            _, cmf = model(qn, vf, imf)

        assert cmf.shape == (2, 256, 16, 32)
        # CMF feature map is NOT constrained to [0, 1] — it is a raw feature map
        assert cmf.dtype == torch.float32

    # ------------------------------------------------------------------ #
    # test_single_batch
    # ------------------------------------------------------------------ #
    def test_single_batch(self) -> None:
        """Batch size 1 works across all paths."""
        model = _make_model()
        model.eval()

        qn, vf, imf = _make_inputs(batch_size=1, H=4, W=4)
        with torch.no_grad():
            scores, cmf = model(qn, vf, imf)

        assert scores.cat_feat.shape == (1, 1, 4, 4)
        assert scores.attr_feat.shape == (1, 1, 4, 4)
        assert scores.rel_feat.shape == (1, 1, 4, 4)
        assert scores.act_feat.shape == (1, 1, 4, 4)
        assert cmf.shape == (1, 256, 4, 4)
        assert not torch.isnan(cmf).any()

    # ------------------------------------------------------------------ #
    # test_deterministic
    # ------------------------------------------------------------------ #
    def test_deterministic(self) -> None:
        """Same input + eval mode → identical output."""
        model = _make_model()
        model.eval()

        torch.manual_seed(42)
        qn1, vf1, imf1 = _make_inputs(batch_size=2, H=8, W=8)
        with torch.no_grad():
            scores1, cmf1 = model(qn1, vf1, imf1)

        torch.manual_seed(42)
        qn2, vf2, imf2 = _make_inputs(batch_size=2, H=8, W=8)
        with torch.no_grad():
            scores2, cmf2 = model(qn2, vf2, imf2)

        assert torch.allclose(cmf1, cmf2, atol=1e-6)
        for a, b in [
            (scores1.cat_feat, scores2.cat_feat),
            (scores1.attr_feat, scores2.attr_feat),
            (scores1.rel_feat, scores2.rel_feat),
            (scores1.act_feat, scores2.act_feat),
        ]:
            assert a is not None and b is not None
            assert torch.allclose(a, b, atol=1e-6)

    # ------------------------------------------------------------------ #
    # test_different_spatial_sizes
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize("H,W", [(4, 4), (8, 8), (16, 16), (16, 32)])
    def test_different_spatial_sizes(self, H: int, W: int) -> None:
        """Various spatial resolutions produce correct shapes."""
        model = _make_model()
        model.eval()

        qn, vf, imf = _make_inputs(batch_size=2, H=H, W=W)
        with torch.no_grad():
            scores, cmf = model(qn, vf, imf)

        for name in ["cat_feat", "attr_feat", "rel_feat", "act_feat"]:
            t = getattr(scores, name)
            assert t.shape == (2, 1, H, W), f"{name}: {t.shape} != (2, 1, {H}, {W})"
        assert cmf.shape == (2, 256, H, W)

    # ------------------------------------------------------------------ #
    # test_gradient_flow
    # ------------------------------------------------------------------ #
    def test_gradient_flow(self) -> None:
        """Train mode backprop flows through all parameters."""
        model = _make_model()
        model.train()

        qn, vf, imf = _make_inputs(batch_size=2, H=8, W=8)
        scores, cmf = model(qn, vf, imf)

        loss = (
            scores.cat_feat.sum()
            + scores.attr_feat.sum()
            + scores.rel_feat.sum()
            + scores.act_feat.sum()
            + cmf.sum()
        )
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} grad is NaN"

    def test_fails_loudly_on_visual_token_shape_mismatch(self) -> None:
        model = _make_model()
        qn, vf, imf = _make_inputs(batch_size=2, H=8, W=8)

        with pytest.raises(ValueError, match=r"H\*W"):
            model(qn, vf[:, :-1, :], imf)
