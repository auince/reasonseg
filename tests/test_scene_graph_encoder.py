from __future__ import annotations

import pytest
import torch

from model.scene_graph_encoder import SceneGraphVisualEncoder


def _fake_features(batch_size: int = 2) -> dict:
    """Minimal SAM2-style multi-scale feature dict used by all tests."""
    return {
        "high_res_feats": [
            torch.randn(batch_size, 256, 32, 32),
            torch.randn(batch_size, 128, 16, 16),
        ],
        "image_embed": torch.randn(batch_size, 64, 8, 8),
    }


class TestSceneGraphVisualEncoder:
    """Unit tests for SceneGraphVisualEncoder."""

    # ------------------------------------------------------------------ #
    # test_forward_shapes
    # ------------------------------------------------------------------ #
    def test_forward_shapes(self) -> None:
        """Normal forward pass — output shapes match spec."""
        model = SceneGraphVisualEncoder()
        model.eval()

        feats = _fake_features(batch_size=2)
        with torch.no_grad():
            hoi, regions, relations = model(feats)

        assert hoi.shape == (2, 5, 256), f"hoi: {hoi.shape}"
        assert regions.shape == (2, 64, 256), f"regions: {regions.shape}"
        assert relations.shape == (2, 64, 50), f"relations: {relations.shape}"

    # ------------------------------------------------------------------ #
    # test_single_batch
    # ------------------------------------------------------------------ #
    def test_single_batch(self) -> None:
        """Batch size 1 — output shapes correct and no NaN."""
        model = SceneGraphVisualEncoder()
        model.eval()

        feats = _fake_features(batch_size=1)
        with torch.no_grad():
            hoi, regions, relations = model(feats)

        assert hoi.shape == (1, 5, 256)
        assert regions.shape == (1, 64, 256)
        assert relations.shape == (1, 64, 50)
        assert not torch.isnan(hoi).any()
        assert not torch.isnan(regions).any()
        assert not torch.isnan(relations).any()

    # ------------------------------------------------------------------ #
    # test_regions_topk
    # ------------------------------------------------------------------ #
    def test_regions_topk(self) -> None:
        """region_count ≤ region_topk even with small feature maps."""
        model = SceneGraphVisualEncoder(region_topk=64)
        model.eval()

        # Small spatial dim so H*W < 64
        feats = {
            "high_res_feats": [
                torch.randn(2, 256, 4, 4),
                torch.randn(2, 128, 2, 2),
            ],
            "image_embed": torch.randn(2, 64, 2, 2),
        }
        with torch.no_grad():
            _, regions, relations = model(feats)

        K = regions.shape[1]
        assert K <= 64, f"K={K}, expected ≤ 64"
        assert relations.shape[1] == K

    # ------------------------------------------------------------------ #
    # test_regions_topk_configurable
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize("region_topk", [16, 32, 64, 100])
    def test_regions_topk_configurable(self, region_topk: int) -> None:
        """Custom region_topk produces the requested K."""
        model = SceneGraphVisualEncoder(region_topk=region_topk)
        model.eval()

        feats = _fake_features(batch_size=2)
        with torch.no_grad():
            _, regions, relations = model(feats)

        H, W = 8, 8  # from image_embed in _fake_features
        expected_K = min(region_topk, H * W)
        assert regions.shape[1] == expected_K
        assert relations.shape[1] == expected_K

    # ------------------------------------------------------------------ #
    # test_deterministic
    # ------------------------------------------------------------------ #
    def test_deterministic(self) -> None:
        """Same input + eval mode → identical output."""
        model = SceneGraphVisualEncoder()
        model.eval()

        torch.manual_seed(42)
        feats1 = _fake_features(batch_size=2)
        with torch.no_grad():
            hoi1, reg1, rel1 = model(feats1)

        torch.manual_seed(42)
        feats2 = _fake_features(batch_size=2)
        with torch.no_grad():
            hoi2, reg2, rel2 = model(feats2)

        assert torch.allclose(hoi1, hoi2, atol=1e-6)
        assert torch.allclose(reg1, reg2, atol=1e-6)
        assert torch.allclose(rel1, rel2, atol=1e-6)

    # ------------------------------------------------------------------ #
    # test_gate_range
    # ------------------------------------------------------------------ #
    def test_gate_range(self) -> None:
        """Gate value after sigmoid is in [0, 1]."""
        model = SceneGraphVisualEncoder()
        gate_val = model.gate.sigmoid().item()
        assert 0.0 <= gate_val <= 1.0, f"gate={gate_val}"

        # Also check that gate is learnable
        assert model.gate.requires_grad

    # ------------------------------------------------------------------ #
    # test_custom_in_channels
    # ------------------------------------------------------------------ #
    def test_custom_in_channels(self) -> None:
        """Explicit in_channels override accepted."""
        model = SceneGraphVisualEncoder(in_channels=[128, 64, 32], hidden_dim=128)
        model.eval()

        feats = {
            "high_res_feats": [
                torch.randn(1, 128, 16, 16),
                torch.randn(1, 64, 8, 8),
            ],
            "image_embed": torch.randn(1, 32, 4, 4),
        }
        with torch.no_grad():
            hoi, regions, relations = model(feats)

        assert hoi.shape == (1, 5, 128)
        assert regions.shape[2] == 128

    # ------------------------------------------------------------------ #
    # test_gradient_flow
    # ------------------------------------------------------------------ #
    def test_gradient_flow(self) -> None:
        """Train mode backprop flows through all parameters."""
        model = SceneGraphVisualEncoder()
        model.train()

        feats = _fake_features(batch_size=2)
        feats = {
            "high_res_feats": [f.detach().requires_grad_(True) for f in feats["high_res_feats"]],
            "image_embed": feats["image_embed"].detach().requires_grad_(True),
        }

        hoi, regions, relations = model(feats)

        # top-K selection (gather) is non-differentiable — region_head
        # params only receive gradients through direct region_raw loss.
        # Simulate two loss paths: (a) the returned tensors, (b) region_raw
        # accessed via a helper to create a differentiable path to region_head.
        feat_last = model.input_proj["2"](feats["image_embed"])
        region_raw = model.region_head(feat_last)

        loss = hoi.sum() + regions.sum() + relations.sum() + region_raw.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} grad is NaN"
