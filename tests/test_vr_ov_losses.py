from __future__ import annotations

import pytest
import torch

from model.vr_ov_losses import DiceLoss, VR_OVLosses
from model.vr_ov_types import CompositionScores


def _make_pred(batch: int = 2, H: int = 8, W: int = 8) -> dict[str, torch.Tensor]:
    """Create synthetic prediction dict with random logit masks."""
    return {"pred_masks": torch.randn(batch, H, W)}


def _make_gt(batch: int = 2, H: int = 8, W: int = 8) -> dict[str, list]:
    """Create synthetic ground-truth dict with binary masks."""
    targets = []
    for _ in range(batch):
        masks = torch.randint(0, 2, (H, W)).float()
        targets.append({"masks": masks})
    return {"targets": targets}


def _make_comp_scores(batch: int = 2, H: int = 8, W: int = 8) -> CompositionScores:
    """Create synthetic CompositionScores with all fields populated."""
    return CompositionScores(
        cat_feat=torch.randn(batch, 1, H, W),
        attr_feat=torch.randn(batch, 1, H, W),
        rel_feat=torch.randn(batch, 1, H, W),
        act_feat=torch.randn(batch, 1, H, W),
    )


class TestVR_OVLosses:
    """Unit tests for the VR_OVLosses multi-task loss module."""

    # ------------------------------------------------------------------ #
    # test_all_lambdas_correct
    # ------------------------------------------------------------------ #
    def test_all_lambdas_correct(self) -> None:
        """Constructor stores lambda values exactly as given."""
        criterion = VR_OVLosses(
            lambda_mask=5.0,
            lambda_attr=1.0,
            lambda_rel=0.5,
            lambda_act=0.5,
            lambda_compose=0.3,
        )
        assert criterion.lambda_mask == 5.0
        assert criterion.lambda_attr == 1.0
        assert criterion.lambda_rel == 0.5
        assert criterion.lambda_act == 0.5
        assert criterion.lambda_compose == 0.3

    # ------------------------------------------------------------------ #
    # test_forward_returns_dict
    # ------------------------------------------------------------------ #
    def test_forward_returns_dict(self) -> None:
        """Forward returns (total, loss_dict) tuple with expected keys."""
        criterion = VR_OVLosses()
        pred = _make_pred()
        gt = _make_gt()
        comp_scores = _make_comp_scores()

        total, loss_dict = criterion(pred, gt, comp_scores)

        assert isinstance(total, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert "loss_mask" in loss_dict
        assert "loss_attr" in loss_dict
        assert "loss_rel" in loss_dict
        assert "loss_act" in loss_dict
        assert "loss_compose" in loss_dict
        assert "loss_total" in loss_dict

    # ------------------------------------------------------------------ #
    # test_forward_no_nan
    # ------------------------------------------------------------------ #
    def test_forward_no_nan(self) -> None:
        """Forward pass produces no NaN in total or any loss term."""
        criterion = VR_OVLosses()
        pred = _make_pred()
        gt = _make_gt()
        comp_scores = _make_comp_scores()

        total, loss_dict = criterion(pred, gt, comp_scores)

        assert not torch.isnan(total)
        assert not torch.isinf(total)
        for key, value in loss_dict.items():
            assert not torch.isnan(value), f"{key} is NaN"
            assert not torch.isinf(value), f"{key} is Inf"

    # ------------------------------------------------------------------ #
    # test_total_is_sum
    # ------------------------------------------------------------------ #
    def test_total_is_sum(self) -> None:
        """Total loss equals sum of weighted individual terms."""
        criterion = VR_OVLosses()
        pred = _make_pred()
        gt = _make_gt()
        comp_scores = _make_comp_scores()

        total, loss_dict = criterion(pred, gt, comp_scores)

        individual_sum = (
            loss_dict["loss_mask"]
            + loss_dict["loss_attr"]
            + loss_dict["loss_rel"]
            + loss_dict["loss_act"]
            + loss_dict["loss_compose"]
        )
        assert torch.allclose(total, individual_sum, atol=1e-6)

    # ------------------------------------------------------------------ #
    # test_no_comp_scores_graceful
    # ------------------------------------------------------------------ #
    def test_no_comp_scores_graceful(self) -> None:
        """comp_scores=None computes only mask loss, no crash."""
        criterion = VR_OVLosses()
        pred = _make_pred()
        gt = _make_gt()

        total, loss_dict = criterion(pred, gt, comp_scores=None)

        assert "loss_mask" in loss_dict
        assert "loss_attr" not in loss_dict
        assert "loss_rel" not in loss_dict
        assert "loss_act" not in loss_dict
        assert "loss_compose" not in loss_dict
        assert "loss_total" in loss_dict
        assert torch.allclose(total, loss_dict["loss_mask"], atol=1e-6)
        assert not torch.isnan(total)

    # ------------------------------------------------------------------ #
    # test_mask_loss_positive
    # ------------------------------------------------------------------ #
    def test_mask_loss_positive(self) -> None:
        """Mask loss > 0 when pred_masks differ from ground-truth."""
        criterion = VR_OVLosses()
        # Create prediction and ground-truth that are opposite.
        pred = {"pred_masks": torch.full((2, 8, 8), 5.0)}  # high logits → sigmoid≈1
        targets = []
        for _ in range(2):
            targets.append({"masks": torch.zeros(8, 8)})  # all zeros
        gt = {"targets": targets}

        total, loss_dict = criterion(pred, gt)

        assert loss_dict["loss_mask"] > 0.0

    # ------------------------------------------------------------------ #
    # test_gradient_flow_mask
    # ------------------------------------------------------------------ #
    def test_gradient_flow_mask(self) -> None:
        """Mask loss backprop flows to pred_masks tensor."""
        criterion = VR_OVLosses()
        pred_masks = torch.randn(2, 8, 8, requires_grad=True)
        pred = {"pred_masks": pred_masks}
        gt = _make_gt()

        total, _ = criterion(pred, gt)
        total.backward()

        assert pred_masks.grad is not None
        assert not torch.isnan(pred_masks.grad).any()


class TestDiceLoss:
    """Unit tests for the DiceLoss module."""

    # ------------------------------------------------------------------ #
    # test_perfect_match
    # ------------------------------------------------------------------ #
    def test_perfect_match(self) -> None:
        """Dice loss ≈ 0 when prediction matches target exactly."""
        dice = DiceLoss(smooth=1.0)
        logits = torch.full((2, 8, 8), 10.0)  # sigmoid(10) ≈ 1
        targets = torch.ones(2, 8, 8)

        loss = dice(logits, targets)
        assert loss < 0.001

    # ------------------------------------------------------------------ #
    # test_complete_mismatch
    # ------------------------------------------------------------------ #
    def test_complete_mismatch(self) -> None:
        """Dice loss ≈ 1 when prediction and target are complements."""
        dice = DiceLoss(smooth=0.0)
        logits = torch.full((2, 8, 8), -10.0)  # sigmoid(-10) ≈ 0
        targets = torch.ones(2, 8, 8)

        loss = dice(logits, targets)
        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-3)

    # ------------------------------------------------------------------ #
    # test_random_range
    # ------------------------------------------------------------------ #
    def test_random_range(self) -> None:
        """Dice loss on random data stays in [0, 1]."""
        dice = DiceLoss(smooth=1.0)
        logits = torch.randn(4, 16, 16)
        targets = torch.randint(0, 2, (4, 16, 16)).float()

        loss = dice(logits, targets)
        assert 0.0 <= loss.item() <= 1.0

    def test_gated_terms_excluded(self) -> None:
        """When lambdas are zero, gated terms do not appear in output dict."""
        criterion = VR_OVLosses(lambda_attr=0.0, lambda_rel=0.0, lambda_act=0.0, lambda_compose=0.0)
        pred = _make_pred()
        gt = _make_gt()
        comp_scores = _make_comp_scores()
        total, loss_dict = criterion(pred, gt, comp_scores)
        assert "loss_mask" in loss_dict
        assert "loss_attr" not in loss_dict
        assert "loss_rel" not in loss_dict
        assert "loss_act" not in loss_dict
        assert "loss_compose" not in loss_dict

    def test_enabled_terms_included(self) -> None:
        """When lambdas are non-zero and features exist, terms appear."""
        criterion = VR_OVLosses(lambda_attr=1.0)
        pred = _make_pred()
        gt = _make_gt()
        comp_scores = _make_comp_scores()
        total, loss_dict = criterion(pred, gt, comp_scores)
        assert "loss_mask" in loss_dict
        assert "loss_attr" in loss_dict
