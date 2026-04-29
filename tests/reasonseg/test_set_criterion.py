from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch


class _StaticMatcher:
    def __call__(self, outputs, targets):
        del outputs, targets
        return [(torch.tensor([0]), torch.tensor([0]))]


def test_set_criterion_can_skip_distributed_num_masks_reduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criterion_module = importlib.import_module("reasonseg.modeling.criterion")
    criterion = criterion_module.SetCriterion(
        num_classes=1,
        matcher=_StaticMatcher(),
        weight_dict={"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0},
        eos_coef=0.0,
        losses=["labels"],
    )
    reduce_calls: list[torch.Tensor] = []
    monkeypatch.setattr(
        criterion_module,
        "is_dist_avail_and_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        torch,
        "distributed",
        SimpleNamespace(all_reduce=lambda tensor: reduce_calls.append(tensor.clone())),
    )

    outputs = {
        "pred_logits": torch.zeros((1, 1, 1), dtype=torch.float32),
        "pred_masks": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
    }
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "masks": torch.zeros((1, 2, 2), dtype=torch.float32),
        }
    ]

    loss_dict = criterion(outputs, targets, reduce_num_masks=False)

    assert reduce_calls == []
    assert set(loss_dict) == {"loss_ce"}
