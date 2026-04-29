# pyright: reportMissingImports=false, reportUnknownVariableType=false
from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from reasonseg.modeling.evf_sam2 import (
    _BEIT_PRETRAIN_UNEXPECTED_KEYS,
    _load_state_dict_or_raise,
)


def test_checkpoint_load_accepts_only_documented_beit_pretrain_extra_heads() -> None:
    module = torch.nn.Linear(2, 1)
    checkpoint = OrderedDict(
        {
            "weight": torch.ones_like(module.weight),
            "bias": torch.zeros_like(module.bias),
            "mlm_head.weight": torch.ones(1),
            "mlm_head.bias": torch.ones(1),
            "mim_head.weight": torch.ones(1),
            "mim_head.bias": torch.ones(1),
        }
    )

    _load_state_dict_or_raise(
        module,
        checkpoint,
        context="BEiT encoder preload",
        allowed_unexpected_keys=_BEIT_PRETRAIN_UNEXPECTED_KEYS,
    )

    assert torch.equal(module.weight, checkpoint["weight"])
    assert torch.equal(module.bias, checkpoint["bias"])


def test_checkpoint_load_rejects_missing_model_keys() -> None:
    module = torch.nn.Linear(2, 1)
    checkpoint = OrderedDict({"weight": torch.ones_like(module.weight)})

    with pytest.raises(RuntimeError, match="missing keys"):
        _load_state_dict_or_raise(
            module,
            checkpoint,
            context="EVF-SAM2 model load",
        )


def test_checkpoint_load_rejects_undocumented_unexpected_keys() -> None:
    module = torch.nn.Linear(2, 1)
    checkpoint = OrderedDict(
        {
            "weight": torch.ones_like(module.weight),
            "bias": torch.zeros_like(module.bias),
            "classifier.weight": torch.ones(1),
        }
    )

    with pytest.raises(RuntimeError, match="unexpected keys"):
        _load_state_dict_or_raise(
            module,
            checkpoint,
            context="BEiT encoder preload",
            allowed_unexpected_keys=_BEIT_PRETRAIN_UNEXPECTED_KEYS,
        )
