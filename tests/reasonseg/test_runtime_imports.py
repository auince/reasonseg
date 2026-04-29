# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
from __future__ import annotations

import reasonseg


def test_task6_root_runtime_import_surface_is_available() -> None:
    from reasonseg.modeling.open_world_sam2 import OpenWorldSAM2
    from reasonseg.runtime.train import main as train_main

    assert reasonseg.parse_query("red dog")["target"] == "dog"
    assert OpenWorldSAM2.__name__ == "OpenWorldSAM2"
    assert callable(train_main)
