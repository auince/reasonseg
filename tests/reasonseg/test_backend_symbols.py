# pyright: reportMissingImports=false
from __future__ import annotations

from pathlib import Path

from reasonseg.modeling.evf_sam2 import _load_backend_symbols


def test_task6_backend_symbols_are_loadable() -> None:
    build_sam2, beit3_wrapper, get_base_config, get_large_config = (
        _load_backend_symbols()
    )

    assert callable(build_sam2)
    assert beit3_wrapper.__name__ == "BEiT3Wrapper"
    assert callable(get_base_config)
    assert callable(get_large_config)


def test_backend_symbols_resolve_to_root_owned_modules() -> None:
    build_sam2, beit3_wrapper, _, _ = _load_backend_symbols()

    build_sam2_path = Path(build_sam2.__code__.co_filename).resolve()

    assert "resources/code" not in str(build_sam2_path)
    assert build_sam2_path.as_posix().endswith(
        "/model/segment_anything_2/sam2/build_sam.py"
    )
    assert beit3_wrapper.__module__ == "model.unilm.beit3.modeling_utils"
