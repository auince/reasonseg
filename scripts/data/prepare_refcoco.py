from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_cli_surface():
    module_path = Path(__file__).resolve().parents[2] / "reasonseg" / "cli_surface.py"
    spec = importlib.util.spec_from_file_location("reasonseg_cli_surface", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI surface from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    return _load_cli_surface().prepare_refcoco_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
