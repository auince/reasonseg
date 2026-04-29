from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_cli_surface():
    module_path = Path(__file__).resolve().parents[1] / "reasonseg" / "cli_surface.py"
    spec = importlib.util.spec_from_file_location("reasonseg_cli_surface", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI surface from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    return _load_cli_surface().test_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
