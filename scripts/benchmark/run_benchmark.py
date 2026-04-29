from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_reasonseg_package():
    package_dir = Path(__file__).resolve().parents[2] / "reasonseg"
    init_path = package_dir / "__init__.py"
    if "reasonseg" in sys.modules:
        return sys.modules["reasonseg"]
    spec = importlib.util.spec_from_file_location(
        "reasonseg",
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load ReasonSeg package from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_cli_surface():
    _load_reasonseg_package()
    module_path = Path(__file__).resolve().parents[2] / "reasonseg" / "cli_surface.py"
    spec = importlib.util.spec_from_file_location("reasonseg.cli_surface", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI surface from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    return _load_cli_surface().run_benchmark_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
