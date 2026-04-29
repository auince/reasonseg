from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "reasonseg"
        / "benchmarks"
        / "smoke_manifests.py"
    )
    spec = importlib.util.spec_from_file_location(
        "reasonseg_smoke_manifests", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load smoke manifest module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    return _load_smoke_module().main()


if __name__ == "__main__":
    raise SystemExit(main())
