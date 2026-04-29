from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def ensure_root_model_package_loaded() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "model"
    init_file = package_root / "__init__.py"

    existing = sys.modules.get("model")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if (
            existing_file is None
            or Path(existing_file).resolve() != init_file.resolve()
        ):
            raise RuntimeError(
                "Refusing to load ReasonSeg root-owned backend package because a different "
                f"'model' module is already present: {existing_file!r}."
            )
        return

    spec = importlib.util.spec_from_file_location(
        "model",
        init_file,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load root-owned model package from {package_root}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
