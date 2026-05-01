from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable, cast

# Ensure repo root is on sys.path so model.* imports work
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_cli_surface() -> ModuleType:
    _ensure_reasonseg_package_loaded()
    return importlib.import_module("reasonseg.cli_surface")


def _ensure_reasonseg_package_loaded() -> None:
    package_root = Path(__file__).resolve().parents[1] / "reasonseg"
    init_file = package_root / "__init__.py"

    existing = sys.modules.get("reasonseg")
    if existing is not None:
        existing_file = cast(str | None, getattr(existing, "__file__", None))
        if (
            existing_file is None
            or Path(existing_file).resolve() != init_file.resolve()
        ):
            raise RuntimeError(
                "Refusing to load ReasonSeg package because a different 'reasonseg' module "
                + f"is already present: {existing_file!r}."
            )
        return

    spec = importlib.util.spec_from_file_location(
        "reasonseg",
        init_file,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load ReasonSeg package from {package_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def main(argv: list[str] | None = None) -> int:
    train_main = cast(Callable[[list[str] | None], int], _load_cli_surface().train_main)
    return train_main(argv)


_ensure_reasonseg_package_loaded()


if __name__ == "__main__":
    raise SystemExit(main())
