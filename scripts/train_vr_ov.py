from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable, cast


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
        if existing_file is None or Path(existing_file).resolve() != init_file.resolve():
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


def build_parser():
    cli_surface = _load_cli_surface()
    parser = cli_surface.build_train_parser()
    parser.description = "Train VR-OV with curriculum scheduling and phased freezing."
    parser.add_argument(
        "--phase",
        choices=("1a", "1b", "1c", "2", "3"),
        help="Optional VR-OV training phase. 1a/1b/1c enable deterministic module freezing.",
    )
    parser.add_argument(
        "--curriculum-levels",
        nargs="+",
        default=["L1", "L2", "L3", "L4"],
        help="Curriculum levels in progression order. Defaults to L1 L2 L3 L4.",
    )
    parser.add_argument(
        "--curriculum-switch-interval",
        type=int,
        default=0,
        help="Iterations per curriculum stage. Defaults to max_iter / num_levels when unset.",
    )
    parser.add_argument(
        "--query-dropout-p",
        type=float,
        default=0.2,
        help="Probability of zeroing each query-graph node during VR-OV training.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay used for additive VR-OV checkpoint tracking.",
    )
    parser.add_argument(
        "--vr-ov-seed",
        type=int,
        default=0,
        help="Deterministic seed for curriculum/dropout runtime helpers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.vr_ov_enabled = True
    train_main = cast(Callable[[object], int], _load_cli_surface()._load_runtime_entrypoint("train"))
    return train_main(args)


_ensure_reasonseg_package_loaded()


if __name__ == "__main__":
    raise SystemExit(main())
