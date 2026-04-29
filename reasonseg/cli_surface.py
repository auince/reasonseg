from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ReasonSeg from a root-owned paper-ready entrypoint."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from OUTPUT_DIR/run_<index>/last_checkpoint instead of starting from pretrained weights.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional config overrides passed through as KEY VALUE pairs.",
    )
    return parser


def build_eval_parser(command_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run ReasonSeg {command_name} from a root-owned entrypoint."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Explicit checkpoint file to evaluate; eval/test never resume from last_checkpoint.",
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional config overrides passed through as KEY VALUE pairs.",
    )
    return parser


def build_prepare_refcoco_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or verify RefCOCO-family assets from the root command surface."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export benchmark reports from the root-owned ReasonSeg benchmark entrypoint."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-format", choices=("json", "csv"), default="json")
    return parser


def train_main(argv: list[str] | None = None) -> int:
    args = build_train_parser().parse_args(argv)
    return _load_runtime_entrypoint("train")(args)


def eval_main(argv: list[str] | None = None) -> int:
    args = build_eval_parser("eval").parse_args(argv)
    return _load_runtime_entrypoint("eval")(args)


def test_main(argv: list[str] | None = None) -> int:
    args = build_eval_parser("test").parse_args(argv)
    return _load_runtime_entrypoint("test")(args)


def prepare_refcoco_main(argv: list[str] | None = None) -> int:
    args = build_prepare_refcoco_parser().parse_args(argv)
    materializer = _load_refcoco_materializer()
    try:
        result = materializer.prepare_refcoco_data(
            args.data_root,
            materialize=args.materialize,
            verify_only=args.verify_only,
        )
    except materializer.RefCOCODataError as error:
        print(str(error), file=sys.stderr)
        return 1

    materialized_count = len(result["materialized_outputs"])
    verified_count = len(result["verified_outputs"])
    print(
        (
            f"RefCOCO-family data ready under {result['data_root']}: "
            f"materialized={materialized_count}, verified={verified_count}"
        )
    )
    return 0


def run_benchmark_main(argv: list[str] | None = None) -> int:
    args = build_benchmark_parser().parse_args(argv)
    benchmark_runner = _load_benchmark_runner()
    try:
        benchmark_runner.run_benchmark(
            spec_path=args.spec,
            pred_root=args.pred_root,
            output_path=args.output,
            output_format=args.output_format,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


def _load_runtime_entrypoint(command_name: str):
    if __package__:
        if command_name == "train":
            from .runtime.train import main as train_main

            return train_main
        if command_name == "eval":
            from .runtime.eval import main as eval_main

            return eval_main
        if command_name == "test":
            from .runtime.eval import main as test_main

            return test_main
        raise ValueError(f"Unsupported runtime command '{command_name}'")

    _ensure_reasonseg_package_loaded()
    module = importlib.import_module(f"reasonseg.runtime.{command_name}")
    return module.main


def _ensure_reasonseg_package_loaded() -> None:
    if "reasonseg" in sys.modules:
        return

    package_root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "reasonseg",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reasonseg package from {package_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def _load_refcoco_materializer():
    module_path = Path(__file__).resolve().parent / "data" / "refcoco_materializer.py"
    spec = importlib.util.spec_from_file_location(
        "reasonseg_refcoco_materializer", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RefCOCO materializer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark_runner():
    if __package__:
        from . import benchmark_runner

        return benchmark_runner

    module_path = Path(__file__).resolve().parent / "benchmark_runner.py"
    spec = importlib.util.spec_from_file_location(
        "reasonseg_benchmark_runner", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
