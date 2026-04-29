from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_DIR = ROOT / "outputs"
TASK_CONFIGS = {
    "refcoco": ROOT / "configs" / "refcoco" / "refcoco_reasonseg.yaml",
    "refcoco+": ROOT / "configs" / "refcoco" / "refcoco_plus_reasonseg.yaml",
    "refcocog": ROOT / "configs" / "refcoco" / "refcocog_reasonseg.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOv8-style high-level ReasonSeg training launcher."
    )
    parser.add_argument(
        "--task",
        choices=tuple(TASK_CONFIGS.keys()),
        required=True,
        help="Training task alias mapped to the repo-owned config.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument(
        "--name",
        help="Experiment name under --project. Defaults to the task alias.",
    )
    parser.add_argument(
        "--device",
        default="0,1",
        help="Single GPU index or comma-separated GPU indices, e.g. 0 or 0,1.",
    )
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional config overrides passed through as KEY VALUE pairs.",
    )
    return parser


def _parse_device_indices(raw_value: str) -> list[int]:
    indices = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not indices:
        raise ValueError("--device must include at least one GPU index.")
    return indices


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    run_name = args.name or args.task
    return Path(args.project).expanduser().resolve() / run_name


def _resolve_train_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        str(ROOT / "scripts" / "train.py"),
        "--config",
        str(TASK_CONFIGS[args.task].resolve()),
        "--data-root",
        str(Path(args.data_root).expanduser().resolve()),
        "--output-dir",
        str(_resolve_output_dir(args)),
        "--run-index",
        "0",
        "--num-gpus",
        "1",
    ]
    if args.batch is not None:
        argv.extend(["--batch-size", str(args.batch)])
    if args.lr is not None:
        argv.extend(["--lr", str(args.lr)])
    if args.max_iter is not None:
        argv.extend(["--max-iter", str(args.max_iter)])
    if args.checkpoint is not None:
        argv.extend(["--checkpoint", str(Path(args.checkpoint).expanduser().resolve())])
    if args.resume:
        argv.append("--resume")
    if args.opts:
        argv.append("--opts")
        argv.extend(list(args.opts))
    return argv


def build_launch_command(args: argparse.Namespace) -> list[str]:
    if args.resume and args.checkpoint is not None:
        raise ValueError("--resume and --checkpoint cannot be used together.")

    output_dir = _resolve_output_dir(args)
    device_indices = _parse_device_indices(args.device)
    train_argv = _resolve_train_argv(args)

    if len(device_indices) == 1:
        inner_command = [sys.executable, *train_argv]
    else:
        inner_command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(len(device_indices)),
            "--multi_gpu",
            "--tee",
            "3",
            "--log_dir",
            str(output_dir / "accelerate_logs"),
            *train_argv,
        ]

    return [
        sys.executable,
        str(ROOT / "scripts" / "watch_train.py"),
        "--output-dir",
        str(output_dir),
        "--run-index",
        "0",
        "--gpu-indices",
        ",".join(str(index) for index in device_indices),
        "--",
        *inner_command,
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = build_launch_command(args)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
