from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_yolo_train_module():
    module_path = ROOT / "scripts" / "yolo_train.py"
    spec = importlib.util.spec_from_file_location("reasonseg_yolo_train", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_launch_command_uses_direct_train_for_single_gpu(tmp_path: Path) -> None:
    yolo_train = _load_yolo_train_module()

    args = yolo_train.build_parser().parse_args(
        [
            "--task",
            "refcoco",
            "--data-root",
            str(tmp_path / "datasets"),
            "--project",
            str(tmp_path / "outputs"),
            "--name",
            "exp1",
            "--device",
            "0",
            "--batch",
            "2",
            "--lr",
            "0.0002",
            "--max-iter",
            "100",
        ]
    )

    command = yolo_train.build_launch_command(args)

    assert command[:2] == [sys.executable, str(ROOT / "scripts" / "watch_train.py")]
    assert (
        "--gpu-indices" in command
        and command[command.index("--gpu-indices") + 1] == "0"
    )
    assert str(ROOT / "scripts" / "train.py") in command
    assert "accelerate.commands.launch" not in command
    assert str(tmp_path / "outputs" / "exp1") in command
    assert (
        "--batch-size" in command and command[command.index("--batch-size") + 1] == "2"
    )
    assert "--max-iter" in command and command[command.index("--max-iter") + 1] == "100"


def test_build_launch_command_uses_accelerate_for_multi_gpu(tmp_path: Path) -> None:
    yolo_train = _load_yolo_train_module()

    args = yolo_train.build_parser().parse_args(
        [
            "--task",
            "refcocog",
            "--data-root",
            str(tmp_path / "datasets"),
            "--project",
            str(tmp_path / "outputs"),
            "--device",
            "0,1",
            "--resume",
            "--opts",
            "TEST.EVAL_PERIOD",
            "0",
        ]
    )

    command = yolo_train.build_launch_command(args)

    assert "accelerate.commands.launch" in command
    assert "--num_processes" in command
    assert command[command.index("--num_processes") + 1] == "2"
    assert str(ROOT / "scripts" / "train.py") in command
    assert "--resume" in command
    assert "--opts" in command
    assert str(tmp_path / "outputs" / "refcocog" / "accelerate_logs") in command


def test_build_launch_command_rejects_resume_and_checkpoint(tmp_path: Path) -> None:
    yolo_train = _load_yolo_train_module()
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_text("checkpoint\n", encoding="utf-8")

    args = yolo_train.build_parser().parse_args(
        [
            "--task",
            "refcoco+",
            "--data-root",
            str(tmp_path / "datasets"),
            "--device",
            "0,1",
            "--resume",
            "--checkpoint",
            str(checkpoint_path),
        ]
    )

    with pytest.raises(ValueError, match="cannot be used together"):
        yolo_train.build_launch_command(args)


def test_main_runs_built_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yolo_train = _load_yolo_train_module()
    seen: dict[str, object] = {}

    class _CompletedProcess:
        returncode = 0

    def _fake_run(command, check=False):
        seen["command"] = command
        return _CompletedProcess()

    monkeypatch.setattr(
        yolo_train.subprocess,
        "run",
        _fake_run,
    )

    result = yolo_train.main(
        [
            "--task",
            "refcoco",
            "--data-root",
            str(tmp_path / "datasets"),
            "--device",
            "0",
        ]
    )

    assert result == 0
    command = seen["command"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert str(ROOT / "scripts" / "watch_train.py") in command
