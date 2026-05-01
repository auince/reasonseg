from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def _load_train_vr_ov_module():
    module_path = ROOT / "scripts" / "train_vr_ov.py"
    spec = importlib.util.spec_from_file_location("reasonseg_train_vr_ov", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_parser_includes_vr_ov_training_arguments(tmp_path: Path) -> None:
    train_vr_ov = _load_train_vr_ov_module()

    args = train_vr_ov.build_parser().parse_args(
        [
            "--config",
            str(ROOT / "configs" / "vr_ov" / "vr_ov_base.yaml"),
            "--data-root",
            str(tmp_path / "datasets"),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--phase",
            "1a",
        ]
    )

    assert args.phase == "1a"
    assert args.curriculum_levels == ["L1", "L2", "L3", "L4"]
    assert args.query_dropout_p == 0.2
    assert args.ema_decay == 0.999


def test_main_dispatches_parsed_namespace_with_vr_ov_flag(
    monkeypatch, tmp_path: Path
) -> None:
    train_vr_ov = _load_train_vr_ov_module()
    seen: dict[str, object] = {}

    def _fake_loader(command_name: str):
        assert command_name == "train"

        def _entrypoint(args):
            seen["args"] = args
            return 0

        return _entrypoint

    cli_surface = train_vr_ov._load_cli_surface()
    monkeypatch.setattr(cli_surface, "_load_runtime_entrypoint", _fake_loader)

    result = train_vr_ov.main(
        [
            "--config",
            str(ROOT / "configs" / "vr_ov" / "vr_ov_base.yaml"),
            "--data-root",
            str(tmp_path / "datasets"),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--phase",
            "1b",
        ]
    )

    assert result == 0
    args = seen["args"]
    assert getattr(args, "vr_ov_enabled") is True
    assert getattr(args, "phase") == "1b"
