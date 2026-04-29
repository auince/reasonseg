# pyright: reportMissingImports=false
from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from reasonseg.data.registry import RefExpRegistryError, get_registered_refexp_dataset
from reasonseg.runtime import common as runtime_common


ROOT = Path(__file__).resolve().parents[2]
SmokeCase = tuple[
    str, Callable[[list[str] | None], int], Path, str | None, Path, list[str]
]


def _load_cli_surface_module():
    module_path = ROOT / "reasonseg" / "cli_surface.py"
    spec = importlib.util.spec_from_file_location("reasonseg_cli_surface", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_script_module(relative_path: str, module_name: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_materialized_dataset_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "datasets"
    annotations_root = data_root / "coco" / "annotations"
    annotations_root.mkdir(parents=True)
    (data_root / "coco" / "train2014").mkdir(parents=True)

    for file_name in (
        "refcoco_unc_train.json",
        "refcoco_unc_val.json",
        "refcoco_unc_testA.json",
        "refcoco_unc_testB.json",
        "refcoco_plus_unc_train.json",
        "refcoco_plus_unc_val.json",
        "refcoco_plus_unc_testA.json",
        "refcoco_plus_unc_testB.json",
        "refcocog_umd_train.json",
        "refcocog_umd_val.json",
        "refcocog_umd_test.json",
    ):
        (annotations_root / file_name).write_text("{}", encoding="utf-8")

    return data_root


def test_train_eval_test_dispatch_to_runtime_entrypoints(monkeypatch) -> None:
    cli_surface = _load_cli_surface_module()
    seen: list[tuple[str, str, str | None]] = []

    def _fake_loader(command_name: str):
        def _entrypoint(args):
            seen.append(
                (
                    command_name,
                    str(args.config),
                    getattr(args, "split", None),
                )
            )
            return 0

        return _entrypoint

    monkeypatch.setattr(cli_surface, "_load_runtime_entrypoint", _fake_loader)

    assert (
        cli_surface.train_main(
            [
                "--config",
                "configs/refcoco/refcoco_reasonseg.yaml",
                "--data-root",
                "datasets",
                "--output-dir",
                "outputs/train",
            ]
        )
        == 0
    )
    assert (
        cli_surface.eval_main(
            [
                "--config",
                "configs/refcoco/refcoco_reasonseg.yaml",
                "--data-root",
                "datasets",
                "--checkpoint",
                "model.pth",
                "--split",
                "refcoco_val_unc",
                "--output-dir",
                "outputs/eval",
            ]
        )
        == 0
    )
    assert (
        cli_surface.test_main(
            [
                "--config",
                "configs/refcoco/refcoco_reasonseg.yaml",
                "--data-root",
                "datasets",
                "--checkpoint",
                "model.pth",
                "--split",
                "refcoco_testA_unc",
                "--output-dir",
                "outputs/test",
            ]
        )
        == 0
    )

    assert seen == [
        ("train", "configs/refcoco/refcoco_reasonseg.yaml", None),
        ("eval", "configs/refcoco/refcoco_reasonseg.yaml", "refcoco_val_unc"),
        ("test", "configs/refcoco/refcoco_reasonseg.yaml", "refcoco_testA_unc"),
    ]


@pytest.mark.parametrize("entrypoint", ["eval_main", "test_main"])
def test_eval_like_entrypoints_require_checkpoint(entrypoint: str, capsys) -> None:
    cli_surface = _load_cli_surface_module()
    argv = [
        "--config",
        "configs/refcoco/refcoco_reasonseg.yaml",
        "--data-root",
        "datasets",
        "--split",
        "refcoco_val_unc",
        "--output-dir",
        "outputs/eval",
    ]

    with pytest.raises(SystemExit) as error:
        getattr(cli_surface, entrypoint)(argv)

    assert error.value.code == 2
    assert "--checkpoint" in capsys.readouterr().err


def test_eval_command_rejects_unknown_refcoco_split_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_surface = _load_cli_surface_module()
    data_root = _build_materialized_dataset_root(tmp_path)
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_text("checkpoint\n", encoding="utf-8")

    def _validate_split(args):
        _ = get_registered_refexp_dataset(args.split, args.data_root)
        return 0

    monkeypatch.setattr(
        cli_surface,
        "_load_runtime_entrypoint",
        lambda command_name: _validate_split,
    )

    with pytest.raises(
        RefExpRegistryError, match="Unknown RefCOCO dataset alias 'refcoco_dev_unc'"
    ):
        cli_surface.eval_main(
            [
                "--config",
                "configs/refcoco/refcoco_reasonseg.yaml",
                "--data-root",
                str(data_root),
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "refcoco_dev_unc",
                "--output-dir",
                str(tmp_path / "outputs" / "eval"),
            ]
        )


def test_root_script_smoke_matrix_writes_canonical_runtime_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_surface = _load_cli_surface_module()
    train_script = _load_script_module("scripts/train.py", "reasonseg_train_script")
    eval_script = _load_script_module("scripts/eval.py", "reasonseg_eval_script")
    test_script = _load_script_module("scripts/test.py", "reasonseg_test_script")
    seen: list[tuple[str, str, str | None]] = []

    def _fake_loader(command_name: str):
        def _entrypoint(args):
            seen.append(
                (
                    command_name,
                    str(Path(args.config).resolve()),
                    getattr(args, "split", None),
                )
            )
            output_dir = runtime_common.get_output_dir(args)
            output_dir.mkdir(parents=True, exist_ok=True)
            runtime_common.get_config_dump_path(output_dir).write_text(
                json.dumps(
                    {
                        "command": command_name,
                        "config": str(Path(args.config).resolve()),
                        "split": getattr(args, "split", None),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / runtime_common.LOG_NAME).write_text(
                f"{command_name} smoke\n", encoding="utf-8"
            )
            if command_name == "train":
                runtime_common.write_json_artifact(
                    runtime_common.get_train_metrics_path(output_dir),
                    {"loss_total": 0.0},
                )
                checkpoint_path = output_dir / "model_final.pth"
                checkpoint_path.write_text("checkpoint\n", encoding="utf-8")
                runtime_common.write_last_checkpoint(output_dir, checkpoint_path)
            else:
                inference_dir = runtime_common.get_inference_output_dir(output_dir)
                runtime_common.write_json_artifact(
                    inference_dir / runtime_common.PREDICTIONS_NAME,
                    [{"dataset_name": args.split, "intersection": 1, "union": 1}],
                )
                runtime_common.write_json_artifact(
                    inference_dir / runtime_common.METRICS_NAME,
                    {"grounding": {"grounding/mIoU": 100.0}},
                )
            return 0

        return _entrypoint

    monkeypatch.setattr(cli_surface, "_load_runtime_entrypoint", _fake_loader)
    monkeypatch.setattr(train_script, "_load_cli_surface", lambda: cli_surface)
    monkeypatch.setattr(eval_script, "_load_cli_surface", lambda: cli_surface)
    monkeypatch.setattr(test_script, "_load_cli_surface", lambda: cli_surface)

    smoke_cases: list[SmokeCase] = [
        (
            "refcoco-train",
            train_script.main,
            ROOT / "configs/refcoco/refcoco_reasonseg.yaml",
            None,
            tmp_path / "outputs" / "refcoco-train",
            [
                runtime_common.CONFIG_DUMP_NAME,
                runtime_common.LOG_NAME,
                runtime_common.TRAIN_METRICS_NAME,
                "model_final.pth",
                runtime_common.LAST_CHECKPOINT_NAME,
            ],
        ),
        (
            "refcoco-eval",
            eval_script.main,
            ROOT / "configs/refcoco/refcoco_reasonseg.yaml",
            "refcoco_val_unc",
            tmp_path / "outputs" / "refcoco-eval",
            [
                runtime_common.CONFIG_DUMP_NAME,
                runtime_common.LOG_NAME,
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.PREDICTIONS_NAME}",
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.METRICS_NAME}",
            ],
        ),
        (
            "refcoco-plus-eval",
            eval_script.main,
            ROOT / "configs/refcoco/refcoco_plus_reasonseg.yaml",
            "refcoco_plus_val_unc",
            tmp_path / "outputs" / "refcoco-plus-eval",
            [
                runtime_common.CONFIG_DUMP_NAME,
                runtime_common.LOG_NAME,
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.PREDICTIONS_NAME}",
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.METRICS_NAME}",
            ],
        ),
        (
            "refcoco-plus-test",
            test_script.main,
            ROOT / "configs/refcoco/refcoco_plus_reasonseg.yaml",
            "refcoco_plus_testA_unc",
            tmp_path / "outputs" / "refcoco-plus-test",
            [
                runtime_common.CONFIG_DUMP_NAME,
                runtime_common.LOG_NAME,
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.PREDICTIONS_NAME}",
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.METRICS_NAME}",
            ],
        ),
        (
            "refcocog-eval",
            eval_script.main,
            ROOT / "configs/refcoco/refcocog_reasonseg.yaml",
            "refcocog_val_umd",
            tmp_path / "outputs" / "refcocog-eval",
            [
                runtime_common.CONFIG_DUMP_NAME,
                runtime_common.LOG_NAME,
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.PREDICTIONS_NAME}",
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.METRICS_NAME}",
            ],
        ),
        (
            "refcocog-test",
            test_script.main,
            ROOT / "configs/refcoco/refcocog_reasonseg.yaml",
            "refcocog_test_umd",
            tmp_path / "outputs" / "refcocog-test",
            [
                runtime_common.CONFIG_DUMP_NAME,
                runtime_common.LOG_NAME,
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.PREDICTIONS_NAME}",
                f"{runtime_common.INFERENCE_SUBDIR_NAME}/{runtime_common.METRICS_NAME}",
            ],
        ),
    ]

    checkpoint_path = tmp_path / "smoke-model.pth"
    checkpoint_path.write_text("checkpoint\n", encoding="utf-8")
    data_root = tmp_path / "datasets"
    data_root.mkdir()

    for name, main_func, config_path, split, output_dir, expected_files in smoke_cases:
        argv = [
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ]
        if split is not None:
            argv.extend(["--checkpoint", str(checkpoint_path), "--split", split])

        assert main_func(argv) == 0, name
        run_dir = output_dir / "run_0"
        for relative_path in expected_files:
            assert (run_dir / relative_path).is_file(), (name, relative_path)

    assert seen == [
        (
            "train",
            str((ROOT / "configs/refcoco/refcoco_reasonseg.yaml").resolve()),
            None,
        ),
        (
            "eval",
            str((ROOT / "configs/refcoco/refcoco_reasonseg.yaml").resolve()),
            "refcoco_val_unc",
        ),
        (
            "eval",
            str((ROOT / "configs/refcoco/refcoco_plus_reasonseg.yaml").resolve()),
            "refcoco_plus_val_unc",
        ),
        (
            "test",
            str((ROOT / "configs/refcoco/refcoco_plus_reasonseg.yaml").resolve()),
            "refcoco_plus_testA_unc",
        ),
        (
            "eval",
            str((ROOT / "configs/refcoco/refcocog_reasonseg.yaml").resolve()),
            "refcocog_val_umd",
        ),
        (
            "test",
            str((ROOT / "configs/refcoco/refcocog_reasonseg.yaml").resolve()),
            "refcocog_test_umd",
        ),
    ]
