# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import reasonseg.evaluation.grounding as grounding_module
from reasonseg.runtime import common as runtime_common
from reasonseg.runtime import eval as runtime_eval
from reasonseg.runtime import train as runtime_train


class _RuntimeCfg:
    def __init__(self, output_dir: Path, *, max_iter: int = 1, test_split: str = ""):
        self.OUTPUT_DIR = str(output_dir)
        self.SOLVER = SimpleNamespace(
            MAX_ITER=max_iter,
            CHECKPOINT_PERIOD=1,
            IMS_PER_BATCH=2,
            BASE_LR=0.001,
        )
        self.DATASETS = SimpleNamespace(
            TRAIN=("refcoco_train_unc",),
            TEST=(test_split,) if test_split else (),
        )
        self.MODEL = SimpleNamespace(WEIGHTS="cfg-model-weights.pth")
        self.TEST = SimpleNamespace(EVAL_PERIOD=0)
        self.SEED = -1
        self.CUDNN_BENCHMARK = False

    def dump(self) -> str:
        return "OUTPUT_DIR: test\n"


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[object] = []

    def forward(self, batch):
        self.calls.append(batch)
        return {
            "loss_total": (self.weight * 0)
            + torch.tensor(float(batch), dtype=torch.float32)
        }


class _FakeScheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


class _FakeProgressBar:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.postfixes: list[dict[str, str]] = []
        self.closed = False

    def update(self, value: int = 1) -> None:
        self.updates.append(value)

    def set_postfix(self, values: dict[str, str]) -> None:
        self.postfixes.append(values)

    def close(self) -> None:
        self.closed = True


def test_setup_runtime_logging_writes_canonical_run_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    cfg = _RuntimeCfg(output_dir)
    seen: dict[str, object] = {"setup_logger": []}

    def _setup_logger(
        output: str, distributed_rank: int, name: str = "detectron2"
    ) -> None:
        assert distributed_rank == 0
        cast(list[str], seen["setup_logger"]).append(name)
        (Path(output) / runtime_common.LOG_NAME).write_text(
            "runtime log\n", encoding="utf-8"
        )

    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "comm": SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 1),
            "default_setup": lambda cfg, args: (_ for _ in ()).throw(
                AssertionError("default_setup should not run")
            ),
            "seed_all_rng": lambda seed: seen.__setitem__("seed", seed),
            "setup_logger": _setup_logger,
        },
    )

    runtime_common.setup_runtime_logging(
        cfg,
        argparse.Namespace(
            config=tmp_path / "config.yaml",
            dist_url="auto",
            machine_rank=0,
            num_gpus=1,
            num_machines=1,
            opts=[],
            resume=False,
        ),
        eval_only=False,
    )

    assert (
        runtime_common.get_config_dump_path(output_dir).read_text(encoding="utf-8")
        == cfg.dump()
    )
    assert (output_dir / runtime_common.LOG_NAME).read_text(
        encoding="utf-8"
    ) == "runtime log\n"
    assert seen["setup_logger"] == ["fvcore", "detectron2", "reasonseg"]
    assert seen["seed"] is None


def test_last_checkpoint_round_trip_requires_a_real_checkpoint_file(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    output_dir.mkdir(parents=True)
    checkpoint_path = output_dir / "model_final.pth"
    checkpoint_path.write_text("checkpoint\n", encoding="utf-8")

    runtime_common.write_last_checkpoint(output_dir, checkpoint_path)

    assert runtime_common.read_last_checkpoint(output_dir) == checkpoint_path.resolve()

    runtime_common.get_last_checkpoint_path(output_dir).write_text(
        "missing_model.pth\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        runtime_common.read_last_checkpoint(output_dir)


def test_train_worker_uses_pretrained_checkpoint_without_resuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    pretrained_path = tmp_path / "pretrained.pth"
    pretrained_path.write_text("weights\n", encoding="utf-8")
    cfg = _RuntimeCfg(output_dir, max_iter=3)
    cfg.SOLVER.CHECKPOINT_PERIOD = 2
    model = _FakeModel()
    scheduler = _FakeScheduler()
    seen: dict[str, object] = {}

    class _FakeCheckpointer:
        def __init__(self, model, *, save_dir: str, optimizer, scheduler) -> None:
            seen["checkpointer"] = self
            self.save_dir = save_dir
            self.resume_or_load_calls: list[tuple[str | None, bool]] = []
            self.save_calls: list[tuple[str, dict[str, int]]] = []

        def resume_or_load(self, path: str | None, *, resume: bool):
            self.resume_or_load_calls.append((path, resume))
            return {}

        def save(self, name: str, **kwargs) -> None:
            self.save_calls.append((name, kwargs))
            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / f"{name}.pth").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(runtime_common, "setup_cfg", lambda args: cfg)
    monkeypatch.setattr(
        runtime_common, "setup_runtime_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_common, "maybe_wrap_model", lambda wrapped_model: wrapped_model
    )
    monkeypatch.setattr(
        runtime_common, "build_refcoco_train_loader", lambda cfg: [1, 2, 3]
    )
    monkeypatch.setattr(
        runtime_common,
        "build_optimizer",
        lambda cfg, model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "DetectionCheckpointer": _FakeCheckpointer,
            "build_lr_scheduler": lambda cfg, optimizer: scheduler,
            "build_model": lambda cfg: model,
            "comm": SimpleNamespace(
                reduce_dict=lambda loss_dict: loss_dict,
                is_main_process=lambda: True,
            ),
        },
    )

    result = runtime_train._worker(
        argparse.Namespace(resume=False, checkpoint=pretrained_path)
    )

    checkpointer = seen["checkpointer"]
    assert checkpointer.resume_or_load_calls == [
        (str(pretrained_path.resolve()), False)
    ]
    assert checkpointer.save_calls == [
        ("model_0000001", {"iteration": 1}),
        ("model_final", {"iteration": 2}),
    ]
    assert model.calls == [1, 2, 3]
    assert scheduler.step_count == 3
    assert (
        json.loads(runtime_common.get_train_metrics_path(output_dir).read_text())
        == result
    )
    assert (
        runtime_common.read_last_checkpoint(output_dir)
        == (output_dir / "model_final.pth").resolve()
    )


def test_train_worker_resume_uses_last_checkpoint_and_continues_from_saved_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    output_dir.mkdir(parents=True)
    resumed_checkpoint = output_dir / "model_final.pth"
    resumed_checkpoint.write_text("checkpoint\n", encoding="utf-8")
    runtime_common.write_last_checkpoint(output_dir, resumed_checkpoint)
    cfg = _RuntimeCfg(output_dir, max_iter=3)
    cfg.SOLVER.CHECKPOINT_PERIOD = 2
    model = _FakeModel()
    scheduler = _FakeScheduler()
    seen: dict[str, object] = {}

    class _FakeCheckpointer:
        def __init__(self, model, *, save_dir: str, optimizer, scheduler) -> None:
            seen["checkpointer"] = self
            self.save_dir = save_dir
            self.resume_or_load_calls: list[tuple[str | None, bool]] = []

        def resume_or_load(self, path: str | None, *, resume: bool):
            self.resume_or_load_calls.append((path, resume))
            return {"iteration": 1}

        def save(self, name: str, **kwargs) -> None:
            (Path(self.save_dir) / f"{name}.pth").write_text(
                "checkpoint\n", encoding="utf-8"
            )

    monkeypatch.setattr(runtime_common, "setup_cfg", lambda args: cfg)
    monkeypatch.setattr(
        runtime_common, "setup_runtime_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_common, "maybe_wrap_model", lambda wrapped_model: wrapped_model
    )
    monkeypatch.setattr(
        runtime_common, "build_refcoco_train_loader", lambda cfg: [10, 20, 30]
    )
    monkeypatch.setattr(
        runtime_common,
        "build_optimizer",
        lambda cfg, model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "DetectionCheckpointer": _FakeCheckpointer,
            "build_lr_scheduler": lambda cfg, optimizer: scheduler,
            "build_model": lambda cfg: model,
            "comm": SimpleNamespace(
                reduce_dict=lambda loss_dict: loss_dict,
                is_main_process=lambda: True,
            ),
        },
    )

    runtime_train._worker(
        argparse.Namespace(resume=True, checkpoint=tmp_path / "ignored-pretrained.pth")
    )

    checkpointer = seen["checkpointer"]
    assert checkpointer.resume_or_load_calls == [
        (str(resumed_checkpoint.resolve()), False)
    ]
    assert model.calls == [10]
    assert scheduler.step_count == 1


def test_train_worker_logs_periodic_progress_and_final_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    cfg = _RuntimeCfg(output_dir, max_iter=3)
    cfg.SOLVER.CHECKPOINT_PERIOD = 2
    model = _FakeModel()
    scheduler = _FakeScheduler()

    class _FakeCheckpointer:
        def __init__(self, model, *, save_dir: str, optimizer, scheduler) -> None:
            self.save_dir = save_dir

        def resume_or_load(self, path: str | None, *, resume: bool):
            return {}

        def save(self, name: str, **kwargs) -> None:
            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / f"{name}.pth").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(runtime_common, "setup_cfg", lambda args: cfg)
    monkeypatch.setattr(
        runtime_common, "setup_runtime_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_common, "maybe_wrap_model", lambda wrapped_model: wrapped_model
    )
    monkeypatch.setattr(
        runtime_common, "build_refcoco_train_loader", lambda cfg: [1, 2, 3]
    )
    monkeypatch.setattr(
        runtime_common,
        "build_optimizer",
        lambda cfg, model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "DetectionCheckpointer": _FakeCheckpointer,
            "build_lr_scheduler": lambda cfg, optimizer: scheduler,
            "build_model": lambda cfg: model,
            "comm": SimpleNamespace(
                reduce_dict=lambda loss_dict: loss_dict,
                is_main_process=lambda: True,
            ),
        },
    )

    with caplog.at_level(logging.INFO, logger="reasonseg"):
        runtime_train._worker(argparse.Namespace(resume=False, checkpoint=None))

    progress_messages = [
        record.message
        for record in caplog.records
        if record.name == "reasonseg" and record.message.startswith("train progress")
    ]
    assert len(progress_messages) == 3
    assert "iter=1/3" in progress_messages[0]
    assert "iter=2/3" in progress_messages[1]
    assert "iter=3/3" in progress_messages[2]


def test_train_worker_updates_tqdm_with_current_step_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    cfg = _RuntimeCfg(output_dir, max_iter=3)
    model = _FakeModel()
    scheduler = _FakeScheduler()
    seen: dict[str, object] = {}

    class _FakeCheckpointer:
        def __init__(self, model, *, save_dir: str, optimizer, scheduler) -> None:
            self.save_dir = save_dir

        def resume_or_load(self, path: str | None, *, resume: bool):
            return {}

        def save(self, name: str, **kwargs) -> None:
            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / f"{name}.pth").write_text("checkpoint\n", encoding="utf-8")

    def _fake_tqdm(*args, **kwargs):
        bar = _FakeProgressBar(*args, **kwargs)
        seen["bar"] = bar
        return bar

    monkeypatch.setattr(runtime_train, "tqdm", _fake_tqdm)
    monkeypatch.setattr(runtime_common, "setup_cfg", lambda args: cfg)
    monkeypatch.setattr(
        runtime_common, "setup_runtime_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_common, "maybe_wrap_model", lambda wrapped_model: wrapped_model
    )
    monkeypatch.setattr(
        runtime_common, "build_refcoco_train_loader", lambda cfg: [1, 2, 3]
    )
    monkeypatch.setattr(
        runtime_common,
        "build_optimizer",
        lambda cfg, model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "DetectionCheckpointer": _FakeCheckpointer,
            "build_lr_scheduler": lambda cfg, optimizer: scheduler,
            "build_model": lambda cfg: model,
            "comm": SimpleNamespace(
                reduce_dict=lambda loss_dict: loss_dict,
                is_main_process=lambda: True,
            ),
        },
    )

    runtime_train._worker(argparse.Namespace(resume=False, checkpoint=None))

    bar = cast(_FakeProgressBar, seen["bar"])
    assert bar.kwargs["desc"] == "train"
    assert bar.kwargs["total"] == 3
    assert bar.updates == [1, 1, 1]
    assert bar.postfixes == [
        {"loss": "1.0000"},
        {"loss": "2.0000"},
        {"loss": "3.0000"},
    ]
    assert bar.closed


def test_train_worker_logs_configuration_and_checkpoint_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    cfg = _RuntimeCfg(output_dir, max_iter=3)
    cfg.SOLVER.CHECKPOINT_PERIOD = 2
    model = _FakeModel()
    scheduler = _FakeScheduler()

    class _FakeCheckpointer:
        def __init__(self, model, *, save_dir: str, optimizer, scheduler) -> None:
            self.save_dir = save_dir

        def resume_or_load(self, path: str | None, *, resume: bool):
            return {}

        def save(self, name: str, **kwargs) -> None:
            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / f"{name}.pth").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(runtime_common, "setup_cfg", lambda args: cfg)
    monkeypatch.setattr(
        runtime_common, "setup_runtime_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_common, "maybe_wrap_model", lambda wrapped_model: wrapped_model
    )
    monkeypatch.setattr(
        runtime_common, "build_refcoco_train_loader", lambda cfg: [1, 2]
    )
    monkeypatch.setattr(
        runtime_common,
        "build_optimizer",
        lambda cfg, model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "DetectionCheckpointer": _FakeCheckpointer,
            "build_lr_scheduler": lambda cfg, optimizer: scheduler,
            "build_model": lambda cfg: model,
            "comm": SimpleNamespace(
                reduce_dict=lambda loss_dict: loss_dict,
                is_main_process=lambda: True,
            ),
        },
    )

    with caplog.at_level(logging.INFO, logger="reasonseg"):
        runtime_train._worker(argparse.Namespace(resume=False, checkpoint=None))

    summary_messages = [
        record.message
        for record in caplog.records
        if record.name == "reasonseg"
        and (
            record.message.startswith("train config")
            or record.message.startswith("train init")
            or record.message.startswith("checkpoint saved")
            or record.message.startswith("train complete")
        )
    ]
    assert len(summary_messages) == 4
    assert "datasets_train=refcoco_train_unc" in summary_messages[0]
    assert "resume=false" in summary_messages[1]
    assert "checkpoint saved iter=2/3" in summary_messages[2]
    assert "train complete final_checkpoint=" in summary_messages[3]


def test_train_worker_runs_periodic_eval_and_logs_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    cfg = _RuntimeCfg(output_dir, max_iter=3, test_split="refcoco_val_unc")
    cfg.SOLVER.CHECKPOINT_PERIOD = 2
    cfg.TEST.EVAL_PERIOD = 2
    model = _FakeModel()
    scheduler = _FakeScheduler()
    seen_eval_calls: list[dict[str, object]] = []

    class _FakeCheckpointer:
        def __init__(self, model, *, save_dir: str, optimizer, scheduler) -> None:
            self.save_dir = save_dir

        def resume_or_load(self, path: str | None, *, resume: bool):
            return {}

        def save(self, name: str, **kwargs) -> None:
            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / f"{name}.pth").write_text("checkpoint\n", encoding="utf-8")

    def _fake_run_evaluation(eval_model, eval_cfg, *, deps, output_dir):
        seen_eval_calls.append(
            {
                "training": eval_model.training,
                "output_dir": Path(output_dir),
                "dataset": eval_cfg.DATASETS.TEST[0],
            }
        )
        return {"grounding": {"grounding/mIoU": 55.5}}

    monkeypatch.setattr(runtime_common, "setup_cfg", lambda args: cfg)
    monkeypatch.setattr(
        runtime_common, "setup_runtime_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_common, "maybe_wrap_model", lambda wrapped_model: wrapped_model
    )
    monkeypatch.setattr(
        runtime_common, "build_refcoco_train_loader", lambda cfg: [1, 2, 3]
    )
    monkeypatch.setattr(
        runtime_common,
        "build_optimizer",
        lambda cfg, model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(runtime_eval, "run_evaluation", _fake_run_evaluation)
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "DetectionCheckpointer": _FakeCheckpointer,
            "build_lr_scheduler": lambda cfg, optimizer: scheduler,
            "build_model": lambda cfg: model,
            "comm": SimpleNamespace(
                reduce_dict=lambda loss_dict: loss_dict,
                is_main_process=lambda: True,
            ),
        },
    )

    with caplog.at_level(logging.INFO, logger="reasonseg"):
        runtime_train._worker(argparse.Namespace(resume=False, checkpoint=None))

    assert seen_eval_calls == [
        {
            "training": False,
            "output_dir": output_dir
            / runtime_common.INFERENCE_SUBDIR_NAME
            / "iter_0000002",
            "dataset": "refcoco_val_unc",
        },
        {
            "training": False,
            "output_dir": output_dir
            / runtime_common.INFERENCE_SUBDIR_NAME
            / "iter_0000003",
            "dataset": "refcoco_val_unc",
        },
    ]
    eval_messages = [
        record.message
        for record in caplog.records
        if record.name == "reasonseg" and record.message.startswith("eval progress")
    ]
    assert eval_messages == [
        (
            "eval progress iter=2/3 dataset=refcoco_val_unc "
            f"inference_dir={output_dir / runtime_common.INFERENCE_SUBDIR_NAME / 'iter_0000002'} "
            "metrics=grounding/mIoU=55.500000"
        ),
        (
            "eval progress iter=3/3 dataset=refcoco_val_unc "
            f"inference_dir={output_dir / runtime_common.INFERENCE_SUBDIR_NAME / 'iter_0000003'} "
            "metrics=grounding/mIoU=55.500000"
        ),
    ]
    assert model.training


def test_train_worker_cleans_up_memory_after_periodic_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    cfg = _RuntimeCfg(output_dir, max_iter=2, test_split="refcoco_val_unc")
    cfg.TEST.EVAL_PERIOD = 1
    model = _FakeModel()
    scheduler = _FakeScheduler()
    seen_events: list[str] = []

    class _FakeCheckpointer:
        def __init__(self, model, *, save_dir: str, optimizer, scheduler) -> None:
            self.save_dir = save_dir

        def resume_or_load(self, path: str | None, *, resume: bool):
            return {}

        def save(self, name: str, **kwargs) -> None:
            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / f"{name}.pth").write_text("checkpoint\n", encoding="utf-8")

    def _fake_run_evaluation(eval_model, eval_cfg, *, deps, output_dir):
        seen_events.append(f"eval:{eval_model.training}")
        return {"grounding": {"grounding/mIoU": 11.0}}

    monkeypatch.setattr(runtime_common, "setup_cfg", lambda args: cfg)
    monkeypatch.setattr(
        runtime_common, "setup_runtime_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_common, "maybe_wrap_model", lambda wrapped_model: wrapped_model
    )
    monkeypatch.setattr(
        runtime_common, "build_refcoco_train_loader", lambda cfg: [1, 2]
    )
    monkeypatch.setattr(
        runtime_common,
        "build_optimizer",
        lambda cfg, model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(runtime_eval, "run_evaluation", _fake_run_evaluation)
    monkeypatch.setattr(
        runtime_train,
        "_cleanup_after_evaluation",
        lambda: seen_events.append("cleanup"),
    )
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "DetectionCheckpointer": _FakeCheckpointer,
            "build_lr_scheduler": lambda cfg, optimizer: scheduler,
            "build_model": lambda cfg: model,
            "comm": SimpleNamespace(
                reduce_dict=lambda loss_dict: loss_dict,
                is_main_process=lambda: True,
            ),
        },
    )

    runtime_train._worker(argparse.Namespace(resume=False, checkpoint=None))

    assert seen_events == ["eval:False", "cleanup", "eval:False", "cleanup"]
    assert model.training


def test_eval_worker_uses_explicit_checkpoint_and_canonical_inference_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    checkpoint_path = tmp_path / "eval-model.pth"
    checkpoint_path.write_text("checkpoint\n", encoding="utf-8")
    cfg = _RuntimeCfg(output_dir, test_split="refcoco_val_unc")
    cfg.MODEL.WEIGHTS = "wrong-model-path.pth"
    seen: dict[str, object] = {}

    class _FakeCheckpointer:
        def __init__(self, model, *, save_dir: str) -> None:
            self.resume_or_load_calls: list[tuple[str | None, bool]] = []
            seen["checkpointer"] = self

        def resume_or_load(self, path: str | None, *, resume: bool):
            self.resume_or_load_calls.append((path, resume))
            return {}

    class _FakeGroundingEvaluator:
        def __init__(
            self, dataset_name: str, *, output_dir: Path, distributed: bool
        ) -> None:
            seen["evaluator"] = {
                "dataset_name": dataset_name,
                "output_dir": output_dir,
                "distributed": distributed,
            }
            self._calls = 0

        def process(self, inputs, outputs) -> None:
            self._calls += 1

        def progress_metrics(self) -> dict[str, float]:
            return {
                "oIoU": 1.0,
                "mIoU": 1.0,
                "Prec@50": 1.0,
                "Prec@75": 1.0,
            }

        def evaluate(self) -> dict[str, dict[str, float]]:
            return {"grounding": {"cIoU": 1.0}}

    class _FakeEvalWorkerModel(torch.nn.Module):
        def forward(self, batch):
            seen["loader"] = batch
            return [{"grounding_mask": torch.ones(1, 1, 1)}]

    import reasonseg.evaluation.grounding as grounding_module

    monkeypatch.setattr(grounding_module, "GroundingEvaluator", _FakeGroundingEvaluator)
    monkeypatch.setattr(runtime_common, "setup_cfg", lambda args, **kwargs: cfg)
    monkeypatch.setattr(
        runtime_common, "setup_runtime_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_common, "maybe_wrap_model", lambda wrapped_model: wrapped_model
    )
    _fake_grounding_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    monkeypatch.setattr(
        runtime_common,
        "build_refcoco_test_loader",
        lambda cfg, dataset_name: [
            [
                {
                    "groundings": {"masks": _fake_grounding_mask, "texts": []},
                    "image_id": 1,
                    "file_name": "test.jpg",
                    "prompt": ["test"],
                }
            ]
        ],
    )
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "DetectionCheckpointer": _FakeCheckpointer,
            "build_model": lambda cfg: _FakeEvalWorkerModel(),
            "comm": SimpleNamespace(get_world_size=lambda: 1, get_rank=lambda: 0),
        },
    )

    with caplog.at_level(logging.INFO, logger="reasonseg"):
        result = runtime_eval._worker(
            argparse.Namespace(checkpoint=checkpoint_path, split="refcoco_val_unc")
        )

    checkpointer = seen["checkpointer"]
    assert checkpointer.resume_or_load_calls == [
        (str(checkpoint_path.resolve()), False)
    ]
    assert seen["evaluator"] == {
        "dataset_name": "refcoco_val_unc",
        "output_dir": runtime_common.get_inference_output_dir(output_dir),
        "distributed": False,
    }
    assert seen["loader"] is not None
    assert result == {"grounding": {"cIoU": 1.0}}
    eval_messages = [
        record.message
        for record in caplog.records
        if record.name == "reasonseg"
        and (
            record.message.startswith("eval config")
            or record.message.startswith("eval complete")
        )
    ]
    assert eval_messages == [
        (
            "eval config dataset=refcoco_val_unc "
            f"checkpoint={checkpoint_path.resolve()} output_dir={output_dir} "
            f"inference_dir={runtime_common.get_inference_output_dir(output_dir)}"
        ),
        "eval complete dataset=refcoco_val_unc metrics=cIoU=1.000000",
    ]


def test_run_evaluation_updates_tqdm_with_running_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    cfg = _RuntimeCfg(output_dir, test_split="refcoco_val_unc")
    seen: dict[str, object] = {}

    class _FakeEvalModel(torch.nn.Module):
        def forward(self, batch):
            return [
                {
                    "grounding_mask": torch.ones(1, 1, 1),
                    "grounding_scores": torch.ones(1),
                }
            ]

    class _FakeGroundingEvaluator:
        def __init__(
            self, dataset_name: str, *, output_dir: Path, distributed: bool
        ) -> None:
            seen["evaluator_init"] = {
                "dataset_name": dataset_name,
                "output_dir": output_dir,
                "distributed": distributed,
            }
            self.calls = 0

        def process(self, inputs, outputs) -> None:
            self.calls += 1

        def progress_metrics(self) -> dict[str, float]:
            return {
                "oIoU": 10.0 * self.calls,
                "mIoU": 20.0 * self.calls,
                "Prec@50": 30.0 * self.calls,
                "Prec@75": 40.0 * self.calls,
            }

        def evaluate(self) -> dict[str, dict[str, float]]:
            return {"grounding": {"grounding/cIoU": 99.0}}

    def _fake_tqdm(*args, **kwargs):
        bar = _FakeProgressBar(*args, **kwargs)
        seen["bar"] = bar
        return bar

    monkeypatch.setattr(runtime_eval, "tqdm", _fake_tqdm)
    _fake_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    _fake_input = {
        "groundings": {"masks": _fake_mask, "texts": []},
        "image_id": 1,
        "file_name": "test.jpg",
        "prompt": ["test"],
    }
    monkeypatch.setattr(
        runtime_common,
        "build_refcoco_test_loader",
        lambda cfg, dataset_name: [[_fake_input], [_fake_input]],
    )
    monkeypatch.setattr(grounding_module, "GroundingEvaluator", _FakeGroundingEvaluator)

    model = _FakeEvalModel()
    model.train()
    results = runtime_eval.run_evaluation(
        model,
        cfg,
        deps={"comm": SimpleNamespace(get_world_size=lambda: 1, get_rank=lambda: 0)},
        output_dir=runtime_common.get_inference_output_dir(output_dir),
    )

    bar = cast(_FakeProgressBar, seen["bar"])
    assert bar.kwargs["desc"] == "eval:refcoco_val_unc"
    assert bar.kwargs["total"] == 2
    assert bar.updates == [1, 1]
    assert bar.postfixes == [
        {"oIoU": "10.00", "mIoU": "20.00", "Prec@50": "30.00", "Prec@75": "40.00"},
        {"oIoU": "20.00", "mIoU": "40.00", "Prec@50": "60.00", "Prec@75": "80.00"},
    ]
    assert bar.closed
    assert model.training
    assert results == {"grounding": {"grounding/cIoU": 99.0}}


def test_eval_checkpoint_resolution_requires_existing_checkpoint_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        runtime_common.resolve_eval_checkpoint_path(
            argparse.Namespace(checkpoint=tmp_path / "missing-model.pth")
        )


def test_launch_train_main_uses_detectron2_launch_without_external_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        num_gpus=2, num_machines=1, machine_rank=0, dist_url="auto"
    )
    seen: list[object] = []

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(
        runtime_common,
        "launch_main",
        lambda main_func, call_args: seen.append((main_func, call_args)),
    )

    runtime_common.launch_train_main(runtime_train._worker, args)

    assert seen == [(runtime_train._worker, args)]


def test_launch_train_main_uses_external_distributed_context_without_detectron2_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        num_gpus=1, num_machines=1, machine_rank=0, dist_url="auto"
    )
    seen: dict[str, object] = {"events": []}

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    initialized = {"value": False}
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "comm": SimpleNamespace(
                get_local_size=lambda: 0,
                create_local_process_group=lambda size: seen["events"].append(
                    ("create_local_process_group", size)
                ),
                synchronize=lambda: seen["events"].append(("synchronize", None)),
            )
        },
    )
    monkeypatch.setattr(
        runtime_common,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                set_device=lambda device: seen["events"].append(("set_device", device)),
            ),
            distributed=SimpleNamespace(
                is_initialized=lambda: initialized["value"],
                init_process_group=lambda **kwargs: (
                    initialized.update(value=True)
                    or seen["events"].append(("init_process_group", kwargs))
                ),
                destroy_process_group=lambda: seen["events"].append(
                    ("destroy_process_group", None)
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        runtime_common,
        "launch_main",
        lambda main_func, call_args: (_ for _ in ()).throw(
            AssertionError("detectron2 launch should not run")
        ),
    )

    def _worker(call_args):
        seen["worker_args"] = call_args
        return {"ok": 1}

    result = runtime_common.launch_train_main(_worker, args)

    assert result == {"ok": 1}
    assert seen["worker_args"] is args
    assert seen["events"] == [
        (
            "init_process_group",
            {
                "backend": "NCCL",
                "init_method": "env://",
                "world_size": 2,
                "rank": 1,
            },
        ),
        ("create_local_process_group", 2),
        ("set_device", 1),
        ("synchronize", None),
        ("destroy_process_group", None),
    ]


def test_train_main_delegates_to_launch_train_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace()
    seen: list[tuple[object, object]] = []

    monkeypatch.setattr(
        runtime_common,
        "launch_train_main",
        lambda main_func, call_args: seen.append((main_func, call_args)),
    )

    assert runtime_train.main(args) == 0
    assert seen == [(runtime_train._worker, args)]


def test_grounding_evaluator_writes_canonical_inference_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs" / "run_0" / runtime_common.INFERENCE_SUBDIR_NAME
    monkeypatch.setattr(grounding_module, "detectron2_eval_available", True)
    monkeypatch.setattr(grounding_module, "synchronize", lambda: None)
    monkeypatch.setattr(grounding_module, "all_gather", lambda value: [value])
    monkeypatch.setattr(grounding_module, "is_main_process", lambda: True)

    evaluator = grounding_module.GroundingEvaluator(
        "refcoco_val_unc",
        output_dir=output_dir,
        distributed=False,
    )
    evaluator._accumulator.add(intersection=3, union=4)
    evaluator._predictions.append(
        {
            "dataset_name": "refcoco_val_unc",
            "image_id": 1,
            "prompt_index": 0,
            "prompt": "dog",
            "candidate_prompts": ["dog"],
            "score": 0.9,
            "intersection": 3.0,
            "union": 4.0,
        }
    )

    result = evaluator.evaluate()

    assert result == {
        "grounding": {
            "grounding/cIoU": 75.0,
            "grounding/mIoU": 75.0,
            "grounding/precision@0.5": 100.0,
            "grounding/precision@0.6": 100.0,
            "grounding/precision@0.7": 100.0,
            "grounding/precision@0.8": 0.0,
            "grounding/precision@0.9": 0.0,
        }
    }
    assert json.loads((output_dir / runtime_common.PREDICTIONS_NAME).read_text()) == [
        {
            "dataset_name": "refcoco_val_unc",
            "image_id": 1,
            "prompt_index": 0,
            "prompt": "dog",
            "candidate_prompts": ["dog"],
            "score": 0.9,
            "intersection": 3.0,
            "union": 4.0,
        }
    ]
    assert json.loads((output_dir / runtime_common.METRICS_NAME).read_text()) == result
