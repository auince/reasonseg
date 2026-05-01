# pyright: reportMissingImports=false, reportExplicitAny=false, reportAny=false
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml


CONFIG_DUMP_NAME = "config.yaml"
LOG_NAME = "log.txt"
TRAIN_METRICS_NAME = "train_metrics.json"
LAST_CHECKPOINT_NAME = "last_checkpoint"
INFERENCE_SUBDIR_NAME = "inference"
PREDICTIONS_NAME = "predictions.json"
METRICS_NAME = "metrics.json"


_runtime_deps: dict[str, Any] | None = None


def _load_config_source_dict(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {config_path}")

    base_entry = payload.pop("_BASE_", None)
    merged_payload: dict[str, Any] = {}
    if base_entry:
        base_paths = (base_entry,) if isinstance(base_entry, str) else tuple(base_entry)
        for relative_base in base_paths:
            base_payload = _load_config_source_dict((config_path.parent / relative_base).resolve())
            merged_payload = _deep_merge_dicts(merged_payload, base_payload)
    return _deep_merge_dicts(merged_payload, payload)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _apply_config_overrides(payload: dict[str, Any], opts: list[str]) -> dict[str, Any]:
    if len(opts) % 2 != 0:
        raise ValueError("Config overrides must be passed as KEY VALUE pairs.")

    updated_payload = copy.deepcopy(payload)
    for key, raw_value in zip(opts[0::2], opts[1::2]):
        cursor = updated_payload
        segments = key.split(".")
        for segment in segments[:-1]:
            next_value = cursor.get(segment)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[segment] = next_value
            cursor = next_value
        cursor[segments[-1]] = yaml.safe_load(raw_value)
    return updated_payload


def _config_uses_canonical_vr_ov(config_path: Path, opts: list[str]) -> bool:
    payload = _apply_config_overrides(_load_config_source_dict(config_path), opts)
    model_payload = payload.get("MODEL", {})
    if not isinstance(model_payload, dict):
        return False
    return model_payload.get("META_ARCHITECTURE") == "VR_OV"


def _import_runtime_deps() -> dict[str, Any]:
    global _runtime_deps
    if _runtime_deps is not None:
        return _runtime_deps
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import (
        build_detection_test_loader,
        build_detection_train_loader,
    )
    from detectron2.engine import create_ddp_model, default_setup, launch
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import build_lr_scheduler
    from detectron2.solver.build import maybe_add_gradient_clipping
    from detectron2.utils import comm
    from detectron2.utils.env import seed_all_rng
    from detectron2.utils.logger import setup_logger

    _runtime_deps = {
        "DetectionCheckpointer": DetectionCheckpointer,
        "build_detection_test_loader": build_detection_test_loader,
        "build_detection_train_loader": build_detection_train_loader,
        "build_lr_scheduler": build_lr_scheduler,
        "build_model": build_model,
        "comm": comm,
        "create_ddp_model": create_ddp_model,
        "default_setup": default_setup,
        "get_cfg": get_cfg,
        "launch": launch,
        "maybe_add_gradient_clipping": maybe_add_gradient_clipping,
        "seed_all_rng": seed_all_rng,
        "setup_logger": setup_logger,
    }
    return _runtime_deps


def get_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).expanduser().resolve() / f"run_{args.run_index}"


def get_config_dump_path(output_dir: Path | str) -> Path:
    return Path(output_dir) / CONFIG_DUMP_NAME


def get_train_metrics_path(output_dir: Path | str) -> Path:
    return Path(output_dir) / TRAIN_METRICS_NAME


def get_last_checkpoint_path(output_dir: Path | str) -> Path:
    return Path(output_dir) / LAST_CHECKPOINT_NAME


def get_inference_output_dir(output_dir: Path | str) -> Path:
    return Path(output_dir) / INFERENCE_SUBDIR_NAME


def write_json_artifact(path: Path | str, payload: Any) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_last_checkpoint(output_dir: Path | str, checkpoint_path: Path | str) -> Path:
    output_path = Path(output_dir)
    resolved_checkpoint = Path(checkpoint_path)
    record_path = get_last_checkpoint_path(output_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_value = (
        resolved_checkpoint.name
        if resolved_checkpoint.parent == output_path
        else str(resolved_checkpoint.resolve())
    )
    record_path.write_text(checkpoint_value + "\n", encoding="utf-8")
    return record_path


def read_last_checkpoint(output_dir: Path | str) -> Path | None:
    record_path = get_last_checkpoint_path(output_dir)
    if not record_path.exists():
        return None
    raw_value = record_path.read_text(encoding="utf-8").strip()
    if not raw_value:
        raise RuntimeError(f"Checkpoint record {record_path} is empty.")
    checkpoint_path = Path(raw_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = record_path.parent / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint record {record_path} points to missing file {checkpoint_path}."
        )
    return checkpoint_path


def resolve_train_checkpoint_path(
    args: argparse.Namespace, output_dir: Path | str
) -> str | None:
    if getattr(args, "resume", False):
        checkpoint_path = read_last_checkpoint(output_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(
                "--resume requires an existing output run with a valid last_checkpoint file."
            )
        return str(checkpoint_path)
    if getattr(args, "checkpoint", None):
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        return str(checkpoint_path)
    return None


def resolve_eval_checkpoint_path(args: argparse.Namespace) -> str:
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is None:
        raise ValueError("Eval/test requires an explicit --checkpoint path.")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return str(checkpoint_path)


def setup_cfg(
    args: argparse.Namespace,
    *,
    eval_split: str | None = None,
    force_grounding_eval: bool = False,
) -> Any:
    deps = _import_runtime_deps()

    from model.vr_ov_config import validate_vr_ov_config
    from reasonseg.data.runtime_refcoco import register_refcoco_datasets
    from reasonseg.modeling import add_open_world_sam2_config  # noqa: F401

    data_root = Path(args.data_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    os.environ["DETECTRON2_DATASETS"] = str(data_root)
    register_refcoco_datasets(data_root)

    cfg = deps["get_cfg"]()
    cfg.set_new_allowed(True)
    add_open_world_sam2_config(
        cfg,
        include_vr_ov_compat=not _config_uses_canonical_vr_ov(config_path, list(args.opts)),
    )
    cfg.merge_from_file(str(config_path))
    cfg.merge_from_list(list(args.opts))

    if eval_split is not None:
        cfg.DATASETS.TEST = (eval_split,)
    if getattr(args, "checkpoint", None):
        cfg.MODEL.WEIGHTS = str(Path(args.checkpoint).expanduser().resolve())
    if getattr(args, "batch_size", None) is not None:
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    if getattr(args, "lr", None) is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if getattr(args, "max_iter", None) is not None:
        cfg.SOLVER.MAX_ITER = args.max_iter

    cfg.OUTPUT_DIR = str(get_output_dir(args))

    if force_grounding_eval:
        cfg.MODEL.OpenWorldSAM2.TEST.REFER_ON = True
        cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON = False
        cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON = False
        cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON = False

    validate_vr_ov_config(cfg)
    cfg.freeze()
    return cfg


def setup_runtime_logging(
    cfg: Any, args: argparse.Namespace, *, eval_only: bool
) -> None:
    deps = _import_runtime_deps()
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    rank = int(deps["comm"].get_rank())
    world_size = int(deps["comm"].get_world_size())
    deps["setup_logger"](output=cfg.OUTPUT_DIR, distributed_rank=rank, name="fvcore")
    deps["setup_logger"](output=cfg.OUTPUT_DIR, distributed_rank=rank)
    deps["setup_logger"](
        output=cfg.OUTPUT_DIR,
        distributed_rank=rank,
        name="reasonseg",
    )
    get_config_dump_path(cfg.OUTPUT_DIR).write_text(cfg.dump(), encoding="utf-8")
    logger = logging.getLogger("reasonseg")
    logger.info(
        "runtime setup mode=%s rank=%d world_size=%d config=%s output_dir=%s",
        "eval" if eval_only else "train",
        rank,
        world_size,
        str(Path(args.config).expanduser().resolve()),
        cfg.OUTPUT_DIR,
    )
    seed = int(getattr(cfg, "SEED", -1))
    deps["seed_all_rng"](None if seed < 0 else seed + rank)
    if not eval_only:
        torch.backends.cudnn.benchmark = bool(getattr(cfg, "CUDNN_BENCHMARK", False))


def maybe_wrap_model(model: torch.nn.Module) -> torch.nn.Module:
    deps = _import_runtime_deps()
    if deps["comm"].get_world_size() > 1:
        return deps["create_ddp_model"](model, broadcast_buffers=False, find_unused_parameters=True)
    return model


def build_refcoco_train_loader(cfg: Any) -> Any:
    deps = _import_runtime_deps()
    from reasonseg.data.dataset_mappers import RefCOCODatasetMapper

    return deps["build_detection_train_loader"](
        cfg,
        mapper=RefCOCODatasetMapper(cfg, is_train=True),
    )


def build_refcoco_test_loader(cfg: Any, dataset_name: str) -> Any:
    deps = _import_runtime_deps()
    from reasonseg.data.dataset_mappers import RefCOCODatasetMapper

    return deps["build_detection_test_loader"](
        cfg,
        dataset_name=dataset_name,
        mapper=RefCOCODatasetMapper(cfg, is_train=False),
    )


def build_optimizer(cfg: Any, model: torch.nn.Module) -> torch.optim.Optimizer:
    deps = _import_runtime_deps()
    defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: list[dict[str, Any]] = []
    memo: set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad or value in memo:
                continue
            memo.add(value)
            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
            if (
                "relative_position_bias_table" in module_param_name
                or "absolute_pos_embed" in module_param_name
            ):
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    optimizer: torch.optim.Optimizer
    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    else:
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
    if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
        optimizer = deps["maybe_add_gradient_clipping"](cfg, optimizer)
    return optimizer


def is_external_distributed_launch() -> bool:
    return "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def run_with_external_distributed_context(
    main_func: Any, args: argparse.Namespace
) -> Any:
    deps = _import_runtime_deps()
    dist = torch.distributed
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    has_gpu = torch.cuda.is_available()

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="NCCL" if has_gpu else "GLOO",
            init_method="env://",
            world_size=world_size,
            rank=int(os.environ["RANK"]),
        )

    try:
        if world_size > 1:
            deps["comm"].create_local_process_group(local_world_size)

        if has_gpu:
            torch.cuda.set_device(local_rank)

        if world_size > 1:
            deps["comm"].synchronize()

        return main_func(args)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def launch_train_main(main_func: Any, args: argparse.Namespace) -> Any:
    if is_external_distributed_launch():
        return run_with_external_distributed_context(main_func, args)
    return launch_main(main_func, args)


def launch_main(main_func: Any, args: argparse.Namespace) -> Any:
    deps = _import_runtime_deps()
    return deps["launch"](
        main_func,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
