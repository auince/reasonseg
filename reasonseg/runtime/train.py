# pyright: reportMissingImports=false, reportExplicitAny=false, reportAny=false, reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from tqdm.auto import tqdm


DEFAULT_LOG_PERIOD = 20
VR_OV_LEVEL_ORDER = ("L1", "L2", "L3", "L4")
_VR_OV_FILTERABLE_BATCH_FIELDS = (
    "prompt",
    "unique_categories",
    "instances",
    "query_text",
    "query_struct",
    "requested_target",
    "slice_tags",
    "positive_mask_count",
    "query_metadata",
    "composed_prompt",
)
_VR_OV_PHASE_TRAINABLE_PREFIXES: dict[str, tuple[str, ...]] = {
    "1a": ("vr_ov_scene_graph",),
    "1b": ("vr_ov_comp_matcher",),
    "1c": ("vr_ov_refine_decoder",),
}

_loss_history: list[tuple[int, float]] = []


@dataclass(frozen=True)
class VROVCurriculumState:
    levels: tuple[str, ...]
    switch_interval: int

    def level_for_iteration(self, iteration: int) -> str:
        stage_index = min(iteration // self.switch_interval, len(self.levels) - 1)
        return self.levels[stage_index]


@dataclass
class VROVBatchMetrics:
    current_level: str
    total_prompts: int = 0
    kept_prompts: int = 0
    forced_prompt_keeps: int = 0


@dataclass
class VROVRuntimeMetrics:
    batch_count: int = 0
    total_prompts: int = 0
    kept_prompts: int = 0
    forced_prompt_keeps: int = 0
    dropout_events: int = 0
    query_nodes_seen: int = 0
    query_nodes_kept: int = 0
    loss_history: list[float] = field(default_factory=list)
    last_batch: VROVBatchMetrics | None = None


class ModelEmaState:
    def __init__(self, model: torch.nn.Module, *, decay: float) -> None:
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow_params = {
            name: param.detach().cpu().clone()
            for name, param in _unwrap_model(model).named_parameters()
            if param.is_floating_point()
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": {
                name: tensor.clone() for name, tensor in self.shadow_params.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict["num_updates"])
        self.shadow_params = {
            str(name): cast(torch.Tensor, tensor).detach().cpu().clone()
            for name, tensor in cast(dict[str, torch.Tensor], state_dict["shadow_params"]).items()
        }

    def update(self, model: torch.nn.Module) -> None:
        root_model = _unwrap_model(model)
        with torch.no_grad():
            for name, param in root_model.named_parameters():
                if name not in self.shadow_params or not param.is_floating_point():
                    continue
                source = param.detach().cpu()
                self.shadow_params[name].mul_(self.decay).add_(source, alpha=1.0 - self.decay)
        self.num_updates += 1

    def save_ema_checkpoint(
        self,
        checkpointer: object,
        model: torch.nn.Module,
        *,
        checkpoint_name: str,
        iteration: int,
    ) -> None:
        root_model = _unwrap_model(model)
        original_params = {
            name: param.detach().cpu().clone()
            for name, param in root_model.named_parameters()
            if name in self.shadow_params and param.is_floating_point()
        }
        try:
            with torch.no_grad():
                for name, param in root_model.named_parameters():
                    if name not in self.shadow_params or not param.is_floating_point():
                        continue
                    param.copy_(self.shadow_params[name].to(device=param.device, dtype=param.dtype))
            checkpointer.save(checkpoint_name, iteration=iteration)
        finally:
            with torch.no_grad():
                for name, param in root_model.named_parameters():
                    if name not in original_params or not param.is_floating_point():
                        continue
                    param.copy_(original_params[name].to(device=param.device, dtype=param.dtype))


def normalize_vr_ov_levels(levels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(level.strip().upper() for level in levels if level.strip())
    if not normalized:
        raise ValueError("VR-OV curriculum must include at least one level.")
    unknown = [level for level in normalized if level not in VR_OV_LEVEL_ORDER]
    if unknown:
        raise ValueError(f"Unsupported VR-OV curriculum levels: {unknown}")
    return normalized


def resolve_vr_ov_curriculum_state(args: argparse.Namespace, max_iter: int) -> VROVCurriculumState:
    levels = normalize_vr_ov_levels(getattr(args, "curriculum_levels", VR_OV_LEVEL_ORDER))
    requested_interval = int(getattr(args, "curriculum_switch_interval", 0) or 0)
    switch_interval = requested_interval
    if switch_interval <= 0:
        switch_interval = max(1, max_iter // max(len(levels), 1))
    return VROVCurriculumState(levels=levels, switch_interval=switch_interval)


def classify_vr_ov_curriculum_level(
    query_struct: dict[str, Any] | None,
    slice_tag: str | None = None,
) -> str:
    if query_struct is not None:
        if query_struct.get("exists") is False or query_struct.get("negatives"):
            return "L4"
        if query_struct.get("relations") or query_struct.get("actions"):
            return "L3"
        if query_struct.get("attributes"):
            return "L2"
        return "L1"

    normalized_slice = (slice_tag or "").strip().lower()
    if normalized_slice == "no_target":
        return "L4"
    if normalized_slice == "relation_action":
        return "L3"
    if normalized_slice == "attribute":
        return "L2"
    return "L1"


def get_vr_ov_phase_trainable_prefixes(phase: str | None) -> tuple[str, ...]:
    if phase is None:
        return ()
    normalized = phase.strip().lower()
    if normalized in ("2", "3"):
        return ()
    if normalized not in _VR_OV_PHASE_TRAINABLE_PREFIXES:
        raise ValueError(
            f"Unsupported VR-OV pretraining phase '{phase}'. Expected one of {sorted(_VR_OV_PHASE_TRAINABLE_PREFIXES)}."
        )
    return _VR_OV_PHASE_TRAINABLE_PREFIXES[normalized]


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return cast(torch.nn.Module, getattr(model, "module", model))


def apply_vr_ov_phase_freezing(
    model: torch.nn.Module,
    phase: str | None,
) -> dict[str, object]:
    prefixes = get_vr_ov_phase_trainable_prefixes(phase)
    root_model = _unwrap_model(model)
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    if not prefixes:
        trainable_names = [name for name, param in root_model.named_parameters() if param.requires_grad]
        return {
            "phase": phase,
            "trainable_prefixes": prefixes,
            "trainable_names": trainable_names,
            "frozen_names": frozen_names,
        }

    for name, param in root_model.named_parameters():
        should_train = any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        param.requires_grad = should_train
        if should_train:
            trainable_names.append(name)
        else:
            frozen_names.append(name)

    if not trainable_names:
        raise RuntimeError(
            f"VR-OV phase '{phase}' did not match any trainable parameters for prefixes {prefixes}."
        )
    return {
        "phase": phase,
        "trainable_prefixes": prefixes,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
    }


def _clone_query_graph_with_dropout(
    query_graph: object,
    *,
    dropout_p: float,
    generator: torch.Generator,
) -> tuple[object, int, int]:
    from model.vr_ov_types import QueryGraph

    if not isinstance(query_graph, QueryGraph):
        return query_graph, 0, 0
    node_count = len(query_graph.nodes)
    if node_count == 0 or dropout_p <= 0.0:
        return query_graph, node_count, node_count

    keep_mask = torch.rand(node_count, generator=generator) >= dropout_p
    if not bool(keep_mask.any()):
        forced_index = int(torch.randint(node_count, (1,), generator=generator).item())
        keep_mask[forced_index] = True
    keep_indices = keep_mask.nonzero(as_tuple=False).flatten().tolist()
    keep_set = set(keep_indices)
    dropped_nodes = [
        node if keep_mask[index] else torch.zeros_like(node)
        for index, node in enumerate(query_graph.nodes)
    ]

    edge_index = query_graph.edges
    if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
        keep_edge_mask = torch.tensor(
            [
                int(edge_index[0, edge_pos].item()) in keep_set
                and int(edge_index[1, edge_pos].item()) in keep_set
                for edge_pos in range(edge_index.shape[1])
            ],
            dtype=torch.bool,
            device=edge_index.device,
        )
        edge_index = edge_index[:, keep_edge_mask]

    return (
        QueryGraph(
            nodes=dropped_nodes,
            edges=edge_index,
            node_types=list(query_graph.node_types),
        ),
        len(keep_indices),
        node_count,
    )


def filter_batch_by_vr_ov_curriculum(
    batch: Sequence[dict[str, object]],
    *,
    current_level: str,
) -> tuple[list[dict[str, object]], VROVBatchMetrics]:
    level_rank = {level: index for index, level in enumerate(VR_OV_LEVEL_ORDER)}
    current_rank = level_rank[current_level]
    filtered_batch: list[dict[str, object]] = []
    batch_metrics = VROVBatchMetrics(current_level=current_level)

    for batch_input in batch:
        prompt_list = list(cast(list[object], batch_input.get("prompt", [])))
        if not prompt_list:
            filtered_batch.append(dict(batch_input))
            continue

        query_structs = cast(list[dict[str, Any]] | None, batch_input.get("query_struct"))
        slice_tags = cast(list[str] | None, batch_input.get("slice_tags"))
        prompt_levels = [
            classify_vr_ov_curriculum_level(
                query_structs[index] if query_structs is not None else None,
                slice_tags[index] if slice_tags is not None else None,
            )
            for index in range(len(prompt_list))
        ]
        candidate_indices = [
            index
            for index, level in enumerate(prompt_levels)
            if level_rank[level] <= current_rank
        ]
        batch_metrics.total_prompts += len(prompt_list)

        if not candidate_indices:
            fallback_index = min(
                range(len(prompt_levels)), key=lambda index: level_rank[prompt_levels[index]]
            )
            candidate_indices = [fallback_index]
            batch_metrics.forced_prompt_keeps += 1

        batch_metrics.kept_prompts += len(candidate_indices)
        updated_input = dict(batch_input)
        for field_name in _VR_OV_FILTERABLE_BATCH_FIELDS:
            field_value = batch_input.get(field_name)
            if isinstance(field_value, list) and len(field_value) == len(prompt_list):
                updated_input[field_name] = [field_value[index] for index in candidate_indices]
        filtered_batch.append(updated_input)

    return filtered_batch, batch_metrics


class VROVRuntimeController:
    def __init__(self, args: argparse.Namespace, *, max_iter: int) -> None:
        self.phase = getattr(args, "phase", None)
        self.curriculum = resolve_vr_ov_curriculum_state(args, max_iter)
        self.query_dropout_p = float(getattr(args, "query_dropout_p", 0.0))
        self.ema_decay = float(getattr(args, "ema_decay", 0.0))
        self.metrics = VROVRuntimeMetrics()
        self._dropout_generator = torch.Generator()
        self._dropout_generator.manual_seed(int(getattr(args, "vr_ov_seed", 0)))

    def prepare_batch(
        self,
        batch: Sequence[dict[str, object]],
        *,
        iteration: int,
    ) -> list[dict[str, object]]:
        current_level = self.curriculum.level_for_iteration(iteration)
        prepared_batch, batch_metrics = filter_batch_by_vr_ov_curriculum(
            batch,
            current_level=current_level,
        )
        self.metrics.batch_count += 1
        self.metrics.total_prompts += batch_metrics.total_prompts
        self.metrics.kept_prompts += batch_metrics.kept_prompts
        self.metrics.forced_prompt_keeps += batch_metrics.forced_prompt_keeps
        self.metrics.last_batch = batch_metrics
        return prepared_batch

    def record_dropout(self, *, kept_nodes: int, total_nodes: int) -> None:
        self.metrics.dropout_events += 1
        self.metrics.query_nodes_kept += kept_nodes
        self.metrics.query_nodes_seen += total_nodes

    def record_total_loss(self, total_loss: float) -> None:
        self.metrics.loss_history.append(total_loss)

    def progress_metrics(self) -> dict[str, float | str]:
        current_level = self.curriculum.level_for_iteration(
            max(self.metrics.batch_count - 1, 0)
        )
        prompt_keep_ratio = (
            self.metrics.kept_prompts / self.metrics.total_prompts
            if self.metrics.total_prompts
            else 1.0
        )
        mean_query_keep = (
            self.metrics.query_nodes_kept / self.metrics.dropout_events
            if self.metrics.dropout_events
            else 0.0
        )
        return {
            "phase": self.phase or "joint",
            "curriculum_level": current_level,
            "prompt_keep_ratio": prompt_keep_ratio,
            "forced_prompt_keeps": float(self.metrics.forced_prompt_keeps),
            "query_dropout_mean_kept": mean_query_keep,
        }

    def final_metrics_payload(
        self,
        *,
        final_losses: dict[str, float],
        freeze_summary: dict[str, object],
        ema_state: ModelEmaState | None,
    ) -> dict[str, object]:
        return {
            **final_losses,
            "loss_history": self.metrics.loss_history,
            "vr_ov": {
                "phase": self.phase,
                "curriculum_levels": list(self.curriculum.levels),
                "curriculum_switch_interval": self.curriculum.switch_interval,
                "query_dropout_p": self.query_dropout_p,
                "query_dropout_mean_kept": (
                    self.metrics.query_nodes_kept / self.metrics.dropout_events
                    if self.metrics.dropout_events
                    else 0.0
                ),
                "prompt_keep_ratio": (
                    self.metrics.kept_prompts / self.metrics.total_prompts
                    if self.metrics.total_prompts
                    else 1.0
                ),
                "forced_prompt_keeps": self.metrics.forced_prompt_keeps,
                "ema_decay": ema_state.decay if ema_state is not None else None,
                "ema_updates": ema_state.num_updates if ema_state is not None else 0,
                "freeze_summary": {
                    **freeze_summary,
                    "trainable_param_count": len(cast(list[str], freeze_summary["trainable_names"])),
                    "frozen_param_count": len(cast(list[str], freeze_summary["frozen_names"])),
                },
            },
        }


def install_vr_ov_query_dropout(
    model: torch.nn.Module,
    controller: VROVRuntimeController,
) -> None:
    root_model = _unwrap_model(model)
    query_parser = getattr(root_model, "vr_ov_query_parser", None)
    if query_parser is None or controller.query_dropout_p <= 0.0:
        return
    if getattr(query_parser, "_vr_ov_dropout_installed", False):
        return

    original_forward = query_parser.forward

    def _wrapped_forward(*args: object, **kwargs: object) -> object:
        query_graph = original_forward(*args, **kwargs)
        dropped_query_graph, kept_nodes, total_nodes = _clone_query_graph_with_dropout(
            query_graph,
            dropout_p=controller.query_dropout_p,
            generator=controller._dropout_generator,
        )
        if total_nodes > 0:
            controller.record_dropout(kept_nodes=kept_nodes, total_nodes=total_nodes)
        return dropped_query_graph

    query_parser.forward = _wrapped_forward  # type: ignore[method-assign]
    query_parser._vr_ov_dropout_installed = True  # type: ignore[attr-defined]


def _format_vr_ov_progress_suffix(progress_metrics: dict[str, float | str]) -> str:
    return (
        f" phase={progress_metrics['phase']}"
        f" curriculum={progress_metrics['curriculum_level']}"
        f" prompt_keep={float(progress_metrics['prompt_keep_ratio']):.3f}"
        f" qdrop_keep={float(progress_metrics['query_dropout_mean_kept']):.3f}"
        f" forced_keeps={int(progress_metrics['forced_prompt_keeps'])}"
    )


def _build_checkpointer(
    deps: dict[str, object],
    model: torch.nn.Module,
    *,
    output_dir: str,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    ema_state: ModelEmaState | None,
) -> object:
    if ema_state is None:
        return deps["DetectionCheckpointer"](
            model,
            save_dir=output_dir,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    return deps["DetectionCheckpointer"](
        model,
        save_dir=output_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        ema_state=ema_state,
    )


def _maybe_create_vr_ov_controller(
    args: argparse.Namespace,
    *,
    max_iter: int,
) -> VROVRuntimeController | None:
    if not bool(getattr(args, "vr_ov_enabled", False)):
        return None
    return VROVRuntimeController(args, max_iter=max_iter)


def _save_loss_plot(output_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(_loss_history) < 2:
        return
    iters, losses = zip(*_loss_history)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iters, losses, linewidth=0.5, color="steelblue", alpha=0.8)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.set_title("Training Loss Curve", fontsize=13)
    ax.grid(True, alpha=0.3)
    if len(losses) > 100:
        from numpy import convolve, ones
        window = min(50, len(losses) // 4)
        smoothed = convolve(losses, ones(window) / window, mode="valid")
        ax.plot(iters[window - 1 :], smoothed, linewidth=1.5, color="darkorange", label=f"SMA({window})")
        ax.legend(fontsize=10)
    fig.tight_layout()
    save_path = Path(output_dir) / "loss_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _format_dataset_names(dataset_names: object) -> str:
    if isinstance(dataset_names, tuple):
        return ",".join(str(name) for name in dataset_names)
    return str(dataset_names)


def _is_main_process(deps: dict[str, object]) -> bool:
    comm = deps.get("comm")
    if comm is None:
        return True
    return bool(comm.is_main_process())


def _get_log_period(checkpoint_period: int) -> int:
    if checkpoint_period > 1:
        return min(checkpoint_period, 500)
    return DEFAULT_LOG_PERIOD


def _should_log_progress(
    iteration: int, start_iter: int, max_iter: int, log_period: int
) -> bool:
    completed_iteration = iteration + 1
    return (
        iteration == start_iter
        or completed_iteration % log_period == 0
        or iteration == max_iter - 1
    )


def _should_save_periodic_checkpoint(
    iteration: int, max_iter: int, checkpoint_period: int
) -> bool:
    if checkpoint_period <= 0 or iteration == max_iter - 1:
        return False
    return (iteration + 1) % checkpoint_period == 0


def _should_run_periodic_eval(iteration: int, max_iter: int, eval_period: int) -> bool:
    if eval_period <= 0:
        return False
    completed_iteration = iteration + 1
    return completed_iteration % eval_period == 0 or iteration == max_iter - 1


def _has_eval_dataset(cfg: object) -> bool:
    dataset_names = getattr(cfg.DATASETS, "TEST", ())
    return bool(dataset_names)


def _cleanup_after_evaluation() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _run_periodic_evaluation(
    cfg: object,
    model: object,
    deps: dict[str, object],
    logger: logging.Logger,
    *,
    iteration: int,
    max_iter: int,
) -> dict[str, object]:
    from reasonseg.runtime.common import get_inference_output_dir
    from reasonseg.runtime.eval import _format_metrics, run_evaluation

    completed_iteration = iteration + 1
    inference_dir = (
        get_inference_output_dir(cfg.OUTPUT_DIR) / f"iter_{completed_iteration:07d}"
    )
    dataset_name = cfg.DATASETS.TEST[0]
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        results = run_evaluation(model, cfg, deps=deps, output_dir=inference_dir)
    finally:
        if was_training:
            model.train()
        _cleanup_after_evaluation()

    if _is_main_process(deps):
        logger.info(
            "eval progress iter=%d/%d dataset=%s inference_dir=%s metrics=%s",
            completed_iteration,
            max_iter,
            dataset_name,
            inference_dir,
            _format_metrics(results),
        )
    return results


def _log_train_configuration(
    logger: logging.Logger,
    cfg: object,
    *,
    checkpoint_path: str | None,
    start_iter: int,
    resume: bool,
) -> None:
    logger.info(
        "train config datasets_train=%s datasets_test=%s batch_size=%d base_lr=%.8f max_iter=%d checkpoint_period=%d eval_period=%d output_dir=%s",
        _format_dataset_names(cfg.DATASETS.TRAIN),
        _format_dataset_names(cfg.DATASETS.TEST),
        int(cfg.SOLVER.IMS_PER_BATCH),
        float(cfg.SOLVER.BASE_LR),
        int(cfg.SOLVER.MAX_ITER),
        int(cfg.SOLVER.CHECKPOINT_PERIOD),
        int(cfg.TEST.EVAL_PERIOD),
        cfg.OUTPUT_DIR,
    )
    logger.info(
        "train init resume=%s start_iter=%d weights=%s",
        str(bool(resume)).lower(),
        start_iter,
        checkpoint_path if checkpoint_path is not None else "none",
    )


def _create_train_progress_bar(*, enabled: bool, start_iter: int, max_iter: int) -> Any:
    if not enabled:
        return None
    return tqdm(
        total=max_iter - start_iter,
        desc="train",
        dynamic_ncols=True,
        leave=True,
    )


def _update_train_progress_bar(
    progress_bar: Any,
    *,
    total_loss: float,
) -> None:
    if progress_bar is None:
        return
    progress_bar.update(1)
    progress_bar.set_postfix({"loss": f"{total_loss:.4f}"})


def _worker(args: argparse.Namespace) -> dict[str, float]:
    from reasonseg.runtime.common import (
        _import_runtime_deps,
        build_optimizer,
        build_refcoco_train_loader,
        get_train_metrics_path,
        maybe_wrap_model,
        resolve_train_checkpoint_path,
        setup_cfg,
        setup_runtime_logging,
        write_json_artifact,
        write_last_checkpoint,
    )

    deps = _import_runtime_deps()
    cfg = setup_cfg(args)
    setup_runtime_logging(cfg, args, eval_only=False)

    model = deps["build_model"](cfg)
    model = maybe_wrap_model(model)
    vr_ov_controller = _maybe_create_vr_ov_controller(args, max_iter=int(cfg.SOLVER.MAX_ITER))
    freeze_summary = {
        "phase": getattr(args, "phase", None),
        "trainable_prefixes": (),
        "trainable_names": [name for name, param in _unwrap_model(model).named_parameters() if param.requires_grad],
        "frozen_names": [],
    }
    if vr_ov_controller is not None:
        freeze_summary = apply_vr_ov_phase_freezing(model, getattr(args, "phase", None))
        install_vr_ov_query_dropout(model, vr_ov_controller)
    optimizer = build_optimizer(cfg, model)
    scheduler = deps["build_lr_scheduler"](cfg, optimizer)
    ema_state = None
    if vr_ov_controller is not None and vr_ov_controller.ema_decay > 0.0:
        ema_state = ModelEmaState(model, decay=vr_ov_controller.ema_decay)
    checkpointer = _build_checkpointer(
        deps,
        model,
        output_dir=cfg.OUTPUT_DIR,
        optimizer=optimizer,
        scheduler=scheduler,
        ema_state=ema_state,
    )
    logger = logging.getLogger("reasonseg")
    checkpoint_path = resolve_train_checkpoint_path(args, cfg.OUTPUT_DIR)
    checkpoint_state = cast(Any, checkpointer).resume_or_load(checkpoint_path, resume=False)
    start_iter = 0
    if args.resume:
        start_iter = int(checkpoint_state.get("iteration", -1)) + 1

    _log_train_configuration(
        logger,
        cfg,
        checkpoint_path=checkpoint_path,
        start_iter=start_iter,
        resume=args.resume,
    )
    data_loader = build_refcoco_train_loader(cfg)
    data_iterator = iter(data_loader)
    final_losses: dict[str, float] = {}
    checkpoint_period = int(cfg.SOLVER.CHECKPOINT_PERIOD)
    eval_period = int(cfg.TEST.EVAL_PERIOD)
    log_period = _get_log_period(checkpoint_period)
    is_main_process = _is_main_process(deps)
    model.train()
    progress_bar = _create_train_progress_bar(
        enabled=is_main_process,
        start_iter=start_iter,
        max_iter=cfg.SOLVER.MAX_ITER,
    )
    _loss_history.clear()
    logger.info(
        "train loop start start_iter=%d max_iter=%d checkpoint_period=%d log_period=%d",
        start_iter,
        cfg.SOLVER.MAX_ITER,
        checkpoint_period,
        log_period,
    )
    try:
        for iteration in range(start_iter, cfg.SOLVER.MAX_ITER):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                batch = next(data_iterator)
            if vr_ov_controller is not None:
                batch = vr_ov_controller.prepare_batch(
                    cast(Sequence[dict[str, object]], batch),
                    iteration=iteration,
                )
            loss_dict = model(batch)
            losses = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()
            if ema_state is not None:
                ema_state.update(model)

            if iteration > 0 and iteration % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            reduced_loss_dict = {
                name: float(value.detach().cpu())
                for name, value in deps["comm"].reduce_dict(loss_dict).items()
            }
            del loss_dict
            final_losses = reduced_loss_dict
            total_loss = sum(reduced_loss_dict.values())
            _update_train_progress_bar(progress_bar, total_loss=total_loss)
            _loss_history.append((iteration + 1, total_loss))
            if vr_ov_controller is not None:
                vr_ov_controller.record_total_loss(total_loss)

            if is_main_process and _should_log_progress(
                iteration, start_iter, cfg.SOLVER.MAX_ITER, log_period
            ):
                completed_iteration = iteration + 1
                loss_terms = ", ".join(
                    f"{name}={value:.6f}"
                    for name, value in sorted(reduced_loss_dict.items())
                )
                _cuda_mem = torch.cuda.max_memory_allocated() / 1024**3
                _cuda_res = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    "train progress iter=%d/%d lr=%.8f total_loss=%.6f gpu_mem=%.1f/%.1fGB%s%s",
                    completed_iteration,
                    cfg.SOLVER.MAX_ITER,
                    float(optimizer.param_groups[0]["lr"]),
                    total_loss,
                    _cuda_mem,
                    _cuda_res,
                    f" ({loss_terms})" if loss_terms else "",
                    (
                        _format_vr_ov_progress_suffix(vr_ov_controller.progress_metrics())
                        if vr_ov_controller is not None
                        else ""
                    ),
                )
                _save_loss_plot(cfg.OUTPUT_DIR)

            if is_main_process and _should_save_periodic_checkpoint(
                iteration, cfg.SOLVER.MAX_ITER, checkpoint_period
            ):
                checkpoint_name = f"model_{iteration:07d}"
                cast(Any, checkpointer).save(checkpoint_name, iteration=iteration)
                checkpoint_output_path = write_last_checkpoint(
                    cfg.OUTPUT_DIR, Path(cfg.OUTPUT_DIR) / f"{checkpoint_name}.pth"
                )
                logger.info(
                    "checkpoint saved iter=%d/%d path=%s last_checkpoint=%s",
                    iteration + 1,
                    cfg.SOLVER.MAX_ITER,
                    Path(cfg.OUTPUT_DIR) / f"{checkpoint_name}.pth",
                    checkpoint_output_path,
                )
                _save_loss_plot(cfg.OUTPUT_DIR)

            if _has_eval_dataset(cfg) and _should_run_periodic_eval(
                iteration, cfg.SOLVER.MAX_ITER, eval_period
            ):
                _run_periodic_evaluation(
                    cfg,
                    model,
                    deps,
                    logger,
                    iteration=iteration,
                    max_iter=cfg.SOLVER.MAX_ITER,
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if is_main_process:
        cast(Any, checkpointer).save(
            "model_final",
            iteration=max(cfg.SOLVER.MAX_ITER - 1, start_iter - 1),
        )
        final_checkpoint_path = Path(cfg.OUTPUT_DIR) / "model_final.pth"
        if ema_state is not None:
            ema_state.save_ema_checkpoint(
                checkpointer,
                model,
                checkpoint_name="model_final_ema",
                iteration=max(cfg.SOLVER.MAX_ITER - 1, start_iter - 1),
            )
        checkpoint_output_path = write_last_checkpoint(
            cfg.OUTPUT_DIR, final_checkpoint_path
        )
        metrics_path = get_train_metrics_path(cfg.OUTPUT_DIR)
        metrics_payload: dict[str, object] = final_losses
        if vr_ov_controller is not None:
            metrics_payload = vr_ov_controller.final_metrics_payload(
                final_losses=final_losses,
                freeze_summary=freeze_summary,
                ema_state=ema_state,
            )
        write_json_artifact(metrics_path, metrics_payload)
        _save_loss_plot(cfg.OUTPUT_DIR)
        logger.info(
            "train complete final_checkpoint=%s last_checkpoint=%s metrics_path=%s",
            final_checkpoint_path,
            checkpoint_output_path,
            metrics_path,
        )
    return final_losses


def main(argv: argparse.Namespace | Sequence[str] | None = None) -> int:
    if argv is None or isinstance(argv, Sequence):
        raise RuntimeError("reasonseg.runtime.train.main expects parsed CLI arguments.")
    launch_train_main = __import__(
        "reasonseg.runtime.common", fromlist=["launch_train_main"]
    ).launch_train_main
    launch_train_main(_worker, argv)
    return 0
