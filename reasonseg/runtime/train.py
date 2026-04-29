# pyright: reportMissingImports=false, reportExplicitAny=false, reportAny=false, reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


DEFAULT_LOG_PERIOD = 20

_loss_history: list[tuple[int, float]] = []


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
    optimizer = build_optimizer(cfg, model)
    scheduler = deps["build_lr_scheduler"](cfg, optimizer)
    checkpointer = deps["DetectionCheckpointer"](
        model,
        save_dir=cfg.OUTPUT_DIR,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    logger = logging.getLogger("reasonseg")
    checkpoint_path = resolve_train_checkpoint_path(args, cfg.OUTPUT_DIR)
    checkpoint_state = checkpointer.resume_or_load(checkpoint_path, resume=False)
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
            loss_dict = model(batch)
            losses = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

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
                    "train progress iter=%d/%d lr=%.8f total_loss=%.6f gpu_mem=%.1f/%.1fGB%s",
                    completed_iteration,
                    cfg.SOLVER.MAX_ITER,
                    float(optimizer.param_groups[0]["lr"]),
                    total_loss,
                    _cuda_mem,
                    _cuda_res,
                    f" ({loss_terms})" if loss_terms else "",
                )
                _save_loss_plot(cfg.OUTPUT_DIR)

            if is_main_process and _should_save_periodic_checkpoint(
                iteration, cfg.SOLVER.MAX_ITER, checkpoint_period
            ):
                checkpoint_name = f"model_{iteration:07d}"
                checkpointer.save(checkpoint_name, iteration=iteration)
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
        checkpointer.save(
            "model_final",
            iteration=max(cfg.SOLVER.MAX_ITER - 1, start_iter - 1),
        )
        final_checkpoint_path = Path(cfg.OUTPUT_DIR) / "model_final.pth"
        checkpoint_output_path = write_last_checkpoint(
            cfg.OUTPUT_DIR, final_checkpoint_path
        )
        metrics_path = get_train_metrics_path(cfg.OUTPUT_DIR)
        write_json_artifact(metrics_path, final_losses)
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
