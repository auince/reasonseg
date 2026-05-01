# pyright: reportMissingImports=false, reportExplicitAny=false, reportAny=false
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


def _format_metrics(metrics: dict[str, object]) -> str:
    pairs: list[str] = []
    for group_name, group_values in sorted(metrics.items()):
        if isinstance(group_values, dict):
            for metric_name, metric_value in sorted(group_values.items()):
                if isinstance(metric_value, (int, float)):
                    pairs.append(f"{metric_name}={metric_value:.6f}")
                else:
                    pairs.append(
                        f"{metric_name}={json.dumps(metric_value, sort_keys=True)}"
                    )
        else:
            pairs.append(f"{group_name}={json.dumps(group_values, sort_keys=True)}")
    return ", ".join(pairs)


def _is_main_process(comm: Any) -> bool:
    if hasattr(comm, "is_main_process"):
        return bool(comm.is_main_process())
    if hasattr(comm, "get_rank"):
        return int(comm.get_rank()) == 0
    return True


def _create_eval_progress_bar(*, loader: Any, dataset_name: str, enabled: bool) -> Any:
    if not enabled:
        return None
    total = len(loader) if hasattr(loader, "__len__") else None
    return tqdm(
        total=total, desc=f"eval:{dataset_name}", dynamic_ncols=True, leave=True
    )


def _update_eval_progress(
    progress_bar: Any,
    progress_metrics: dict[str, float] | None,
) -> None:
    if progress_bar is None:
        return
    progress_bar.update(1)
    if progress_metrics is None:
        return
    progress_bar.set_postfix(
        {
            "oIoU": f"{progress_metrics['oIoU']:.2f}",
            "mIoU": f"{progress_metrics['mIoU']:.2f}",
            "Prec@50": f"{progress_metrics['Prec@50']:.2f}",
            "Prec@75": f"{progress_metrics['Prec@75']:.2f}",
        }
    )


def _save_eval_visualization(
    vis_samples: list[dict[str, Any]],
    output_dir: Path,
    eval_iter: int,
) -> None:
    import random

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image
    from matplotlib.patches import Polygon

    candidates = [s for s in vis_samples if s["gt_area"] > 0]
    if not candidates:
        return
    num_vis = min(4, len(candidates))
    selected = random.sample(candidates, num_vis) if len(candidates) >= num_vis else candidates

    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(selected):
        img_path = sample["file_name"]
        pred_mask = sample["pred_mask"]
        gt_mask = sample["gt_mask"]

        try:
            original = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            continue

        h, w = original.shape[:2]
        if pred_mask.shape[:2] != (h, w):
            from torch.nn import functional as F
            import torch as _torch
            pm = _torch.as_tensor(pred_mask, dtype=_torch.float32)
            pred_mask = F.interpolate(pm[None, None], (h, w), mode="nearest")[0, 0].numpy()
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.max(axis=0) if gt_mask.shape[0] > 0 else np.zeros((h, w), dtype=bool)
        if gt_mask.shape[:2] != (h, w):
            from torch.nn import functional as F
            import torch as _torch
            gm = _torch.as_tensor(gt_mask, dtype=_torch.float32)
            gt_mask = F.interpolate(gm[None, None], (h, w), mode="nearest")[0, 0].numpy()

        pred_overlay = _overlay_mask(original.copy(), pred_mask.astype(bool), color=(0, 255, 0))
        gt_overlay = _overlay_mask(original.copy(), gt_mask.astype(bool), color=(255, 0, 0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")
        axes[1].imshow(pred_overlay)
        axes[1].set_title("Prediction", fontsize=10)
        axes[1].axis("off")
        axes[2].imshow(gt_overlay)
        axes[2].set_title("Ground Truth", fontsize=10)
        axes[2].axis("off")

        green_patch = mpatches.Patch(color="green", alpha=0.4, label="Pred")
        red_patch = mpatches.Patch(color="red", alpha=0.4, label="GT")
        fig.legend(handles=[green_patch, red_patch], loc="lower center", ncol=2, fontsize=9)
        fig.suptitle(f"Eval iter={eval_iter}  sample {idx+1}", fontsize=12)
        plt.tight_layout()
        save_path = vis_dir / f"iter_{eval_iter:07d}_sample{idx+1}.png"
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    logger = logging.getLogger("reasonseg")
    logger.info("eval vis saved iter=%d count=%d dir=%s", eval_iter, len(selected), vis_dir)


def _overlay_mask(
    image: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float = 0.4
) -> np.ndarray:
    overlay = image.copy()
    for c in range(3):
        overlay[mask, c] = (overlay[mask, c] * (1 - alpha) + color[c] * alpha).astype(np.uint8)
    return overlay


def run_evaluation(
    model: Any,
    cfg: Any,
    *,
    deps: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    from reasonseg.evaluation.grounding import GroundingEvaluator
    from reasonseg.runtime.common import (
        _import_runtime_deps,
        build_refcoco_test_loader,
        get_inference_output_dir,
    )

    runtime_deps = _import_runtime_deps() if deps is None else deps
    dataset_name = cfg.DATASETS.TEST[0]
    inference_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_inference_output_dir(cfg.OUTPUT_DIR)
    )
    loader = build_refcoco_test_loader(cfg, dataset_name)
    evaluator = GroundingEvaluator(
        dataset_name,
        output_dir=inference_dir,
        distributed=runtime_deps["comm"].get_world_size() > 1,
    )
    progress_bar = _create_eval_progress_bar(
        loader=loader,
        dataset_name=dataset_name,
        enabled=_is_main_process(runtime_deps["comm"]),
    )
    was_training = bool(getattr(model, "training", False))
    model.eval()
    vis_samples: list[dict[str, Any]] = []
    vr_ov_artifacts: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for inputs in loader:
                outputs = model(inputs)
                evaluator.process(inputs, outputs)
                for out in outputs:
                    if "vr_ov_compositional" in out:
                        vr_ov_artifacts.append(out["vr_ov_compositional"])
                if len(vis_samples) < 20:
                    for inp, out in zip(inputs, outputs):
                        pred_masks = (out["grounding_mask"].sigmoid() > 0.5).cpu().numpy()
                        gt_masks = inp["groundings"]["masks"].bool().numpy()
                        for p_idx, (pm, gm) in enumerate(zip(pred_masks, gt_masks)):
                            if len(vis_samples) >= 20:
                                break
                            gt_area = float(gm.sum())
                            vis_samples.append({
                                "file_name": inp.get("file_name", ""),
                                "pred_mask": pm,
                                "gt_mask": gm,
                                "gt_area": gt_area,
                                "prompt_idx": p_idx,
                            })
                progress_metrics = evaluator.progress_metrics()
                _update_eval_progress(progress_bar, progress_metrics)
        results = evaluator.evaluate()
        if results is not None and inference_dir is not None and _is_main_process(runtime_deps["comm"]):
            eval_iter = int(Path(inference_dir).name.split("_")[-1]) if "_" in Path(inference_dir).name else 0
            _save_eval_visualization(vis_samples, inference_dir, eval_iter)
            if vr_ov_artifacts:
                (inference_dir / "vr_ov_compositional.json").write_text(
                    json.dumps(vr_ov_artifacts, indent=2) + "\n",
                    encoding="utf-8",
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if was_training:
            model.train()
        del loader
        del evaluator
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if results is None:
        return {}
    return results


def _worker(args: argparse.Namespace) -> dict[str, object]:
    from reasonseg.runtime.common import (
        _import_runtime_deps,
        get_inference_output_dir,
        maybe_wrap_model,
        resolve_eval_checkpoint_path,
        setup_cfg,
        setup_runtime_logging,
    )

    deps = _import_runtime_deps()
    cfg = setup_cfg(args, eval_split=args.split, force_grounding_eval=True)
    setup_runtime_logging(cfg, args, eval_only=True)
    checkpoint_path = resolve_eval_checkpoint_path(args)
    logger = logging.getLogger("reasonseg")

    meta_arch = getattr(cfg.MODEL, "META_ARCHITECTURE", "")
    if meta_arch == "VR_OV":
        from model.vr_ov_config import validate_vr_ov_config
        try:
            validate_vr_ov_config(cfg)
        except ValueError as exc:
            logger.error("VR-OV eval config validation failed: %s", exc)
            raise

    model = deps["build_model"](cfg)
    model = maybe_wrap_model(model)
    deps["DetectionCheckpointer"](model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        checkpoint_path,
        resume=False,
    )

    dataset_name = cfg.DATASETS.TEST[0]
    logger.info(
        "eval config dataset=%s checkpoint=%s output_dir=%s inference_dir=%s",
        dataset_name,
        checkpoint_path,
        cfg.OUTPUT_DIR,
        get_inference_output_dir(cfg.OUTPUT_DIR),
    )
    results = run_evaluation(
        model,
        cfg,
        deps=deps,
        output_dir=get_inference_output_dir(cfg.OUTPUT_DIR),
    )
    logger.info(
        "eval complete dataset=%s metrics=%s", dataset_name, _format_metrics(results)
    )
    return results


def main(args: argparse.Namespace | Sequence[str] | None = None) -> int:
    if args is None or isinstance(args, Sequence):
        raise RuntimeError("reasonseg.runtime.eval.main expects parsed CLI arguments.")
    launch_main = __import__(
        "reasonseg.runtime.common", fromlist=["launch_main"]
    ).launch_main
    launch_main(_worker, args)
    return 0
