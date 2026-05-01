# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUntypedBaseClass=false, reportUnannotatedClassAttribute=false, reportUnusedImport=false, reportExplicitAny=false, reportAny=false, reportConstantRedefinition=false, reportGeneralTypeIssues=false, reportMissingSuperCall=false, reportArgumentType=false, reportAttributeAccessIssue=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Final, cast

import torch

try:
    from detectron2.evaluation.evaluator import DatasetEvaluator
    from detectron2.utils.comm import all_gather, is_main_process, synchronize
except ModuleNotFoundError:

    class DatasetEvaluator:  # type: ignore[no-redef]
        pass

    def all_gather(value):
        return [value]

    def is_main_process() -> bool:
        return True

    def synchronize() -> None:
        return None

    detectron2_eval_available = False
else:
    detectron2_eval_available = True


GROUNDING_METRIC_FAMILY: Final[str] = "openworldsam_grounding"
GROUNDING_THRESHOLDS: Final[tuple[float, ...]] = (0.5, 0.6, 0.7, 0.8, 0.9)
GROUNDING_PROGRESS_THRESHOLDS: Final[tuple[float, ...]] = tuple(
    sorted({*GROUNDING_THRESHOLDS, 0.75})
)
_MORPH_KERNEL_SIZE = 3


def _clean_mask(mask: torch.Tensor) -> torch.Tensor:
    """Apply morphological closing (fill holes) then opening (remove noise)."""
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    # Closing: dilate then erode (fills small holes in the mask)
    mask = torch.nn.functional.max_pool2d(mask, _MORPH_KERNEL_SIZE, stride=1, padding=_MORPH_KERNEL_SIZE // 2)
    mask = -torch.nn.functional.max_pool2d(-mask, _MORPH_KERNEL_SIZE, stride=1, padding=_MORPH_KERNEL_SIZE // 2)
    # Opening: erode then dilate (removes small isolated noise)
    mask = -torch.nn.functional.max_pool2d(-mask, _MORPH_KERNEL_SIZE, stride=1, padding=_MORPH_KERNEL_SIZE // 2)
    mask = torch.nn.functional.max_pool2d(mask, _MORPH_KERNEL_SIZE, stride=1, padding=_MORPH_KERNEL_SIZE // 2)
    return (mask.squeeze() > 0.5).bool()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class GroundingAccumulator:
    def __init__(self) -> None:
        self.sum_intersection: float = 0.0
        self.sum_union: float = 0.0
        self.sum_iou: float = 0.0
        self.total: int = 0
        self.threshold_hits: dict[float, int] = {
            threshold: 0 for threshold in GROUNDING_PROGRESS_THRESHOLDS
        }

    def add(self, *, intersection: float, union: float) -> None:
        _require(union > 0, "Positive grounding predictions must have union > 0.")
        _require(
            0 <= intersection <= union,
            "Grounding prediction intersection must be within [0, union].",
        )
        iou = intersection / union
        self.sum_intersection += intersection
        self.sum_union += union
        self.sum_iou += iou
        self.total += 1
        for threshold in GROUNDING_PROGRESS_THRESHOLDS:
            if iou >= threshold:
                self.threshold_hits[threshold] += 1

    def metrics(self) -> dict[str, float]:
        _require(
            self.total > 0, "Grounding metrics require at least one positive entry."
        )
        results: dict[str, float] = {
            "grounding/cIoU": self.sum_intersection * 100.0 / self.sum_union,
            "grounding/mIoU": self.sum_iou * 100.0 / self.total,
        }
        for threshold in GROUNDING_THRESHOLDS:
            results[f"grounding/precision@{threshold:.1f}"] = (
                self.threshold_hits[threshold] * 100.0 / self.total
            )
        return results

    def overall_iou(self) -> float:
        _require(self.total > 0, "Overall IoU requires at least one positive entry.")
        return self.sum_intersection * 100.0 / self.sum_union

    def mean_iou(self) -> float:
        _require(self.total > 0, "Mean IoU requires at least one positive entry.")
        return self.sum_iou * 100.0 / self.total

    def precision_at(self, threshold: float) -> float:
        _require(self.total > 0, "Precision requires at least one positive entry.")
        _require(
            threshold in self.threshold_hits,
            f"Unsupported precision threshold: {threshold}.",
        )
        return self.threshold_hits[threshold] * 100.0 / self.total

    def state_dict(self) -> dict[str, object]:
        return {
            "sum_intersection": self.sum_intersection,
            "sum_union": self.sum_union,
            "sum_iou": self.sum_iou,
            "total": self.total,
            "threshold_hits": dict(self.threshold_hits),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> GroundingAccumulator:
        accumulator = cls()
        accumulator.sum_intersection = float(state["sum_intersection"])
        accumulator.sum_union = float(state["sum_union"])
        accumulator.sum_iou = float(state["sum_iou"])
        accumulator.total = int(state["total"])
        threshold_hits = cast(dict[float, int], state["threshold_hits"])
        accumulator.threshold_hits = {
            float(threshold): int(count) for threshold, count in threshold_hits.items()
        }
        return accumulator

    @classmethod
    def merge_state_dicts(cls, states: list[dict[str, object]]) -> GroundingAccumulator:
        merged = cls()
        for state in states:
            merged.sum_intersection += float(state["sum_intersection"])
            merged.sum_union += float(state["sum_union"])
            merged.sum_iou += float(state["sum_iou"])
            merged.total += int(state["total"])
            threshold_hits = cast(dict[float, int], state["threshold_hits"])
            for threshold, count in threshold_hits.items():
                merged.threshold_hits[float(threshold)] += int(count)
        return merged


class GroundingEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name: str,
        *,
        output_dir: str | Path | None = None,
        distributed: bool = True,
    ) -> None:
        if not detectron2_eval_available:
            raise ModuleNotFoundError(
                "detectron2 is required for GroundingEvaluator runtime execution."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = None if output_dir is None else Path(output_dir)
        self.reset()

    def reset(self) -> None:
        self._accumulator = GroundingAccumulator()
        self._predictions: list[dict[str, object]] = []

    def process(self, inputs, outputs) -> None:
        for input_entry, output_entry in zip(inputs, outputs):
            predicted_masks = (
                (output_entry["grounding_mask"].sigmoid() > 0.5).cpu()
            )
            predicted_scores = output_entry.get("grounding_scores")
            gt_masks = input_entry["groundings"]["masks"].bool()
            prompts = input_entry.get("prompt", [])
            text_groups = input_entry["groundings"].get("texts", [])
            image_id = int(input_entry.get("image_id", -1))

            for index, (pred_mask, gt_mask) in enumerate(
                zip(predicted_masks, gt_masks)
            ):
                pred_mask = _clean_mask(pred_mask)
                intersection = float((pred_mask & gt_mask).sum().item())
                union = float((pred_mask | gt_mask).sum().item())
                self._accumulator.add(intersection=intersection, union=union)
                prompt = prompts[index] if index < len(prompts) else None
                candidates = text_groups[index] if index < len(text_groups) else []
                score = None
                if predicted_scores is not None and index < len(predicted_scores):
                    score = float(predicted_scores[index].detach().cpu())
                self._predictions.append(
                    {
                        "dataset_name": self._dataset_name,
                        "image_id": image_id,
                        "prompt_index": index,
                        "prompt": prompt,
                        "candidate_prompts": list(candidates),
                        "score": score,
                        "intersection": intersection,
                        "union": union,
                    }
                )

    def evaluate(self) -> dict[str, dict[str, float]] | None:
        if self._distributed:
            synchronize()
            prediction_groups = all_gather(self._predictions)
            state_groups = all_gather(self._accumulator.state_dict())
            if not is_main_process():
                return None
            self._accumulator = GroundingAccumulator.merge_state_dicts(state_groups)
            self._predictions = [item for group in prediction_groups for item in group]

        metrics = self._accumulator.metrics()
        results = {"grounding": metrics}
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            (self._output_dir / "predictions.json").write_text(
                json.dumps(self._predictions, indent=2) + "\n",
                encoding="utf-8",
            )
            (self._output_dir / "metrics.json").write_text(
                json.dumps(results, indent=2) + "\n",
                encoding="utf-8",
            )
        return results

    def progress_metrics(self) -> dict[str, float] | None:
        if self._distributed:
            states = all_gather(self._accumulator.state_dict())
            if not is_main_process():
                return None
            accumulator = GroundingAccumulator.merge_state_dicts(states)
        else:
            accumulator = self._accumulator

        if accumulator.total <= 0:
            return None

        return {
            "oIoU": accumulator.overall_iou(),
            "mIoU": accumulator.mean_iou(),
            "Prec@50": accumulator.precision_at(0.5),
            "Prec@75": accumulator.precision_at(0.75),
        }
