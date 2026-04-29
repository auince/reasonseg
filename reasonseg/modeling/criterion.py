# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._compat import get_world_size
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    smooth: int = 1,
) -> torch.Tensor:
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + smooth) / (denominator + smooth)
    return loss.sum() / num_masks


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.class_weight = weight_dict.get("loss_classes", 1.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loss_labels(self, outputs, targets, indices, num_masks):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        object_logits = src_logits[batch_idx, src_idx].squeeze(-1)
        object_logits = torch.clamp(object_logits, min=-100.0, max=100.0)
        object_loss = (
            (1 - object_logits).mean()
            if object_logits.numel() > 0
            else torch.tensor(0.0, device=src_logits.device)
        )

        mask = torch.ones_like(src_logits, dtype=torch.bool)
        mask[batch_idx, src_idx] = False
        non_object_logits = src_logits[mask].squeeze(-1)
        non_object_logits = torch.clamp(non_object_logits, min=-100.0, max=100.0)
        non_object_loss = (
            (non_object_logits * self.eos_coef).mean()
            if non_object_logits.numel() > 0
            else torch.tensor(0.0, device=src_logits.device)
        )

        loss_ce = object_loss + non_object_loss
        if torch.isnan(loss_ce) or torch.isinf(loss_ce):
            loss_ce = torch.tensor(0.0, device=src_logits.device)
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        masks = [target["masks"] for target in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        del valid
        target_masks = target_masks.to(src_masks)[tgt_idx]

        src_masks = F.interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1).view(src_masks.shape)

        return {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }

    def loss_classes(self, outputs, targets, indices, num_masks):
        if "pred_classes" not in outputs:
            return {"loss_classes": torch.as_tensor(0.0, device=self.device)}

        src_logits = outputs["pred_classes"]
        device = src_logits.device
        if len(targets) == 0 or all(
            len(target.get("classes", [])) == 0 for target in targets
        ):
            loss = F.cross_entropy(
                src_logits.flatten(0, 1),
                torch.zeros(
                    src_logits.shape[0] * src_logits.shape[1],
                    dtype=torch.long,
                    device=device,
                ),
                reduction="mean",
            )
            return {"loss_classes": loss * self.class_weight}

        focal_alpha = 0.25
        focal_gamma = 2.0
        loss = torch.tensor(0.0, device=device)
        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:
                continue
            batch_src_logits = src_logits[batch_index][src_idx]
            if "classes" not in targets[batch_index]:
                tgt_classes = torch.zeros(len(tgt_idx), dtype=torch.long, device=device)
            else:
                tgt_classes = targets[batch_index]["classes"][tgt_idx]
                if not isinstance(tgt_classes, torch.Tensor):
                    tgt_classes = torch.tensor(
                        tgt_classes, dtype=torch.long, device=device
                    )
                elif len(tgt_classes.shape) == 0:
                    tgt_classes = tgt_classes.unsqueeze(0)

            probs = F.softmax(batch_src_logits, dim=-1)
            p_t = probs.gather(1, tgt_classes.unsqueeze(1)).squeeze(1)
            loss_batch = -focal_alpha * (1 - p_t) ** focal_gamma * torch.log(p_t + 1e-8)
            loss += loss_batch.sum()
        if num_masks > 0:
            loss = loss / num_masks
        return {"loss_classes": loss * self.class_weight}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, index) for index, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(tgt, index) for index, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "classes": self.loss_classes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, *, reduce_num_masks: bool = True):
        outputs_without_aux = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        indices = self.matcher(outputs_without_aux, targets)
        num_masks = sum(len(target["labels"]) for target in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=outputs["pred_logits"].device,
        )
        if reduce_num_masks and is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1)

        losses = {}
        for loss_name in self.losses:
            losses.update(
                self.get_loss(loss_name, outputs, targets, indices, num_masks)
            )

        if "aux_outputs" in outputs:
            for index, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss_name in self.losses:
                    loss_dict = self.get_loss(
                        loss_name,
                        aux_outputs,
                        targets,
                        indices,
                        num_masks,
                    )
                    losses.update(
                        {f"{key}_{index}": value for key, value in loss_dict.items()}
                    )
        return losses
