from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def batch_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    hw = inputs.shape[1]
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs,
        torch.ones_like(inputs),
        reduction="none",
    )
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(
        inputs,
        torch.zeros_like(inputs),
        reduction="none",
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm",
        focal_neg,
        1 - targets,
    )
    return loss / hw


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        cost_class_prediction: float = 1,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_class_prediction = cost_class_prediction
        assert (
            cost_class != 0
            or cost_mask != 0
            or cost_dice != 0
            or cost_class_prediction != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        del batch_size

        indices = []
        for batch_index in range(outputs["pred_logits"].shape[0]):
            if (
                targets[batch_index].get("is_negative", False)
                or len(targets[batch_index]["masks"]) == 0
            ):
                indices.append(
                    (
                        torch.tensor([], dtype=torch.int64),
                        torch.tensor([], dtype=torch.int64),
                    )
                )
                continue

            out_prob = outputs["pred_logits"][batch_index]
            out_mask = outputs["pred_masks"][batch_index]
            tgt_mask = targets[batch_index]["masks"].to(out_mask)

            cost_class = -out_prob[:, 0].unsqueeze(1)

            if "pred_classes" in outputs and self.cost_class_prediction > 0:
                out_class = outputs["pred_classes"][batch_index]
                tgt_class = targets[batch_index]["classes"]
                out_class_prob = F.softmax(out_class, dim=1)
                alpha = 0.25
                gamma = 2.0

                num_targets = len(tgt_class)
                if num_targets == 0:
                    cost_class_pred = torch.zeros(
                        (num_queries, 1),
                        device=out_class.device,
                    )
                else:
                    cost_class_pred = torch.zeros(
                        (num_queries, num_targets),
                        device=out_class.device,
                    )
                    for target_index, class_id in enumerate(tgt_class):
                        probability = out_class_prob[:, class_id]
                        cost_class_pred[:, target_index] = (
                            alpha
                            * ((1 - probability) ** gamma)
                            * (-(probability + 1e-8).log())
                        )
            else:
                cost_class_pred = torch.zeros_like(cost_class)

            tgt_mask = F.interpolate(
                tgt_mask[:, None],
                size=out_mask.shape[-2:],
                mode="nearest",
            )
            out_mask = out_mask.flatten(1)
            tgt_mask = tgt_mask[:, 0].flatten(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)
                cost_mask[cost_mask.isnan()] = 1e6
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
                cost_dice[cost_dice.isnan()] = 1e6

            num_targets = cost_mask.shape[1]
            if cost_class.shape[1] != num_targets:
                if cost_class.shape[1] == 1:
                    cost_class = cost_class.expand(-1, num_targets)
                else:
                    cost_class = torch.zeros(
                        (num_queries, num_targets),
                        device=cost_mask.device,
                    )

            if cost_class_pred.shape[1] != num_targets:
                if cost_class_pred.shape[1] == 1:
                    cost_class_pred = cost_class_pred.expand(-1, num_targets)
                else:
                    old_cost = cost_class_pred.clone()
                    cost_class_pred = torch.zeros(
                        (num_queries, num_targets),
                        device=cost_mask.device,
                    )
                    min_targets = min(old_cost.shape[1], num_targets)
                    if min_targets > 0:
                        cost_class_pred[:, :min_targets] = old_cost[:, :min_targets]

            cost = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_class_prediction * cost_class_pred
            )
            indices.append(linear_sum_assignment(cost.reshape(num_queries, -1).cpu()))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self) -> str:
        head = f"Matcher {self.__class__.__name__}"
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
            f"cost_class_prediction: {self.cost_class_prediction}",
        ]
        repr_indent = 4
        lines = [head] + [" " * repr_indent + line for line in body]
        return "\n".join(lines)
