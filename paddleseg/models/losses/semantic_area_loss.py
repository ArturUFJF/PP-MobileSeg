"""Composite loss for semantic segmentation with auxiliary area regression head."""

from typing import Dict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class SemanticAreaLoss(nn.Layer):
    """Combine cross-entropy for semantics with regression loss for area mask.

    Args:
        ce_weight (float): Weight applied to semantic cross-entropy component.
        area_weight (float): Weight applied to area regression component.
        area_loss (str): Type of regression loss. One of {"l1", "l2", "smooth_l1"}.
        ignore_index (int): Label value to ignore for semantic head.
        reduction (str): Reduction for regression loss ("mean" or "sum").
        smooth_l1_beta (float): Beta parameter when ``area_loss`` is ``smooth_l1``.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        area_weight: float = 1.0,
        area_loss: str = "l2",
        ignore_index: int = 255,
        reduction: str = "mean",
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if area_weight < 0 or ce_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if area_loss not in {"l1", "l2", "smooth_l1"}:
            raise ValueError("area_loss must be one of {'l1', 'l2', 'smooth_l1'}")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")

        self.ce_weight = ce_weight
        self.area_weight = area_weight
        self.area_loss_type = area_loss
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_l1_beta = smooth_l1_beta

    def forward(self, logits: Dict[str, paddle.Tensor], labels: Dict[str, paddle.Tensor]):
        if "semantic" not in logits:
            raise KeyError("logits dict must contain 'semantic'")
        if "label" not in labels:
            raise KeyError("labels dict must contain 'label'")

        loss = paddle.zeros([1], dtype=logits["semantic"].dtype)

        if self.ce_weight > 0:
            semantic_logits = logits["semantic"]
            semantic_label = labels["label"].squeeze(1).astype("int64")
            ce = F.cross_entropy(
                semantic_logits,
                semantic_label,
                ignore_index=self.ignore_index,
                reduction="mean",
            )
            loss = loss + self.ce_weight * ce

        if self.area_weight > 0:
            if "area" not in logits:
                raise KeyError("logits dict must contain 'area' when area_weight > 0")
            if "area_mask" not in labels:
                raise KeyError("labels dict must contain 'area_mask' when area_weight > 0")
            area_pred = logits["area"]
            target = labels["area_mask"].astype(area_pred.dtype)
            if target.ndim == 3:
                target = target.unsqueeze(1)
            target = paddle.nn.functional.interpolate(
                target,
                size=area_pred.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            reg = self._regression_loss(area_pred, target)
            loss = loss + self.area_weight * reg

        return loss

    def _regression_loss(self, pred: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        if self.area_loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        elif self.area_loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction=self.reduction)
        else:
            loss = F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta, reduction=self.reduction)
        return loss