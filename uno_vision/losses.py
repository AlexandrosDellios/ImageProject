"""Focal loss over the count bins.

The model outputs (batch, 56, 6) logits; the training loop permutes them to (batch, 6, 56)
so that cross-entropy treats the 6 count bins as the classes and each of the 56 card
classes as one prediction. Targets are (batch, 56) counts.

Most (image, class) pairs have a count of 0 and are easy; the focal term (1 - p)^gamma
shrinks their loss so the rare non-zero counts drive the gradient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss built on cross-entropy, with optional label smoothing.

    inputs: (B, C, ...) logits, here (B, 6, 56); targets: (B, ...) class indices, here (B, 56).
    reduction: "mean", "sum" or "none".
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction {reduction!r}")
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none", label_smoothing=self.label_smoothing)
        # With label smoothing, exp(-ce) is not exactly the target probability, but this is
        # the formula the models were trained with, so it stays.
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, label_smoothing={self.label_smoothing}, reduction={self.reduction!r}"
