"""Counting metrics.

A prediction is a count per class, so precision and recall work on copies:
true positives = min(true, predicted), false positives = the extra predicted copies,
false negatives = the missed copies. Scores are averaged over the classes that occur in
the evaluated set ("active" classes): a class that never appears has no recall.
"""

from typing import Dict, Tuple

import numpy as np
import torch


def active_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    """Among (image, class) pairs whose true count is above 0, how many have the exact
    count predicted. Returns (correct, total). False positives are not counted."""
    mask = labels > 0
    return int((preds[mask] == labels[mask]).sum().item()), int(mask.sum().item())


def counting_scores(targets: np.ndarray, preds: np.ndarray) -> Dict[str, np.ndarray]:
    """targets, preds: (n_images, n_classes) counts. Per-class tp/fp/fn, precision,
    recall, f1, and `active` (the classes present in `targets`)."""
    tp = np.minimum(targets, preds).sum(axis=0)
    fp = np.maximum(0, preds - targets).sum(axis=0)
    fn = np.maximum(0, targets - preds).sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision),
                   where=(precision + recall) != 0)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "active": targets.sum(axis=0) > 0}


def mean_active_f1(targets: np.ndarray, preds: np.ndarray) -> float:
    """Macro F1 over the classes present in `targets` (the number the report quotes)."""
    scores = counting_scores(targets, preds)
    return float(scores["f1"][scores["active"]].mean())
