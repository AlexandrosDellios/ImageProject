"""Small helpers shared by the scripts."""

import random
from typing import Optional

import numpy as np
import torch


def pick_device(name: Optional[str] = None) -> torch.device:
    """`name` if given ("cpu", "cuda", "mps"), otherwise CUDA, then Apple MPS, then CPU."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
