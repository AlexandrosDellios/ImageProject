"""Grad-CAM: which part of a crop drives one (class, count) logit.

The gradient of the chosen logit with respect to a residual stage's feature map, averaged
over positions, gives one weight per channel; the weighted sum of the channels, kept
where positive, is the heatmap. It is computed at the stage's resolution (11 x 11 for
layer4, 21 x 21 for layer3 at the 336-pixel input) and upsampled for display.

    with GradCAM(model, "layer4") as cam:
        heat = cam(image_tensor, CLASS_TO_IDX["r_7"], 1)    # (11, 11), values in [0, 1]
    rgb = overlay(display_crop(path), colorize(heat))

scripts/evaluate.py draws a grid of these for the held-out real crops.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from .transforms import INPUT_SIZE, PadToSquare

# Index of each residual stage in UnoCountingResNet.features (conv1, bn1, relu, maxpool,
# layer1..layer4). The report's Grad-CAM figures came from layer3: the old script asked
# for layer4 through an attribute that does not exist and its fallback took features[-2].
LAYERS = {"layer3": -2, "layer4": -1}


def target_layer(model: nn.Module, name: str = "layer4") -> nn.Module:
    if name not in LAYERS:
        raise ValueError(f"unknown layer {name!r}, expected one of {sorted(LAYERS)}")
    return model.features[LAYERS[name]]


class GradCAM:
    """Hooks one stage of `model` (a module, or "layer3" / "layer4") and returns heatmaps.

    The model should be in eval mode, so dropout and batch-norm statistics do not change
    the logit being explained. Use it as a context manager, or call remove(), so the
    hooks do not stay on the model."""

    def __init__(self, model: nn.Module, layer: Union[str, nn.Module] = "layer4"):
        self.model = model
        self.layer = target_layer(model, layer) if isinstance(layer, str) else layer
        self.activations = None
        self.gradients = None
        self._handles = [
            self.layer.register_forward_hook(self._save_activation),
            self.layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, class_idx: int, count_idx: int) -> np.ndarray:
        """image: one normalised crop, (3, H, W) or (1, 3, H, W), on the model's device.
        Returns the heatmap for logit [class_idx, count_idx], scaled to [0, 1]."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(image)                    # (1, num_classes, max_copies)
            logits[0, class_idx, count_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        self.model.zero_grad()                            # leave no gradients behind
        return cam.cpu().numpy()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def display_crop(image: Union[str, Path, Image.Image], size: int = INPUT_SIZE) -> Image.Image:
    """The crop as the network sees it (black bars, square, resized), before normalisation,
    so the heatmap lines up with it pixel for pixel."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return PadToSquare(fill=0)(image.convert("RGB")).resize((size, size))


def colorize(heatmap: np.ndarray, size: int = INPUT_SIZE, cmap: str = "jet") -> np.ndarray:
    """A [0, 1] heatmap -> (size, size, 3) float RGB in [0, 1], upsampled with Lanczos.
    "jet" matches the report's figures."""
    up = Image.fromarray(np.uint8(255 * np.clip(heatmap, 0, 1))).resize((size, size), Image.LANCZOS)
    return colormaps[cmap](np.asarray(up) / 255.0)[:, :, :3]


def overlay(image: Union[Image.Image, np.ndarray], colored: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colourised heatmap over the crop: (1 - alpha) * crop + alpha * heatmap."""
    base = np.asarray(image)
    base = base / 255.0 if base.dtype == np.uint8 else base.astype(np.float64)
    return np.clip((1 - alpha) * base + alpha * colored, 0, 1)
