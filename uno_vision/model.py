"""ResNet-18 with CBAM attention and a counting head.

The network is trained from scratch (no pretrained weights). After the last residual
stage, CBAM reweights the 512-channel feature map, first per channel ("what"), then per
position ("where"). The head predicts, for each of the 56 classes, logits over 0 to 5
copies: output shape (batch, 56, 6).

The attribute names (features, ca, sa, fc) are the keys of the saved weights: keep them.
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torchvision.models as models

from .classes import MAX_COPIES, NUM_CLASSES


class ChannelAttention(nn.Module):
    """CBAM channel attention: a shared MLP on the average- and max-pooled channels."""

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM spatial attention: a 7x7 convolution over the channel-wise mean and max."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class UnoCountingResNet(nn.Module):
    """ResNet-18 trunk + CBAM + a (num_classes x max_copies) counting head (11.4M parameters)."""

    def __init__(self, num_classes: int = NUM_CLASSES, max_copies: int = MAX_COPIES, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.max_copies = max_copies

        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])   # drop avgpool and fc
        self.ca = ChannelAttention(512)
        self.sa = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes * max_copies),
        )

    def forward(self, x):
        x = self.features(x)
        x = x * self.ca(x)
        x = x * self.sa(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x).view(-1, self.num_classes, self.max_copies)


def load_model(weights: Union[str, Path], device: torch.device) -> UnoCountingResNet:
    """The network with `weights` loaded, on `device`, in eval mode."""
    model = UnoCountingResNet()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


if __name__ == "__main__":   # python -m uno_vision.model
    model = UnoCountingResNet()
    print("output shape:", tuple(model(torch.randn(2, 3, 336, 336)).shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
