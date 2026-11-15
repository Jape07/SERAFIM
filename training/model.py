"""The motion-dipole CNN (task A9, stage 2).

Same stem as Sentinel's AstroNetDECals (3 conv blocks of 16, 32 and 64 filters, 3 x 3, ReLU,
max pool), with BatchNorm and global average pooling instead of a 4096-wide dense layer: ~26k
parameters instead of ~549k, which suits a few thousand stamps. One output: the logit of
p_dipole (sigmoid outside, in the loss and in serafim/ai/predict.py).
"""

from __future__ import annotations

import torch
from torch import nn

from serafim.ai.stamps import CHANNELS


class DipoleCNN(nn.Module):
    def __init__(self, base: int = 16, dropout: float = 0.3, channels: int = len(CHANNELS)):
        super().__init__()
        layers, width = [], channels
        for out in (base, 2 * base, 4 * base):
            layers += [nn.Conv2d(width, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(),
                       nn.MaxPool2d(2)]  # fmt: skip
            width = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


class Ensemble(nn.Module):
    """The mean logit of several networks trained with different seeds: steadier p_dipole than
    any one of them, exported as one ONNX model."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(x) for member in self.members]).mean(0)


def parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
