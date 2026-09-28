"""ResNet-18 encoder for the 2D (and, with in_channels=5, the 2.5D) approach.

Deliberately the stock torchvision ResNet-18. The ONLY change is `in_channels`
on conv1, so the 2D and 2.5D models differ by exactly one number and any
performance gap is attributable to slice context rather than architecture.

Input stays at the native 72x80 crop resolution -- no upsampling to 224, since
we train from random initialisation and there are no ImageNet weights to match.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

from src.common import config as C


def build_encoder(in_channels: int = 1) -> nn.Module:
    """ResNet-18 trunk with the classifier head removed (outputs 512-d)."""
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                        bias=False)
    m.fc = nn.Identity()
    return m


class ResNet18Classifier(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = C.NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = build_encoder(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.encoder(x)))

    def load_encoder(self, state: dict) -> tuple[list[str], list[str]]:
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        return list(missing), list(unexpected)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
