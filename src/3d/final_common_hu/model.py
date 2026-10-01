"""Shared lightweight 3D ResNet-18 for every final 3D experiment."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import BLOCK_COUNTS, WIDTHS


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels),
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CommonHUResNet3D18(nn.Module):
    block_counts = BLOCK_COUNTS
    widths = WIDTHS

    def __init__(self, num_classes: int, dropout: float = 0.20) -> None:
        super().__init__()
        if num_classes not in (2, 3):
            raise ValueError("num_classes must be 2 or 3")
        self.num_classes = num_classes
        self.dropout_probability = dropout
        self.stem = nn.Sequential(
            nn.Conv3d(1, WIDTHS[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(WIDTHS[0]), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = self._layer(WIDTHS[0], WIDTHS[0], 2, 1)
        self.layer2 = self._layer(WIDTHS[0], WIDTHS[1], 2, 2)
        self.layer3 = self._layer(WIDTHS[1], WIDTHS[2], 2, 2)
        self.layer4 = self._layer(WIDTHS[2], WIDTHS[3], 2, 2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(WIDTHS[-1], num_classes)
        self._init_weights()

    @staticmethod
    def _layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock3D(in_channels, out_channels, stride)]
        layers.extend(BasicBlock3D(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01); nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return torch.flatten(self.global_pool(x), 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.dropout(self.forward_features(x)))

    def backbone_state_dict(self) -> dict[str, Tensor]:
        return {key: value for key, value in self.state_dict().items() if key.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4."))}


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
