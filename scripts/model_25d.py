import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet25D(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        # Create ResNet-18
        self.model = resnet18(weights=None)

        # Change first layer from 3 input channels to 5
        self.model.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Change final layer from 1000 classes to 3 classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)