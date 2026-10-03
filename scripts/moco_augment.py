import torch
import random


def augment_view(x):

    # x shape: [5, H, W]

    x = x.clone()

    # Left-right flip
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])

    # Up-down flip
    if random.random() < 0.5:
        x = torch.flip(x, dims=[1])

    # Small intensity change
    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1)
        shift = random.uniform(-0.05, 0.05)
        x = x * scale + shift

    # Small Gaussian noise
    if random.random() < 0.3:
        noise = torch.randn_like(x) * 0.02
        x = x + noise

    # Keep values between 0 and 1
    x = torch.clamp(x, 0, 1)

    return x