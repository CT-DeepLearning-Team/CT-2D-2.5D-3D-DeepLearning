"""One frozen mild augmentation recipe for Model C, binary, and MoCo."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import (
    GAUSSIAN_NOISE_STD,
    INTENSITY_SCALE_RANGE,
    INTENSITY_SHIFT_RANGE,
    ROTATION_DEGREES,
    TRANSLATION_VOXELS,
)


class CommonHUAugmentation:
    """Random mild augmentation for already normalized [0,1] CT crops."""

    def __init__(self) -> None:
        self.max_rotation_radians = math.radians(ROTATION_DEGREES)
        self.rotation_degrees = ROTATION_DEGREES
        self.translation_voxels = TRANSLATION_VOXELS
        self.intensity_scale_range = INTENSITY_SCALE_RANGE
        self.intensity_shift_range = INTENSITY_SHIFT_RANGE
        self.noise_std = GAUSSIAN_NOISE_STD

    @staticmethod
    def _uniform(low: float, high: float, device: torch.device) -> Tensor:
        return torch.empty((), device=device).uniform_(low, high)

    def _theta(self, image: Tensor) -> Tensor:
        _, depth, height, width = image.shape
        device, dtype = image.device, image.dtype
        ax = self._uniform(-self.max_rotation_radians, self.max_rotation_radians, device)
        ay = self._uniform(-self.max_rotation_radians, self.max_rotation_radians, device)
        az = self._uniform(-self.max_rotation_radians, self.max_rotation_radians, device)
        sx, cx = torch.sin(ax), torch.cos(ax)
        sy, cy = torch.sin(ay), torch.cos(ay)
        sz, cz = torch.sin(az), torch.cos(az)
        zero = torch.zeros((), device=device, dtype=dtype)
        one = torch.ones((), device=device, dtype=dtype)
        rx = torch.stack((torch.stack((one, zero, zero)), torch.stack((zero, cx, -sx)), torch.stack((zero, sx, cx))))
        ry = torch.stack((torch.stack((cy, zero, sy)), torch.stack((zero, one, zero)), torch.stack((-sy, zero, cy))))
        rz = torch.stack((torch.stack((cz, -sz, zero)), torch.stack((sz, cz, zero)), torch.stack((zero, zero, one))))
        rotation = rz @ ry @ rx
        translations = torch.stack(tuple(
            self._uniform(-TRANSLATION_VOXELS, TRANSLATION_VOXELS, device)
            for _ in range(3)
        )).to(dtype)
        normalized = torch.stack((2 * translations[0] / width, 2 * translations[1] / height, 2 * translations[2] / depth))
        return torch.cat((rotation, normalized[:, None]), dim=1).unsqueeze(0)

    def __call__(self, image: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        image_batch = image.float().unsqueeze(0)
        theta = self._theta(image_batch[0])
        grid = F.affine_grid(theta, image_batch.shape, align_corners=False)
        transformed = F.grid_sample(image_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(0)
        scale = self._uniform(*INTENSITY_SCALE_RANGE, transformed.device)
        shift = self._uniform(*INTENSITY_SHIFT_RANGE, transformed.device)
        transformed = transformed * scale + shift + torch.randn_like(transformed) * GAUSSIAN_NOISE_STD
        transformed = transformed.clamp(0.0, 1.0).contiguous()
        if mask is None:
            return transformed, None
        mask_batch = mask.float().unsqueeze(0)
        transformed_mask = F.grid_sample(mask_batch, grid, mode="nearest", padding_mode="zeros", align_corners=False).squeeze(0)
        return transformed, (transformed_mask >= 0.5).to(torch.uint8).contiguous()
