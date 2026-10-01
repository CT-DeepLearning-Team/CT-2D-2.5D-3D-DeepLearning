"""Frozen common-HU preprocessing for every final 3D experiment."""

from __future__ import annotations

import numpy as np

from .config import CLIP_MAX, CLIP_MIN, EXPECTED_VOLUME_SHAPE, NORMALIZATION_DENOMINATOR


def preprocess_volume(volume: np.ndarray) -> np.ndarray:
    """Return a float32 [1,D,H,W] tensor-ready array in [0,1]."""

    array = np.asarray(volume)
    if array.shape != EXPECTED_VOLUME_SHAPE:
        raise ValueError(f"Expected {EXPECTED_VOLUME_SHAPE}, got {array.shape}")
    clipped = np.clip(array.astype(np.float32, copy=False), CLIP_MIN, CLIP_MAX)
    normalized = (clipped - np.float32(CLIP_MIN)) / np.float32(NORMALIZATION_DENOMINATOR)
    return np.ascontiguousarray(normalized[None], dtype=np.float32)


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != EXPECTED_VOLUME_SHAPE:
        raise ValueError(f"Expected mask {EXPECTED_VOLUME_SHAPE}, got {array.shape}")
    return np.ascontiguousarray(array.astype(np.uint8, copy=False)[None])
