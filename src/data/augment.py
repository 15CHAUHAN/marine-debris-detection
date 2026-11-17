"""Contrast enhancement and on-the-fly augmentation for training data.

CLAHE (OpenCV, per channel) and spatial/color transforms are applied ONLY to
training patches. Albumentations provides the spatial transforms; the OpenCV
CLAHE step handles the multispectral stack because Albumentations' CLAHE only
supports 1/3 channel images.

The canonical CLAHE implementation lives in src.data.preprocessing for
consistent reuse across training, validation, test, and inference.
"""
from __future__ import annotations

from typing import Callable

import albumentations as A
import cv2
import numpy as np

from src.data.preprocessing import apply_clahe as _apply_clahe
from src.data.preprocessing import clahe_multispectral_chw, preprocess_satellite_input


def apply_clahe(
    image_hwc: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to every channel of a (H, W, C) float32 image in [0, 1].

    Delegates to the canonical implementation in src.data.preprocessing.
    """
    return _apply_clahe(image_hwc, clip_limit=clip_limit, tile_grid=tile_grid)


def build_train_transforms(
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
) -> A.Compose:
    """Spatial/color transforms (image + mask in lockstep). CLAHE is applied separately."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            ),
        ]
    )


class TrainTransform:
    """Full training pipeline: OpenCV CLAHE then Albumentations spatial/color.

    Apply to a (H, W, C) float32 image and (H, W) int64 mask.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid
        self.compose = build_train_transforms()

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = apply_clahe(image, self.clip_limit, self.tile_grid)
        result = self.compose(image=image, mask=mask)
        return result["image"], result["mask"]


def identity(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """No-op passthrough for validation/test evaluation."""
    return image, mask


def wrap(
    transform: TrainTransform | A.Compose | None,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Adapt a pipeline object to the (image, mask) -> (image, mask) callable."""
    if transform is None:
        return identity
    return transform