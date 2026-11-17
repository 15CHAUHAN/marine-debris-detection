"""Canonical multispectral preprocessing for Sentinel-2 imagery.

Standardized pipeline: 2%-98% Percentile Normalization -> Channel-wise CLAHE
Applied identically across training, validation, test, and dashboard inference.
"""
from __future__ import annotations

import cv2
import numpy as np


def _percentile_normalize_chw(image_chw: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    """Band-wise percentile normalization to [0, 1] with NaN tolerance.

    Args:
        image_chw: (C, H, W) float32 input.
        pmin: Lower percentile.
        pmax: Upper percentile.

    Returns:
        Normalized array with same shape, values in [0, 1].
    """
    output = np.zeros_like(image_chw, dtype=np.float32)
    for band in range(image_chw.shape[0]):
        values = image_chw[band].astype(np.float32)
        finite = values[np.isfinite(values)]
        clean = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
        low, high = np.percentile(finite, [pmin, pmax]) if finite.size else (0.0, 1.0)
        if high - low < 1e-6:
            low, high = 0.0, float(finite.max()) if finite.size and finite.max() > 0 else 1.0
        output[band] = np.clip((clean - low) / (high - low), 0.0, 1.0)
    return output


def clahe_multispectral_chw(
    image_chw: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply OpenCV CLAHE to each band of an (C, H, W) float32 array in [0, 1].

    Args:
        image_chw: (C, H, W) float32 input in [0, 1].
        clip_limit: CLAHE clip limit for contrast limiting.
        tile_grid: Tile grid size for adaptive histogram equalization.

    Returns:
        CLAHE-enhanced array with same shape, values in [0, 1].
    """
    out = np.zeros_like(image_chw, dtype=np.float32)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    for c in range(image_chw.shape[0]):
        clean = np.where(np.isfinite(image_chw[c]), image_chw[c], 0.0)
        u8 = np.round(np.clip(clean, 0.0, 1.0) * 255.0).astype(np.uint8)
        eq = clahe.apply(u8)
        out[c] = eq.astype(np.float32) / 255.0
    return out


def preprocess_satellite_input(
    image_chw: np.ndarray,
    apply_clahe: bool = True,
    pmin: float = 2.0,
    pmax: float = 98.0,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Deterministic 2-step pipeline: Percentile Normalization -> Multispectral CLAHE.

    This is the canonical preprocessing for Sentinel-2 multispectral imagery.
    Must be applied identically across training, validation, test, and inference.

    Args:
        image_chw: (C, H, W) float32 input (raw radiance/reflectance values).
        apply_clahe: If True, apply CLAHE after normalization (default True).
        pmin: Lower percentile for normalization.
        pmax: Upper percentile for normalization.
        clip_limit: CLAHE clip limit.
        tile_grid: CLAHE tile grid size.

    Returns:
        Preprocessed array (C, H, W) float32 in [0, 1].
    """
    # Step 1: Band-wise percentile normalization to [0, 1]
    norm = _percentile_normalize_chw(image_chw, pmin=pmin, pmax=pmax)

    # Step 2: Channel-wise CLAHE contrast adjustment
    if apply_clahe:
        return clahe_multispectral_chw(norm, clip_limit=clip_limit, tile_grid=tile_grid)
    return norm


def apply_clahe(
    image_hwc: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to every channel of a (H, W, C) float32 image in [0, 1].

    Legacy wrapper for compatibility with TrainTransform.
    """
    channels = [clahe_multispectral_chw(image_hwc[..., c:c+1].transpose(2, 0, 1),
                                         clip_limit=clip_limit, tile_grid=tile_grid).squeeze(0)
                for c in range(image_hwc.shape[-1])]
    return np.stack(channels, axis=-1)