"""Debris concentration heatmap generation.

Turns a predicted binary Marine Debris mask into a continuous concentration
map by convolving with a Gaussian kernel. The blurred signal expresses local
debris density per patch and is what gets georeferenced and visualized.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def debris_concentration(
    debris_mask: np.ndarray,
    sigma: float = 2.0,
    truncate: float = 4.0,
    normalize: bool = True,
) -> np.ndarray:
    """Blur a binary debris mask into a float32 concentration map.

    Args:
        debris_mask: binary array (H, W), 1 where Marine Debris was predicted.
        sigma: Gaussian standard deviation in pixels.
        truncate: truncate the Gaussian at this many sigmas (used by
            scipy.ndimage.gaussian_filter).
        normalize: rescale the blurred map to [0, 1]. When the mask is empty
            the result stays all zeros.

    Returns:
        float32 (H, W) concentration values. Values are exact blur outputs
        when normalize=False, otherwise scaled to [0, 1].
    """
    mask = np.asarray(debris_mask, dtype=np.float32)
    blurred = gaussian_filter(mask, sigma=sigma, truncate=truncate)
    if not normalize:
        return blurred.astype(np.float32)
    peak = float(np.max(blurred)) if blurred.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(blurred, dtype=np.float32)
    return (blurred / peak).astype(np.float32)