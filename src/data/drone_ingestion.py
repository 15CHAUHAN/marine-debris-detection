"""Drone / aerial RGB image ingestion and satellite-model channel adapter.

Loaders for standard 3-channel RGB drone/aerial imagery (.jpg, .png) plus an
adapter that maps 3-channel RGB to the 11-channel Sentinel-2 input contract
used by the satellite-trained backbones. The 11-channel encoders expect
distinct spectral bands (B01-B12). Feeding 11 identical channels destroys
all spectral information. This module provides a physics-informed spectral
approximation for cases where RGB data must be processed by a multispectral
model, but the architecturally correct path is the TACO YOLOv8 RGB detector.

Adapter modes:
- ``physics_approx`` (default): Physics-based spectral synthesis placing true
  RGB into Sentinel-2 B02/B03/B04 slots, approximating NIR/SWIR from visible.
- ``rgb_bands``: B/G/R placed in B2/B3/B4 slots (indices 1/2/3); remaining
  slots hold per-pixel RGB mean (legacy compatibility).
- ``replicate`` (deprecated): every output channel holds per-pixel RGB mean.
- ``zero_pad``: RGB occupies channels 0..2, remaining 8 are zero.
- ``rgb3``: native 3-channel passthrough for models trained on RGB only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

N_MODEL_CHANNELS = 11

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DroneImageMeta:
    path: Path
    width: int
    height: int
    channels: int


def read_rgb_image(path: Path) -> np.ndarray:
    """Read a .jpg/.png/.bmp drone/aerial image.

    Returns a float32 array of shape ``(3, H, W)`` with values in [0, 1].
    Grayscale inputs are converted to 3 channels by replication; alpha
    channels are dropped.
    """
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.transpose(2, 0, 1)
    return (rgb.astype(np.float32) / 255.0).astype(np.float32)


def read_image_meta(path: Path) -> DroneImageMeta:
    """Return size metadata for an image file without loading pixels."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    h, w = image.shape[:2]
    return DroneImageMeta(path=path, width=w, height=h, channels=3)


def to_model_channels(
    rgb: np.ndarray,
    n_channels: int = N_MODEL_CHANNELS,
    mode: str = "physics_approx",
) -> np.ndarray:
    """Map a float32 ``(3, H, W)`` RGB image in [0, 1] to model channels.

    ``rgb3`` returns the input unchanged (3 channels); all other modes build
    an ``(n_channels, H, W)`` array from the given RGB.

    Physics-informed mode (``physics_approx``) places true RGB into their
    exact Sentinel-2 band slots and synthesizes missing bands:
      - idx 0 (B01 Coastal Aerosol): Rayleigh-scattered blue approximation
      - idx 1 (B02 Blue):           True Blue (rgb[2])
      - idx 2 (B03 Green):          True Green (rgb[1])
      - idx 3 (B04 Red):            True Red (rgb[0])
      - idx 4 (B05 Red Edge 1):     0.7 * Red + 0.3 * Green
      - idx 5 (B06 Red Edge 2):     0.5 * Red + 0.5 * Green
      - idx 6 (B07 Red Edge 3):     0.6 * Red + 0.4 * Green (NIR onset)
      - idx 7 (B08 NIR Broad):      High over vegetation/plastic, low over water
      - idx 8 (B8A Narrow NIR):     Matched to B08
      - idx 9 (B11 SWIR 1):         Water-absorbed base (0.05 * luminance)
      - idx 10 (B12 SWIR 2):        Water-absorbed base (0.02 * luminance)

    For aquatic scenes, water strongly absorbs SWIR (idx 9, 10), so these
    channels are near-zero rather than equal to visible luminance.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"expected (3, H, W) RGB, got {rgb.shape}")
    if mode == "rgb3":
        return rgb
    if mode not in ("physics_approx", "replicate", "zero_pad", "rgb_bands"):
        raise ValueError(
            f"unknown adapter mode {mode!r}; choose from physics_approx, replicate, zero_pad, rgb_bands, rgb3"
        )

    # Extract true R, G, B channels (rgb[0]=Red, rgb[1]=Green, rgb[2]=Blue)
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    mean = rgb.mean(axis=0, keepdims=True)[0]  # (H, W)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    out = np.zeros((n_channels, *rgb.shape[1:]), dtype=np.float32)

    if mode == "physics_approx":
        # True Sentinel-2 band placement
        out[1] = b                    # B02 Blue
        out[2] = g                    # B03 Green
        out[3] = r                    # B04 Red

        # Coastal Aerosol (B01): Rayleigh scattering ~ 0.8 * Blue + 0.2 * mean
        out[0] = 0.8 * b + 0.2 * mean

        # Red Edge bands (B05, B06, B07): progressive transition from Red to NIR
        out[4] = 0.7 * r + 0.3 * g    # B05 Red Edge 1
        out[5] = 0.5 * r + 0.5 * g    # B06 Red Edge 2
        out[6] = 0.6 * r + 0.4 * g    # B07 Red Edge 3 (NIR onset)

        # NIR bands (B08, B8A): vegetation/plastic bright, water dark
        # Approximate: NIR ~ Red * 1.2 - Green * 0.2 (vegetation reflectance)
        nir_approx = np.clip(1.2 * r - 0.2 * g, 0.0, 1.0)
        out[7] = nir_approx           # B08 NIR Broad
        out[8] = nir_approx           # B8A Narrow NIR

        # SWIR bands (B11, B12): strong water absorption -> near zero over water
        # Over land/plastic: weak reflection. Use low fraction of luminance.
        out[9] = 0.05 * luminance     # B11 SWIR 1
        out[10] = 0.02 * luminance    # B12 SWIR 2

        logger.warning(
            "Approximating 11 Sentinel-2 bands from 3-channel RGB. "
            "For true litter detection on RGB photos, use TACO detector."
        )
    elif mode == "replicate":
        import warnings
        warnings.warn(
            "drone_ingestion.to_model_channels mode='replicate' is DEPRECATED and architecturally unsound. "
            "It discards all spectral information by assigning luminance mean to all 11 channels. "
            "Use mode='physics_approx' (default) or route RGB photos to the TACO YOLOv8 detector.",
            DeprecationWarning,
            stacklevel=2,
        )
        out[:] = mean
    elif mode == "zero_pad":
        out[:3] = rgb
    else:  # rgb_bands
        out[1] = b
        out[2] = g
        out[3] = r
        out[0] = mean
        out[4:] = mean

    # Ensure all values are finite and in [0, 1]
    np.clip(out, 0.0, 1.0, out=out)
    return out


class DroneImageDataset(Dataset):
    """Dataset over drone/aerial image files.

    Yields ``(model_input, rgb_display)`` where ``model_input`` is a float32
    ``(n_channels, H, W)`` tensor ready for a segmentation backbone and
    ``rgb_display`` is an uint8 ``(H, W, 3)`` tensor for visualization.
    """

    def __init__(
        self,
        image_paths: list[Path | str],
        adapter_mode: str = "replicate",
        n_channels: int = N_MODEL_CHANNELS,
    ) -> None:
        self.paths = [Path(p) for p in image_paths]
        self.adapter_mode = adapter_mode
        self.n_channels = n_channels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_chw = read_rgb_image(self.paths[index])
        model_input = to_model_channels(
            rgb_chw, n_channels=self.n_channels, mode=self.adapter_mode
        )
        rgb_display = np.ascontiguousarray(
            (np.clip(rgb_chw, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )
        return (
            torch.from_numpy(np.ascontiguousarray(model_input, dtype=np.float32)),
            torch.from_numpy(rgb_display),
        )