"""Physics-aware multispectral weight initialization for Sentinel-2 bands.

When adapting ImageNet pretrained weights (3 RGB channels) to Sentinel-2
multispectral input (11 channels), naive replication across all channels
destroys the physical meaning of infrared bands.

This module provides:
- "physics_split": Maps ImageNet RGB to exact Sentinel-2 B02/B03/B04 slots,
  initializes non-RGB channels with physics-informed priors
- "imagenet_replicate": Legacy naive replication for backward compatibility
- "random": Kaiming initialization for training from scratch
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.init as init

# Sentinel-2 band indices (0-indexed for 11-channel input)
# Index 0: B01  Coastal Aerosol (443 nm)
# Index 1: B02  Blue (490 nm)          <- ImageNet Blue (channel 2)
# Index 2: B03  Green (560 nm)         <- ImageNet Green (channel 1)
# Index 3: B04  Red (665 nm)           <- ImageNet Red (channel 0)
# Index 4: B05  Red Edge 1 (705 nm)
# Index 5: B06  Red Edge 2 (740 nm)
# Index 6: B07  Red Edge 3 (783 nm)
# Index 7: B08  NIR Broad (842 nm)
# Index 8: B8A  Narrow NIR (865 nm)
# Index 9: B11  SWIR 1 (1610 nm)
# Index 10: B12 SWIR 2 (2190 nm)

# ImageNet RGB channel mapping: 0=Red, 1=Green, 2=Blue
IMAGENET_RGB_IDX = (0, 1, 2)
SENTINEL2_RGB_IDX = (3, 2, 1)  # B04=Red, B03=Green, B02=Blue


def _kaiming_normal_scaled(shape: tuple, scale: float = 1.0, mean: float = 0.0) -> torch.Tensor:
    """Generate Kaiming normal initialized tensor with custom scale and mean."""
    # shape is (out_c, kH, kW) for conv weight slice
    # fan_in = in_channels * kH * kW, but for a single output channel slice, we use kH*kW
    fan_in = shape[1] * shape[2]  # kH * kW
    std = math.sqrt(2.0 / fan_in) * scale
    return torch.normal(mean, std, size=shape)


def adapt_multispectral_first_conv(
    first_conv: nn.Conv2d,
    in_channels: int = 11,
    strategy: Literal["physics_split", "imagenet_replicate", "random"] = "physics_split",
) -> nn.Conv2d:
    """Adapt a 3-channel pretrained Conv2d layer to 11 Sentinel-2 channels.

    Sentinel-2 Band Layout (0-indexed):
      0: B01 Coastal Aerosol
      1: B02 Blue          <- ImageNet Blue (idx 2)
      2: B03 Green         <- ImageNet Green (idx 1)
      3: B04 Red           <- ImageNet Red (idx 0)
      4: B05 Red Edge 1
      5: B06 Red Edge 2
      6: B07 Red Edge 3
      7: B08 NIR Broad
      8: B8A Narrow NIR
      9: B11 SWIR 1
      10: B12 SWIR 2

    Strategies:
    - "physics_split" (default): Direct RGB mapping + physics-informed init for IR
    - "imagenet_replicate": Legacy mean+expand across all channels
    - "random": Kaiming normal initialization (for from-scratch training)
    """
    if first_conv.in_channels == in_channels:
        return first_conv

    new_conv = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )

    with torch.no_grad():
        w = first_conv.weight.data  # (out_c, 3, kH, kW)
        out_c, _, kH, kW = w.shape

        if strategy == "imagenet_replicate":
            # Legacy: replicate mean across all channels
            new_conv.weight.data = w.mean(dim=1, keepdim=True).expand(-1, in_channels, -1, -1).clone()

        elif strategy == "random":
            # From-scratch Kaiming initialization
            init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        elif strategy == "physics_split":
            # Start with Kaiming normal for all channels
            init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            # Direct RGB mapping: ImageNet channels -> Sentinel-2 slots
            # ImageNet: 0=Red, 1=Green, 2=Blue
            # Sentinel-2: 3=Red (B04), 2=Green (B03), 1=Blue (B02)
            new_conv.weight.data[:, 3, :, :] = w[:, 0, :, :].clone()  # Red <- Red
            new_conv.weight.data[:, 2, :, :] = w[:, 1, :, :].clone()  # Green <- Green
            new_conv.weight.data[:, 1, :, :] = w[:, 2, :, :].clone()  # Blue <- Blue

            # B01 Coastal Aerosol (idx 0): Rayleigh scattering ~ 0.8*Blue + 0.2*mean
            new_conv.weight.data[:, 0, :, :] = (
                0.8 * w[:, 2, :, :] + 0.2 * w.mean(dim=1)
            ).clone()

            # Red Edge bands (idx 4, 5, 6): interpolate between Red and NIR
            # Initialize with scaled Red weights (Red Edge is spectrally adjacent)
            for idx in (4, 5, 6):
                new_conv.weight.data[:, idx, :, :] = (
                    w[:, 0, :, :] * 0.7 + _kaiming_normal_scaled((out_c, kH, kW), scale=0.3)
                ).clone()

            # NIR bands (idx 7, 8): High vegetation/debris reflection
            # Initialize with strong positive weights (NIR is bright for vegetation/plastic)
            for idx in (7, 8):
                new_conv.weight.data[:, idx, :, :] = (
                    w[:, 0, :, :] * 0.5 + _kaiming_normal_scaled((out_c, kH, kW), scale=0.5, mean=0.1)
                ).clone()

            # SWIR bands (idx 9, 10): Strong water absorption -> near zero over water
            # Initialize with negative bias (absorption features in SWIR for water/plastic)
            for idx, scale in ((9, 0.05), (10, 0.02)):
                new_conv.weight.data[:, idx, :, :] = (
                    w[:, 0, :, :] * 0.1 + _kaiming_normal_scaled((out_c, kH, kW), scale=scale, mean=-0.05)
                ).clone()

            # Rescale to preserve input variance: Var(sum W_i X_i) ≈ Var(sum W_rgb X_rgb)
            # Original 3 channels, now 11 channels -> scale by sqrt(3/11)
            new_conv.weight.data.mul_(math.sqrt(3.0 / in_channels))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()

    return new_conv


def get_seco_encoder(encoder_name: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """Attempt to load SeCo (Seasonal Contrast) or SSL4EO-S12 pretrained encoder.

    Args:
        encoder_name: "resnet50" or "resnet18" (SeCo variants)
        pretrained: Whether to load pretrained weights

    Returns:
        A timm/smp-compatible encoder, or None if SeCo weights unavailable.

    Note:
        SeCo checkpoints are typically hosted on GitHub/GDrive. This function
        attempts to load from local cache (~/.cache/torch/seco) or downloads
        from known URLs. Falls back to ImageNet if unavailable.
    """
    try:
        import timm
        # Try to load from local cache or torchgeo if available
        # For now, return None to signal ImageNet fallback
        # TODO: Implement actual SeCo/SSL4EO-S12 checkpoint loading
        return None
    except ImportError:
        return None


def load_seco_checkpoint(encoder_name: str) -> dict | None:
    """Load SeCo pretrained checkpoint for Sentinel-2.

    Args:
        encoder_name: "resnet50" or "resnet18"

    Returns:
        State dict or None if unavailable.
    """
    # SeCo ResNet-50 Sentinel-2 checkpoint (from official release)
    # Expected location: ~/.cache/torch/seco/seco_resnet50_s2.pth
    # Or download from: https://github.com/zhu-xlab/SeCo/releases
    import os
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "torch" / "seco"
    if encoder_name == "resnet50":
        ckpt_path = cache_dir / "seco_resnet50_s2.pth"
    elif encoder_name == "resnet18":
        ckpt_path = cache_dir / "seco_resnet18_s2.pth"
    else:
        return None

    if ckpt_path.exists():
        return torch.load(ckpt_path, map_location="cpu")
    return None