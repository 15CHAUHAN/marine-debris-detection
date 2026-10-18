"""U-Net with scSE attention gates (the candidate in the A/B comparison)."""
from __future__ import annotations

import segmentation_models_pytorch as smp

from .unet import build_unet


def build_attention_unet(in_channels: int = 11, classes: int = 11, encoder_name: str = "resnet34", pretrained: bool = True, weight_init_strategy: str = "physics_split") -> smp.Unet:
    """Build the U-Net + attention (scSE) candidate model."""
    return build_unet(
        in_channels=in_channels,
        classes=classes,
        encoder_name=encoder_name,
        pretrained=pretrained,
        attention=True,
        weight_init_strategy=weight_init_strategy,
    )