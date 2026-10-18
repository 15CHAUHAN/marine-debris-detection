"""U-Net segmentation models for MARIDA (11 input channels, 11 output classes).

Built on segmentation-models-pytorch. When the encoder is pretrained on
ImageNet (3 channels) but the input has 11 channels, the first convolution
weights are adapted using physics-informed multispectral initialization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.data.encoding import N_CLASSES
from src.models.weight_init import adapt_multispectral_first_conv

DEFAULT_ENCODER = "resnet34"


def _adapt_first_convolution(model: smp.Unet, in_channels: int, strategy: str = "physics_split") -> None:
    """Adapt conv1 weights to in_channels using physics-informed initialization."""
    first_conv = model.encoder.conv1
    new_conv = adapt_multispectral_first_conv(first_conv, in_channels, strategy)
    model.encoder.conv1 = new_conv


def build_unet(
    in_channels: int = 11,
    classes: int = N_CLASSES,
    encoder_name: str = DEFAULT_ENCODER,
    pretrained: bool = True,
    attention: bool = False,
    weight_init_strategy: str = "physics_split",
) -> smp.Unet:
    """Build a U-Net. attention=True adds scSE attention gates to the decoder.

    Args:
        weight_init_strategy: "physics_split" (default), "imagenet_replicate", or "random"
    """
    encoder_weights = "imagenet" if pretrained else None
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
        decoder_attention_type="scse" if attention else None,
    )
    if pretrained and in_channels != 3:
        _adapt_first_convolution(model, in_channels, weight_init_strategy)
    elif in_channels != 3:
        model.encoder.conv1 = nn.Conv2d(
            in_channels,
            model.encoder.conv1.out_channels,
            kernel_size=model.encoder.conv1.kernel_size,
            stride=model.encoder.conv1.stride,
            padding=model.encoder.conv1.padding,
            bias=model.encoder.conv1.bias is not None,
        )
    return model