"""EfficientNet+Transformer U-Net.

An EfficientNet-B0 (timm, ImageNet pretrained) encoder feeds a U-Net decoder
through an intermediate transformer stage: the bottleneck feature map is
embedded, processed by a stack of self-attention layers over spatial tokens
(with a learned positional embedding), and projected back before decoding.
This adds global receptive field on top of the CNN backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

from src.data.encoding import N_CLASSES
from src.models.weight_init import adapt_multispectral_first_conv

_TRANSFORM_DIM = 256


def _adapt_first_convolution(model: smp.Unet, in_channels: int, strategy: str = "physics_split") -> None:
    """Adapt the encoder's first conv to in_channels using physics-informed initialization."""
    encoder = model.encoder
    first_conv = getattr(encoder, "conv1", None) or getattr(encoder, "_conv_stem", None)
    new_conv = adapt_multispectral_first_conv(first_conv, in_channels, strategy)
    if hasattr(encoder, "conv1"):
        encoder.conv1 = new_conv
    else:
        encoder._conv_stem = new_conv


class _BottleneckTransformer(nn.Module):
    """Spatial transformer block applied to the encoder bottleneck."""

    def __init__(self, in_channels: int, nhead: int = 4, num_layers: int = 2, dim: int = _TRANSFORM_DIM) -> None:
        super().__init__()
        self.dim = dim
        self.proj_in = nn.Conv2d(in_channels, dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, dim, 1, 1))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.stack = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj_out = nn.Conv2d(dim, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x) + self.pos
        b, c, hh, ww = h.shape
        tokens = h.flatten(2).transpose(1, 2)
        tokens = self.stack(tokens)
        h = tokens.transpose(1, 2).view(b, self.dim, hh, ww)
        return self.proj_out(h)


class EffnetTransformerUnet(nn.Module):
    """U-Net with EfficientNet-B0 encoder and a transformer bottleneck."""

    def __init__(
        self,
        in_channels: int = 11,
        classes: int = N_CLASSES,
        encoder_name: str = "efficientnet-b0",
        pretrained: bool = True,
        nhead: int = 4,
        num_layers: int = 2,
        weight_init_strategy: str = "physics_split",
    ) -> None:
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=classes,
            activation=None,
        )
        if pretrained and in_channels != 3:
            _adapt_first_convolution(self.unet, in_channels, weight_init_strategy)
        elif in_channels != 3:
            # Adjust in_channels for non-pretrained case
            first_conv = getattr(self.unet.encoder, "conv1", None) or getattr(self.unet.encoder, "_conv_stem", None)
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            if hasattr(self.unet.encoder, "conv1"):
                self.unet.encoder.conv1 = new_conv
            else:
                self.unet.encoder._conv_stem = new_conv
        bottleneck = self.unet.encoder.out_channels[-1]
        self.transformer = _BottleneckTransformer(bottleneck, nhead=nhead, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = list(self.unet.encoder(x))
        features[-1] = self.transformer(features[-1])
        decoder_out = self.unet.decoder(features)
        return self.unet.segmentation_head(decoder_out)


def build_effnet_transformer_unet(
    in_channels: int = 11,
    classes: int = N_CLASSES,
    pretrained: bool = True,
    weight_init_strategy: str = "physics_split",
) -> EffnetTransformerUnet:
    """Build the EfficientNet+Transformer variant."""
    return EffnetTransformerUnet(in_channels=in_channels, classes=classes, pretrained=pretrained, weight_init_strategy=weight_init_strategy)