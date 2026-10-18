"""CNN+U-Net: a plain U-Net with a lightweight CNN encoder.

Deliberately uses a small, randomly-initialized convolutional encoder (no
ImageNet weights) so it is a genuine architectural contrast to the
ResNet-34-pretrained baseline and the attention candidate. The encoder is a
4-stage conv stack (16/32/64/128 channels); the decoder mirrors it with
bilinear upsampling and skip concatenation back to full input resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.encoding import N_CLASSES

_INIT_CHANNELS = 16


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Encoder(nn.Module):
    """4-stage conv stack; returns a feature map per stage BEFORE pooling."""

    def __init__(self, in_channels: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        prev = in_channels
        for c in channels:
            self.stages.append(nn.Sequential(_ConvBlock(prev, c), nn.MaxPool2d(2)))
            prev = c

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        stage_out = x
        for conv_block, pool in self.stages:
            stage_out = conv_block(stage_out)
            features.append(stage_out)
            stage_out = pool(stage_out)
        return features


class _Decoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict()
        for i in range(len(channels) - 1, 0, -1):
            self.blocks[str(i)] = _ConvBlock(channels[i] + channels[i - 1], channels[i - 1])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for i in range(len(features) - 1, 0, -1):
            x = F.interpolate(x, size=features[i - 1].shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, features[i - 1]], dim=1)
            x = self.blocks[str(i)](x)
        return x


class CNNUnet(nn.Module):
    """U-Net with a compact from-scratch CNN encoder."""

    def __init__(self, in_channels: int = 11, classes: int = N_CLASSES) -> None:
        super().__init__()
        self.channels = (_INIT_CHANNELS, _INIT_CHANNELS * 2, _INIT_CHANNELS * 4, _INIT_CHANNELS * 8)
        self.encoder = _Encoder(in_channels, self.channels)
        self.decoder = _Decoder(self.channels)
        self.head = nn.Conv2d(_INIT_CHANNELS, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(self.encoder(x)))


def build_cnn_unet(in_channels: int = 11, classes: int = N_CLASSES) -> CNNUnet:
    """Build the CNN+U-Net variant."""
    return CNNUnet(in_channels=in_channels, classes=classes)