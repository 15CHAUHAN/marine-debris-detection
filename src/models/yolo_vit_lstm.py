"""YOLO + ViT + LSTM hybrid segmentation model.

Uses the YOLOv8n backbone as a multi-scale feature extractor, a Vision
Transformer (ViT) block for spatial self-attention over the bottleneck, and
a lightweight bidirectional GRU to model sequential spatial context before
decoding back to full-resolution 11-class segmentation logits.

The sequential dimension for the GRU is a raster-scan ordering of the
spatial grid tokens produced by the ViT, capturing left-to-right,
top-to-bottom spatial dependencies.

Regularization:
- Configurable dropout (default 0.2) in ViT attention, FFN, and GRU bridge
- LayerNorm before/after attention blocks
- Reduced bottleneck dimension (default 128 vs 256) for parameter efficiency
- Bidirectional GRU (25% fewer params than LSTM) with dropout between steps
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.encoding import N_CLASSES
from src.models.weight_init import adapt_multispectral_first_conv

# Default regularization and bottleneck settings
_DEFAULT_DROPOUT = 0.2
_DEFAULT_BOTTLENECK_DIM = 128
_VIT_NHEAD = 4
_VIT_LAYERS = 2
_GRU_HIDDEN = 64  # bidirectional -> 128 output


def _load_yolo_backbone() -> nn.ModuleList:
    """Load YOLOv8n backbone layers (layers 0-9: Conv stack + SPPF)."""
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.pt")
    return nn.ModuleList(list(yolo.model.model[:10]))


def _adapt_first_conv(backbone: nn.ModuleList, in_channels: int, strategy: str = "physics_split") -> None:
    """Replace the first Conv2d to accept *in_channels* spectral bands.

    Uses physics-informed multispectral weight initialization.
    """
    first_conv = backbone[0].conv
    new_conv = adapt_multispectral_first_conv(first_conv, in_channels, strategy)
    backbone[0].conv = new_conv


class _IntermediateFeatureExtractor(nn.Module):
    """Wraps the YOLO backbone to yield features at specified layer indices.

    Forward returns a dict mapping layer index to its output tensor so that
    the decoder can access skip-connection features at multiple scales.
    """

    def __init__(self, backbone: nn.ModuleList, indices: tuple[int, ...]) -> None:
        super().__init__()
        self.backbone = backbone
        self.indices = set(indices)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        feats: dict[int, torch.Tensor] = {}
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.indices:
                feats[i] = x
        return feats


class _BottleneckViT(nn.Module):
    """Vision Transformer block applied to the bottleneck feature map.

    Includes LayerNorm before attention, dropout in attention and FFN,
    and residual connections with pre-norm architecture.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = _DEFAULT_BOTTLENECK_DIM,
        nhead: int = _VIT_NHEAD,
        num_layers: int = _VIT_LAYERS,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.proj_in = nn.Conv2d(in_channels, dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.norm_pre = nn.LayerNorm(dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-norm: LayerNorm before attention and FFN
        )
        self.stack = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.norm_post = nn.LayerNorm(dim)
        self.proj_out = nn.Conv2d(dim, in_channels, 1)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x) + self.pos
        b, c, hh, ww = h.shape
        tokens = h.flatten(2).transpose(1, 2)  # (B, S, dim)
        tokens = self.norm_pre(tokens)
        tokens = self.stack(tokens)
        tokens = self.norm_post(tokens)
        h = tokens.transpose(1, 2).view(b, self.dim, hh, ww)
        h = self.dropout(h)
        return self.proj_out(h)


class _SpatialBiGRU(nn.Module):
    """Lightweight bidirectional GRU over raster-scanned spatial tokens.

    GRU has 25% fewer parameters than LSTM and trains better on small datasets.
    Dropout applied between recurrent steps and output projection.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = _GRU_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden)
        self.gru = nn.GRU(
            hidden,
            hidden,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # dropout between layers (only 1 layer here)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(hidden * 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, S, C)
        tokens = self.proj(tokens)
        tokens, _ = self.gru(tokens)
        tokens = self.dropout(tokens)
        tokens = self.out_proj(tokens)
        return tokens.transpose(1, 2).view(b, c, h, w)


class _Decoder(nn.Module):
    """Five-stage bilinear decoder with skip connections (8x8 -> 256x256)."""

    def __init__(self, in_ch: int, skip_chs: tuple[int, ...], out_ch: int) -> None:
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_chs[0], in_ch // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch // 4 + skip_chs[1], out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, skips[0]], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, skips[1]], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up4(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up5(x)
        return x


class YoloViTLSTM(nn.Module):
    """YOLO backbone + ViT spatial attention + BiGRU sequential context.

    Produces per-class logits of shape ``(B, 11, 256, 256)`` compatible with
    the existing CE+Dice training pipeline.

    Key regularization knobs:
    - dropout: applied in ViT (attention + FFN) and BiGRU bridge (default 0.2)
    - bottleneck_dim: ViT/GRU token dimension (default 128, was 256)
    """

    # Skip connections ordered by decoder stage: layer 3 (stride 8, 32x32,
    # 64ch) for the first cat, layer 1 (stride 4, 64x64, 32ch) for the
    # second.  Bottleneck is after layer 9 (256ch, 8x8).
    _SKIP_LAYERS = (3, 1)
    _SKIP_CHS = (64, 32)
    _BOTTLENECK_CH = 256

    def __init__(
        self,
        in_channels: int = 11,
        classes: int = N_CLASSES,
        pretrained: bool = True,
        dropout: float = _DEFAULT_DROPOUT,
        bottleneck_dim: int = _DEFAULT_BOTTLENECK_DIM,
        weight_init_strategy: str = "physics_split",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.bottleneck_dim = bottleneck_dim

        backbone = _load_yolo_backbone()
        if pretrained:
            _adapt_first_conv(backbone, in_channels, weight_init_strategy)
        else:
            backbone[0].conv = nn.Conv2d(
                in_channels,
                backbone[0].conv.out_channels,
                kernel_size=backbone[0].conv.kernel_size,
                stride=backbone[0].conv.stride,
                padding=backbone[0].conv.padding,
                bias=backbone[0].conv.bias is not None,
            )
        self.encoder = _IntermediateFeatureExtractor(backbone, self._SKIP_LAYERS + (9,))
        self.vit = _BottleneckViT(
            self._BOTTLENECK_CH,
            dim=bottleneck_dim,
            dropout=dropout,
        )
        self.gru = _SpatialBiGRU(
            self._BOTTLENECK_CH,
            hidden=bottleneck_dim // 2,
            dropout=dropout,
        )
        self.decoder = _Decoder(self._BOTTLENECK_CH, self._SKIP_CHS, 64)
        self.head = nn.Conv2d(64, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        # Bottleneck: ViT then BiGRU
        bn = feats[9]
        bn = self.vit(bn)
        bn = self.gru(bn)
        # Decode with skip connections
        skips = [feats[i] for i in self._SKIP_LAYERS]
        return self.head(self.decoder(bn, skips))

    def count_parameters(self) -> dict:
        """Return parameter counts for profiling."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "m_params": round(total / 1e6, 2),
        }


def build_yolo_vit_lstm(
    in_channels: int = 11,
    classes: int = N_CLASSES,
    pretrained: bool = True,
    dropout: float = _DEFAULT_DROPOUT,
    bottleneck_dim: int = _DEFAULT_BOTTLENECK_DIM,
    weight_init_strategy: str = "physics_split",
) -> YoloViTLSTM:
    """Build the YOLO + ViT + BiGRU variant with configurable regularization."""
    return YoloViTLSTM(
        in_channels=in_channels,
        classes=classes,
        pretrained=pretrained,
        dropout=dropout,
        bottleneck_dim=bottleneck_dim,
        weight_init_strategy=weight_init_strategy,
    )