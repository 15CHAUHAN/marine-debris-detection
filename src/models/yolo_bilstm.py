"""YOLO + BiLSTM hybrid segmentation model.

Uses the YOLOv8n backbone as a multi-scale feature extractor and a
Bidirectional LSTM neck that processes spatial feature sequences from
multiple backbone scales to capture bidirectional spatial and channel
context.  The BiLSTM output is aggregated with the original features and
decoded back to full-resolution 11-class segmentation logits.

The sequential dimension for the BiLSTM is a raster-scan spatial ordering:
each spatial position in the feature map becomes one step in the sequence,
and the BiLSTM reads both left-to-right and right-to-left across the grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.encoding import N_CLASSES
from src.models.weight_init import adapt_multispectral_first_conv

_BILSTM_HIDDEN = 128
_NECK_CH = 128


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
    """Wraps the YOLO backbone to yield features at specified layer indices."""

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


class _BiLSTMNeck(nn.Module):
    """Bidirectional LSTM applied to multi-scale spatial feature sequences.

    For each scale, features are reshaped into a spatial sequence
    (H*W steps, C channels), processed by a shared BiLSTM, and pooled
    back to the spatial grid.  Scale outputs are projected to a common
    channel dimension and summed.
    """

    def __init__(
        self,
        scale_channels: tuple[int, ...],
        hidden: int = _BILSTM_HIDDEN,
        out_ch: int = _NECK_CH,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.projs = nn.ModuleList(
            [nn.Linear(ch, hidden) for ch in scale_channels]
        )
        self.lstm = nn.LSTM(
            hidden, hidden, batch_first=True, bidirectional=True
        )
        self.out_projs = nn.ModuleList(
            [nn.Linear(hidden * 2, out_ch) for _ in scale_channels]
        )

    def forward(self, scales: list[torch.Tensor]) -> torch.Tensor:
        """Process a list of feature maps (from coarsest to finest).

        All scales are spatially resized to the coarsest resolution before
        the BiLSTM so they share the same sequence length.  The result is
        projected back to each scale's original spatial size and summed.
        """
        coarsest = scales[0]
        target_h, target_w = coarsest.shape[-2:]
        aligned = []
        for s in scales:
            if s.shape[-2:] != (target_h, target_w):
                s = F.interpolate(
                    s, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )
            aligned.append(s)

        b = aligned[0].shape[0]
        h, w = target_h, target_w
        seq_len = h * w

        agg = torch.zeros(b, _NECK_CH, h, w, device=aligned[0].device)
        for feat, proj, out_proj in zip(aligned, self.projs, self.out_projs):
            c = feat.shape[1]
            tokens = feat.permute(0, 2, 3, 1).reshape(b, seq_len, c)
            tokens = proj(tokens)
            tokens, _ = self.lstm(tokens)
            tokens = out_proj(tokens)
            grid = tokens.reshape(b, h, w, _NECK_CH).permute(0, 3, 1, 2)
            agg = agg + grid
        return agg


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

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
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


class YoloBiLSTM(nn.Module):
    """YOLO backbone + Bidirectional LSTM neck for segmentation.

    Produces per-class logits of shape ``(B, 11, 256, 256)`` compatible with
    the existing CE+Dice training pipeline.
    """

    # Multi-scale features (coarsest first for the BiLSTM neck target):
    # layer 7 (stride 32, 8x8, 256ch), layer 5 (stride 16, 128ch),
    # layer 3 (stride 8, 64ch).  Skip connections ordered by decoder stage:
    # layer 3 (32x32, 64ch) then layer 1 (64x64, 32ch).
    _SCALE_LAYERS = (7, 5, 3)
    _SCALE_CHS = (256, 128, 64)
    _SKIP_LAYERS = (3, 1)
    _SKIP_CHS = (64, 32)

    def __init__(
        self, in_channels: int = 11, classes: int = N_CLASSES, pretrained: bool = True,
        weight_init_strategy: str = "physics_split"
    ) -> None:
        super().__init__()
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
        self.encoder = _IntermediateFeatureExtractor(
            backbone, self._SCALE_LAYERS + self._SKIP_LAYERS
        )
        self.neck = _BiLSTMNeck(self._SCALE_CHS)
        self.decoder = _Decoder(_NECK_CH, self._SKIP_CHS, 64)
        self.head = nn.Conv2d(64, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        # Multi-scale BiLSTM neck (coarsest first)
        scales = [feats[i] for i in self._SCALE_LAYERS]
        neck_out = self.neck(scales)
        # Decode with skip connections from intermediate layers
        skips = [feats[i] for i in self._SKIP_LAYERS]
        return self.head(self.decoder(neck_out, skips))

    def count_parameters(self) -> dict:
        """Return parameter counts for profiling."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "m_params": round(total / 1e6, 2),
        }


def build_yolo_bilstm(
    in_channels: int = 11,
    classes: int = N_CLASSES,
    pretrained: bool = True,
    weight_init_strategy: str = "physics_split",
) -> YoloBiLSTM:
    """Build the YOLO + BiLSTM variant."""
    return YoloBiLSTM(in_channels=in_channels, classes=classes, pretrained=pretrained, weight_init_strategy=weight_init_strategy)
