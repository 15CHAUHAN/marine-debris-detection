"""CNN + BiLSTM patch classifier: the best-effort BiLSTM stand-in.

MARIDA is single-frame imagery with no temporal axis, so the sequential
dimension is taken to be the spectral one: each Sentinel-2 band is encoded by
a shared CNN into a feature vector, the 11 band-vectors form a sequence that a
bidirectional LSTM reads in both directions, and a small MLP classifies the
patch as Marine Debris present / absent. This captures cross-band dependencies
that a pure per-pixel CNN cannot.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BANDS = 11
HIDDEN = 128


class BandwiseCNN(nn.Module):
    """Shared conv stack applied independently to each band (1 -> 128-d vector)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return h.mean(dim=(2, 3))


class CNNBiLSTMClassifier(nn.Module):
    """Binary (per-patch) Marine Debris classifier over the band sequence."""
    N_CLASSES = 2

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.cnn = BandwiseCNN()
        self.lstm = nn.LSTM(128, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, bands, hh, ww = x.shape
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        frame = x.view(b * bands, 1, 64, 64)
        feats = self.cnn(frame).view(b, bands, -1)
        out, _ = self.lstm(feats)
        pooled = out.mean(dim=1)
        return self.head(pooled)


class PatchClassDataset(torch.utils.data.Dataset):
    """Per-patch binary labels: `any debris pixel` in the target (=channel 0)."""

    def __init__(self, marida, patch_ids, transform_fn=None):
        from src.data.dataset import MaridaSegmentationDataset

        self.base = MaridaSegmentationDataset(marida, patch_ids, transform_fn=transform_fn)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        label = int((target == 0).any())
        return image, label