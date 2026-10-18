"""Debris bounding-box label extraction for the YOLO detection track.

MARIDA masks are pixel annotations; the detection track derives axis-aligned
bounding boxes around connected debris blobs (raw class 1) and exports them in
YOLO txt format (class_index cx cy w h, all normalized to image dimensions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

DEBRIS_CLASS = 1  # raw mask value for Marine Debris
SIZE = 256


@dataclass(frozen=True)
class DebrisBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def debris_boxes(mask: np.ndarray, min_area: int = 1) -> list[DebrisBox]:
    """Return a bounding box per connected debris blob (row=y, col=x)."""
    debris = (mask == DEBRIS_CLASS).astype(np.uint8)
    labeled, n = ndimage.label(debris)
    boxes: list[DebrisBox] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labeled == i)
        area = int(ys.size)
        if area < min_area:
            continue
        boxes.append(DebrisBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return boxes


def to_yolo_lines(boxes: list[DebrisBox], cls: int = 0, size: int = SIZE) -> list[str]:
    """YOLO format: `cls cx cy w h` with all values normalized to [0, 1]."""
    lines = []
    for b in boxes:
        cx = (b.xmin + b.xmax + 1) / 2.0 / size
        cy = (b.ymin + b.ymax + 1) / 2.0 / size
        w = (b.xmax - b.xmin + 1) / size
        h = (b.ymax - b.ymin + 1) / size
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def rgb_composite(image: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """RGB uint8 composite from 11-band float32 image (bands 2,1,0 = B4,B3,B2).

    Non-finite values (known in scene 16PCC) are replaced by the patch median.
    """
    rgb = image[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)
    if not np.isfinite(rgb).all():
        med = np.nanmedian(rgb)
        rgb = np.where(np.isfinite(rgb), rgb, np.nan_to_num(med, nan=0.0))
    p2, p98 = np.percentile(rgb, [lo, hi])
    scale = 255.0 / max(float(p98 - p2), 1e-6)
    out = np.clip((rgb - p2) * scale, 0, 255).astype(np.uint8)
    return out