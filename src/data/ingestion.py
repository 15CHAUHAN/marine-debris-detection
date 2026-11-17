"""MARIDA patch ingestion.

Loads the MARIDA GeoTiff patches as normalized float32 tensors and exposes
one deep seam (MaridaDataset) on top of the raw file tree. All downstream
modules (trainers, evaluators, mappers) consume this module only; they never
touch the GeoTiff file layout themselves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from .encoding import encode_15to11

_SCENE_RE = re.compile(r"^(\d+-\d+-\d+_[A-Z0-9]+)")
_IMAGE_SUFFIX = ".tif"
_S2_PREFIX = "S2_"


@dataclass(frozen=True)
class MaridaPatch:
    patch_id: str
    scene_id: str
    image_path: Path
    mask_path: Path
    confidence_path: Path


@dataclass(frozen=True)
class PatchMeta:
    patch_id: str
    scene_id: str
    crs: str
    transform: tuple
    band_count: int
    width: int
    height: int


def scene_from_patch_id(patch_id: str) -> str:
    match = _SCENE_RE.match(patch_id)
    if not match:
        raise ValueError(f"cannot extract scene id from {patch_id!r}")
    return match.group(1)


def resolve_patch_path(patch_id: str, patch_root: Path) -> Path:
    scene = scene_from_patch_id(patch_id)
    return patch_root / f"{_S2_PREFIX}{scene}" / f"{_S2_PREFIX}{patch_id}{_IMAGE_SUFFIX}"


def _percentile_normalize(image: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    normalized = np.zeros_like(image, dtype=np.float32)
    for band in range(image.shape[0]):
        values = image[band]
        low, high = np.percentile(values, [pmin, pmax])
        if high - low < 1e-6:
            low, high = 0.0, float(values.max()) if values.max() > 0 else 1.0
        normalized[band] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return normalized


class MaridaDataset:
    """Indexes every MARIDA patch under a root directory.

    The file layout is validated once at construction. If patch_ids is None,
    the whole tree is discovered; otherwise the given ids are resolved and any
    missing file raises FileNotFoundError at construction time.
    """

    def __init__(self, root: Path, patch_ids: list[str] | None = None) -> None:
        self.root = Path(root)
        self.patch_root = self.root / "patches"
        ids = patch_ids if patch_ids is not None else self._discover()
        self._entries: dict[str, MaridaPatch] = {}
        missing: list[str] = []
        for patch_id in ids:
            image_path = resolve_patch_path(patch_id, self.patch_root)
            if not image_path.exists():
                missing.append(patch_id)
                continue
            base = image_path.parent / f"{_S2_PREFIX}{patch_id}"
            self._entries[patch_id] = MaridaPatch(
                patch_id=patch_id,
                scene_id=scene_from_patch_id(patch_id),
                image_path=image_path,
                mask_path=Path(f"{base}_cl{_IMAGE_SUFFIX}"),
                confidence_path=Path(f"{base}_conf{_IMAGE_SUFFIX}"),
            )
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} patch ids have no image file, e.g. {missing[:3]}"
            )
        self.ids: tuple[str, ...] = tuple(self._entries)

    def _discover(self) -> list[str]:
        image_files = sorted(
            path
            for path in self.patch_root.glob(f"{_S2_PREFIX}*/*{_IMAGE_SUFFIX}")
            if not path.name.endswith("_cl.tif") and not path.name.endswith("_conf.tif")
        )
        return [path.name[len(_S2_PREFIX):-len(_IMAGE_SUFFIX)] for path in image_files]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, patch_id: str) -> bool:
        return patch_id in self._entries

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple({entry.scene_id for entry in self._entries.values()})

    def get(self, patch_id: str) -> MaridaPatch:
        return self._entries[patch_id]

    def read_scene_id(self, patch_id: str) -> str:
        return self._entries[patch_id].scene_id

    def read_image(self, patch_id: str, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
        """Load one patch, return float32 (C, H, W) normalized to [0, 1]."""
        entry = self._entries[patch_id]
        with rasterio.open(entry.image_path) as source:
            image = source.read()
        return _percentile_normalize(image.astype(np.float32), pmin, pmax)

    def read_mask(self, patch_id: str) -> np.ndarray:
        """Load and encode the class mask, return int64 (H, W) with values 0..11."""
        entry = self._entries[patch_id]
        with rasterio.open(entry.mask_path) as source:
            raw = source.read(1)
        return encode_15to11(raw)

    def read_meta(self, patch_id: str) -> PatchMeta:
        entry = self._entries[patch_id]
        with rasterio.open(entry.image_path) as source:
            crs = source.crs.to_string() if source.crs else str(None)
            transform = tuple(source.transform)
            band_count = source.count
            height = source.height
            width = source.width
        return PatchMeta(
            patch_id=entry.patch_id,
            scene_id=entry.scene_id,
            crs=crs,
            transform=transform,
            band_count=band_count,
            width=width,
            height=height,
        )

    def resolved_count(self) -> int:
        """Number of patches whose image file exists on disk."""
        return len(self._entries)