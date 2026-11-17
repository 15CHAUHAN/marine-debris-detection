"""Plastic L2A dataset ingestion (Sentinel-2 / ACOLITE L2W campaign).

Mirrors the MaridaDataset seam (``read_image`` / ``read_mask`` / ``read_scene_id``
/ ``read_meta`` / ``scene_ids`` / ``__len__`` / ``__contains__``) so that the
shared ``MaridaSegmentationDataset``, ``build_loaders`` and evaluation scripts
work against Plastic L2A without modification.

The 11 bands map 1:1 to the MARIDA band ordering:

    index 0..8  : 442, 492, 559, 665, 704, 739, 780, 833, 864  nm
    index 9..10 : 1610, 2186 nm

Ground truth is still pending for this campaign. ``read_mask`` therefore raises
``GroundTruthUnavailableError`` unless ``allow_pending=True`` (returns an
all-zero placeholder mask), and ``has_ground_truth`` reports the true state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from .encoding import encode_15to11
from .ingestion import PatchMeta, resolve_patch_path, scene_from_patch_id
from .ingestion import _percentile_normalize  # keep 2-98% normalization parity

BANDS_NM = (442, 492, 559, 665, 704, 739, 780, 833, 864, 1610, 2186)
IMAGE_SUFFIX = ".tif"
S2_PREFIX = "S2_"


class GroundTruthUnavailableError(RuntimeError):
    """Raised when a caller needs a mask for a patch that has no GT yet."""


@dataclass(frozen=True)
class PlasticL2aPatch:
    patch_id: str
    scene_id: str
    image_path: Path
    mask_path: Path | None


class PlasticL2aDataset:
    """Indexes Plastic L2A patches under a root directory.

    Layout mirrors the MARIDA tree produced by scripts/prepare_plastic_l2a.py:

        <root>/patches/S2_<scene>/S2_<patch_id>.tif        (11-band image)
        <root>/patches/S2_<scene>/S2_<patch_id>_cl.tif     (optional GT mask)
    """

    def __init__(self, root: Path, patch_ids: list[str] | None = None, allow_pending: bool = False) -> None:
        self.root = Path(root)
        self.patch_root = self.root / "patches"
        self.allow_pending = allow_pending
        ids = patch_ids if patch_ids is not None else self._discover()
        self._entries: dict[str, PlasticL2aPatch] = {}
        missing: list[str] = []
        for patch_id in ids:
            if patch_id.startswith(S2_PREFIX):
                patch_id = patch_id[len(S2_PREFIX):]
            image_path = resolve_patch_path(patch_id, self.patch_root)
            if not image_path.exists():
                missing.append(patch_id)
                continue
            base = image_path.parent / f"{S2_PREFIX}{patch_id}"
            self._entries[patch_id] = PlasticL2aPatch(
                patch_id=patch_id,
                scene_id=scene_from_patch_id(patch_id),
                image_path=image_path,
                mask_path=Path(f"{base}_cl{IMAGE_SUFFIX}") if Path(f"{base}_cl{IMAGE_SUFFIX}").exists() else None,
            )
        if missing:
            raise FileNotFoundError(f"{len(missing)} patch ids have no image file, e.g. {missing[:3]}")
        self.ids: tuple[str, ...] = tuple(self._entries)

    def _discover(self) -> list[str]:
        image_files = sorted(
            path
            for path in self.patch_root.glob(f"{S2_PREFIX}*/*{IMAGE_SUFFIX}")
            if not path.name.endswith("_cl.tif") and not path.name.endswith("_conf.tif")
        )
        return [path.name[len(S2_PREFIX):-len(IMAGE_SUFFIX)] for path in image_files]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, patch_id: str) -> bool:
        return patch_id in self._entries

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple({entry.scene_id for entry in self._entries.values()})

    @property
    def has_ground_truth(self) -> bool:
        return any(entry.mask_path is not None for entry in self._entries.values())

    def get(self, patch_id: str) -> PlasticL2aPatch:
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
        """Load and encode the class mask, return int64 (H, W) 0..11 (or all-0 default)."""
        entry = self._entries[patch_id]
        if entry.mask_path is None:
            if not self.allow_pending:
                raise GroundTruthUnavailableError(
                    f"patch {patch_id} has no ground truth yet; "
                    "construct PlasticL2aDataset(..., allow_pending=True) "
                    "to get an all-zero placeholder mask"
                )
            with rasterio.open(entry.image_path) as source:
                height, width = source.height, source.width
            return np.zeros((height, width), dtype=np.int64)
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
        return len(self._entries)