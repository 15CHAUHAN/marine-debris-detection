"""Georeferenced output for concentration maps.

Writes a concentration array back to a GeoTiff using the original patch's CRS
and affine transform so the result overlays 1:1 with the source Sentinel-2
patch in any GIS viewer (QGIS, ArcGIS, GDAL).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from src.data.ingestion import PatchMeta


def write_concentration_tiff(
    concentration: np.ndarray,
    meta: PatchMeta,
    output_path: Path,
) -> Path:
    """Write a concentration map as a single-band GeoTiff (float32).

    Uses the source patch's CRS and affine transform directly, and the same
    pixel dimensions, so the output registers with the source GeoTiff.
    Unlabeled no-data areas (worthless here) are kept as the source values.

    Args:
        concentration: float32 (H, W) concentration values in [0, 1].
        meta: PatchMeta from the source patch (crs, transform, size).
        output_path: destination file, parent directory is created as needed.

    Returns:
        The output path as written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = concentration.shape[:2]
    transform = Affine(*meta.transform)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=meta.crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(concentration.astype(np.float32), 1)
    return output_path