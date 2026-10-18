"""Prepare the Plastic L2A (Sentinel-2 / ACOLITE L2W) dataset as MARIDA-compatible
georeferenced patches.

Reads the ACOLITE L2W netCDF (this campaign: 20210626) whose 11 ``rhos_*`` bands
map 1:1 to the MARIDA 11-band order:

    index 0..8  : 442, 492, 559, 665, 704, 739, 780, 833, 864  nm
    index 9..10 : 1610, 2186 nm

The netCDF stores the tile south-up (row 0 at the southern edge). We re-reference
it to the standard Sentinel-2 north-up grid (EPSG:32635, 10 m, origin 399960 /
4400040 for tile T35SMD) and tile a grid of ``--grid-size`` patches around the
UAS survey site.

No ground truth exists yet for the Plastic L2A campaign. Patches are written with
their images only; a ``ground_truth_pending.txt`` marker is emitted. If a
georeferenced GT raster is later supplied via ``--gt-path``, per-patch ``_cl.tif``
masks are written as well (GT-later workflow).

Usage:
  python scripts/prepare_plastic_l2a.py
  python scripts/prepare_plastic_l2a.py --gt-path <gt.tif> --out-dir artifacts/plastic_l2a
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ACOLITE rhos band wavelengths -> MARIDA band index (0..10). Order matters.
RHOS_WAVELENGTHS = (442, 492, 559, 665, 704, 739, 780, 833, 864, 1610, 2186)

# Sentinel-2 tile T35SMD north-up geometry (10 m grid, matches SAFE MTD_TL.xml).
TILE_CRS = "EPSG:32635"
TILE_ORIGIN_X = 399960.0
TILE_ORIGIN_Y = 4400040.0
PIXEL_SIZE = 10.0
TILE_PIXELS = 10980  # 10980 x 10980 at 10 m

# Patch id scheme mirrors MARIDA: "<day>-<month>-<year>_<TILE>_<n>".
SCENE_ID = "26-6-21_T35SMD"
PATCH_RE = re.compile(r"^(\d+-\d+-\d+_[A-Z0-9]+)_(\d+)$")


def _wavelength_band_names(l2w: Path) -> dict[int, str]:
    """Return {wavelength: subdataset name} for every rhos_<w> subdataset."""
    with rasterio.open(l2w) as src:
        subs = src.subdatasets
    found: dict[int, str] = {}
    for sd in subs:
        match = re.search(r"rhos_(\d+)$", sd)
        if match:
            found[int(match.group(1))] = sd
    return found


def _north_up_window(center_col: int, center_row: int, patch_size: int):
    """Return netCDF (row_start, row_stop), (col_start, col_stop) for a patch.

    center_col/center_row are north-up SAFE pixel coordinates (col 0 = west,
    row 0 = north). The netCDF is stored south-up, so row order is reversed.
    """
    col0 = center_col - patch_size // 2
    row0_north = center_row - patch_size // 2
    nc_row0 = (TILE_PIXELS - 1) - (row0_north + patch_size - 1)
    col_stop = col0 + patch_size
    nc_row_stop = nc_row0 + patch_size
    if col0 < 0 or nc_row_stop > TILE_PIXELS or row0_north < 0 or col_stop > TILE_PIXELS:
        raise ValueError(
            f"patch outside tile: col {col0}..{col_stop}, north row "
            f"{row0_north}..{row0_north + patch_size}"
        )
    return (nc_row0, nc_row_stop), (col0, col_stop)


def _read_band_window(subdataset: str, win_rows: tuple[int, int], win_cols: tuple[int, int]) -> np.ndarray:
    """Window-read one netCDF band and flip it to north-up orientation."""
    with rasterio.open(subdataset, sharing=False) as src:
        data = src.read(1, window=(win_rows, win_cols))
    return np.flipud(np.asarray(data))


def _north_up_transform(center_col: int, center_row: int, patch_size: int) -> Affine:
    col0 = center_col - patch_size // 2
    row0 = center_row - patch_size // 2
    return Affine(
        PIXEL_SIZE, 0.0, TILE_ORIGIN_X + col0 * PIXEL_SIZE,
        0.0, -PIXEL_SIZE, TILE_ORIGIN_Y - row0 * PIXEL_SIZE,
    )


def _read_gt_window(gt_path: Path, transform: Affine, patch_size: int) -> np.ndarray:
    """Window-read a georeferenced GT raster overlapping the patch extent.

    The GT raster may be clipped to any extent; the window is computed from the
    patch's world bounds against the GT raster's own affine (same CRS assumed).
    """
    with rasterio.open(gt_path) as src:
        if src.crs is not None and src.crs.to_string() != TILE_CRS:
            raise ValueError(f"GT raster CRS {src.crs} != {TILE_CRS}")
        west, south, east, north = rasterio.windows.bounds(
            rasterio.windows.Window(0, 0, patch_size, patch_size), transform
        )
        from rasterio.windows import from_bounds

        window = from_bounds(west, south, east, north, transform=src.transform)
        window = window.round_offsets().crop(src.height, src.width)
        data = src.read(1, window=window, boundless=True, fill_value=0)
    height, width = data.shape
    if height < patch_size or width < patch_size:
        padded = np.zeros((patch_size, patch_size), dtype=np.float64)
        padded[: min(height, patch_size), : min(width, patch_size)] = data[
            : min(height, patch_size), : min(width, patch_size)
        ]
        data = padded
    return np.asarray(data)


def _emit_patch_manifest(scene_dir: Path, entries: list[dict]) -> Path:
    manifest = {
        "dataset": "plastic_l2a",
        "campaign": "20210626",
        "crs": TILE_CRS,
        "pixel_size_m": PIXEL_SIZE,
        "band_order_nm": list(RHOS_WAVELENGTHS),
        "bands": [f"rhos_{w}" for w in RHOS_WAVELENGTHS],
        "ground_truth": "pending",
        "patches": entries,
    }
    out = scene_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Plastic L2A patches.")
    parser.add_argument("--l2w-path", type=Path, default=None,
                        help="Path to the ACOLITE L2W netCDF (default: auto-discover in 20210626/).")
    parser.add_argument("--gt-path", type=Path, default=None,
                        help="Optional georeferenced GT raster; writes _cl.tif masks.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "plastic_l2a")
    parser.add_argument("--center-col", type=int, default=5893,
                        help="North-up SAFE column of the ROI center (UAS survey site).")
    parser.add_argument("--center-row", type=int, default=7875,
                        help="North-up SAFE row of the ROI center (UAS survey site).")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=4,
                        help="N x N patch grid centered on the ROI.")
    parser.add_argument("--product", choices=("rhos", "rhot"), default="rhos",
                        help="ACOLITE reflectance product to extract.")
    args = parser.parse_args()

    l2w = args.l2w_path
    if l2w is None:
        candidates = sorted((ROOT / "20210626" / "20210626_ACOLITE").rglob("S2B_MSI_*_L2W.nc"))
        if not candidates:
            raise FileNotFoundError("no ACOLITE L2W netCDF found; pass --l2w-path")
        l2w = candidates[0]
    if not l2w.exists():
        raise FileNotFoundError(l2w)

    found = _wavelength_band_names(l2w)
    prefix = f"rhos_" if args.product == "rhos" else f"rhot_"
    missing = [w for w in RHOS_WAVELENGTHS if w not in found]
    if missing:
        raise FileNotFoundError(f"missing {args.product} bands in netCDF: {missing}")

    patch_size = args.patch_size
    grid = args.grid_size
    half = grid // 2
    centers = [
        (args.center_col + (dx - half) * patch_size, args.center_row + (dy - half) * patch_size)
        for dy in range(grid)
        for dx in range(grid)
    ]
    centers = [c for c in centers if 0 <= c[0] < TILE_PIXELS and 0 <= c[1] < TILE_PIXELS]

    scene_dir = args.out_dir / "patches" / f"S2_{SCENE_ID}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for idx, (col, row) in enumerate(centers):
        patch_id = f"{SCENE_ID}_{idx}"
        image_path = scene_dir / f"S2_{patch_id}.tif"
        transform = _north_up_transform(col, row, patch_size)
        win_rows, win_cols = _north_up_window(col, row, patch_size)

        image = np.empty((len(RHOS_WAVELENGTHS), patch_size, patch_size), dtype=np.float32)
        for bi, w in enumerate(RHOS_WAVELENGTHS):
            band = _read_band_window(found[w], win_rows, win_cols)
            image[bi] = band.astype(np.float32)
            del band

        with rasterio.open(
            image_path, "w", driver="GTiff", height=patch_size, width=patch_size,
            count=len(RHOS_WAVELENGTHS), dtype="float32", crs=TILE_CRS, transform=transform,
        ) as dst:
            dst.write(image)
        del image

        entry = {
            "patch_id": patch_id,
            "scene_id": SCENE_ID,
            "center_safe_col": col,
            "center_safe_row": row,
            "crs": TILE_CRS,
            "transform": tuple(transform),
            "has_gt": False,
        }

        if args.gt_path is not None:
            mask = _read_gt_window(args.gt_path, transform, patch_size)
            mask_path = scene_dir / f"S2_{patch_id}_cl.tif"
            with rasterio.open(
                mask_path, "w", driver="GTiff", height=patch_size, width=patch_size,
                count=1, dtype="float32", crs=TILE_CRS, transform=transform,
            ) as dst:
                dst.write(mask.astype(np.float32), 1)
            entry["has_gt"] = True

        entries.append(entry)
        print(f"wrote {image_path.name}  center=({col},{row})  transform={tuple(transform)}")

    manifest_path = _emit_patch_manifest(scene_dir, entries)
    (scene_dir / "ground_truth_pending.txt").write_text(
        "No ground truth available yet for the 20210626 Plastic L2A campaign.\n"
        "Re-run with --gt-path <raster> to emit _cl.tif masks.\n"
    )
    n_gt = sum(1 for e in entries if e["has_gt"])
    print(f"\nPrepared {len(entries)} patches (grid {grid}x{grid}) in {scene_dir}")
    print(f"Manifest: {manifest_path}  (GT raster written for {n_gt} patches)")

    if n_gt == 0:
        print("NOTE: ground truth is pending; training directly is not meaningful yet.")


if __name__ == "__main__":
    main()