"""Tests for src/mapping/heatmap.py and src/mapping/georeference.py."""
from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.data.ingestion import MaridaDataset, PatchMeta
from src.mapping.georeference import write_concentration_tiff
from src.mapping.heatmap import debris_concentration

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def marida_module():
    return MaridaDataset(ROOT)


def test_empty_debris_mask_gives_zero_concentration():
    mask = np.zeros((256, 256), dtype=np.int64)
    conc = debris_concentration(mask)
    assert conc.dtype == np.float32
    assert conc.shape == (256, 256)
    assert np.max(conc) == 0.0


def test_single_debris_pixel_concentrates_around_pixel():
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[32, 32] = 1
    conc = debris_concentration(mask)
    center = float(np.max(conc))
    assert center > 0.0
    assert float(conc[32, 32]) == pytest.approx(1.0)
    assert np.all(conc >= 0.0) and np.all(conc <= 1.0)
    neighbors = [conc[31, 32], conc[32, 31], conc[33, 32], conc[32, 33]]
    assert all(n < 1.0 and n > 0.0 for n in neighbors), "heat spreads outward"


def test_concentration_gradient_monotone():
    mask = np.zeros((128, 128), dtype=np.int64)
    mask[:, 64:] = 1
    conc = debris_concentration(mask, normalize=False)
    left = float(np.mean(conc[:, 20]))
    near_edge = float(np.mean(conc[:, 60]))
    right = float(np.mean(conc[:, 110]))
    assert left < near_edge <= right


@pytest.mark.requires_data
def test_write_concentration_tiff_matches_source_georeference(marida_module: MaridaDataset, tmp_path):
    patch_id = marida_module.ids[0]
    meta = marida_module.read_meta(patch_id)
    debris = (marida_module.read_mask(patch_id) == 1).astype(np.int64)
    conc = debris_concentration(debris)
    out = write_concentration_tiff(conc, meta, tmp_path / "map.tif")
    assert out.exists()
    with rasterio.open(out) as dst:
        assert dst.count == 1
        assert dst.crs.to_string() == meta.crs
        assert tuple(dst.transform) == pytest.approx(meta.transform)
        assert (dst.height, dst.width) == (meta.height, meta.width)
        data = dst.read(1)
        assert data.dtype == np.float32
        assert set(np.unique(data)).issubset({0.0, 1.0}) or 0.0 <= data.min() <= data.max() <= 1.0