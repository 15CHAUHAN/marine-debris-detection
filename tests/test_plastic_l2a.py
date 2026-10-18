"""Tests for src/data/plastic_l2a.py (Plastic L2A dataset seam) and
scripts/prepare_plastic_l2a.py (patch extraction from the ACOLITE netCDF).

The suite runs against the real prepared artifacts (artifacts/plastic_l2a)
when present, otherwise against a synthetic mini-layout so the seam contract
is always exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ingestion import PatchMeta, scene_from_patch_id
from src.data.plastic_l2a import (
    BANDS_NM,
    IMAGE_SUFFIX,
    S2_PREFIX,
    GroundTruthUnavailableError,
    PlasticL2aDataset,
)


@pytest.fixture(scope="session")
def real_root() -> Path:
    root = ROOT / "artifacts" / "plastic_l2a"
    return root if (root / "patches").exists() else None


def _synthetic_root(tmp_path: Path, n: int = 3, with_gt: bool = False) -> Path:
    scene = f"{S2_PREFIX}26-6-21_T35SMD"
    scene_dir = tmp_path / "patches" / scene
    scene_dir.mkdir(parents=True)
    transform = Affine(10.0, 0.0, 457610.0, 0.0, -10.0, 4322570.0)
    for i in range(n):
        array = np.random.RandomState(i).rand(11, 256, 256).astype(np.float32)
        path = scene_dir / f"{S2_PREFIX}26-6-21_T35SMD_{i}{IMAGE_SUFFIX}"
        with rasterio.open(
            path, "w", driver="GTiff", height=256, width=256,
            count=11, dtype="float32", crs="EPSG:32635", transform=transform,
        ) as dst:
            dst.write(array)
        if with_gt:
            gt = np.full((256, 256), i % 3, dtype=np.float32)
            with rasterio.open(
                scene_dir / f"{S2_PREFIX}26-6-21_T35SMD_{i}_cl{IMAGE_SUFFIX}",
                "w", driver="GTiff", height=256, width=256,
                count=1, dtype="float32", crs="EPSG:32635", transform=transform,
            ) as dst:
                dst.write(gt, 1)
    return tmp_path


def test_band_order_matches_marida_spec():
    assert BANDS_NM == (442, 492, 559, 665, 704, 739, 780, 833, 864, 1610, 2186)
    assert len(BANDS_NM) == 11


def test_scene_id_parses():
    assert scene_from_patch_id("26-6-21_T35SMD_0") == "26-6-21_T35SMD"


class TestSeam:
    def test_real_dataset_if_present(self, real_root):
        if real_root is None:
            pytest.skip("plastic_l2a artifacts not prepared")
        ds = PlasticL2aDataset(real_root, allow_pending=True)
        assert len(ds) >= 1
        assert ds.scene_ids == ("26-6-21_T35SMD",)
        image = ds.read_image(ds.ids[0])
        assert image.shape == (11, 256, 256)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        meta = ds.read_meta(ds.ids[0])
        assert meta.crs == "EPSG:32635"
        assert len(meta.transform) == 9

    def test_synthetic_image_read(self, tmp_path):
        root = _synthetic_root(tmp_path)
        ds = PlasticL2aDataset(root, allow_pending=True)
        image = ds.read_image(ds.ids[1])
        assert image.shape == (11, 256, 256)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0

    def test_read_mask_pending_raises_by_default(self, tmp_path):
        root = _synthetic_root(tmp_path, with_gt=False)
        ds = PlasticL2aDataset(root)
        with pytest.raises(GroundTruthUnavailableError):
            ds.read_mask(ds.ids[0])

    def test_read_mask_allow_pending_returns_zeros(self, tmp_path):
        root = _synthetic_root(tmp_path, with_gt=False)
        ds = PlasticL2aDataset(root, allow_pending=True)
        mask = ds.read_mask(ds.ids[0])
        assert mask.shape == (256, 256)
        assert mask.dtype == np.int64
        assert mask.max() == 0

    def test_read_mask_with_gt(self, tmp_path):
        root = _synthetic_root(tmp_path, with_gt=True)
        ds = PlasticL2aDataset(root)
        assert ds.has_ground_truth is True
        mask = ds.read_mask(ds.ids[0])
        assert mask.shape == (256, 256)
        assert set(np.unique(mask)) <= set(range(12))

    def test_has_ground_truth_flag(self, tmp_path):
        assert PlasticL2aDataset(_synthetic_root(tmp_path, with_gt=False)).has_ground_truth is False

    def test_meta_contains_georeferencing(self, tmp_path):
        root = _synthetic_root(tmp_path)
        ds = PlasticL2aDataset(root, allow_pending=True)
        meta = ds.read_meta(ds.ids[0])
        assert isinstance(meta, PatchMeta)
        assert "EPSG" in meta.crs
        assert meta.band_count == 11

    def test_contains_and_scene_ids(self, tmp_path):
        root = _synthetic_root(tmp_path, n=2)
        ds = PlasticL2aDataset(root, allow_pending=True)
        assert "26-6-21_T35SMD_0" in ds
        assert "26-6-21_T35SMD_9" not in ds
        assert ds.scene_ids == ("26-6-21_T35SMD",)
        assert len(ds) == 2

    def test_missing_ids_raise(self, tmp_path):
        root = _synthetic_root(tmp_path, n=1)
        with pytest.raises(FileNotFoundError):
            PlasticL2aDataset(root, patch_ids=["26-6-21_T35SMD_0", "99-1-1_ZZZZZ_0"])


class TestPrepareHelpers:
    def test_north_up_window_flips_rows(self):
        from scripts.prepare_plastic_l2a import _north_up_window

        # Patch whose bottom row touches the southern edge (row 10979).
        # netCDF stores the southern-most row at index 0, so rows invert.
        win_rows, win_cols = _north_up_window(center_col=128, center_row=10852, patch_size=256)
        assert win_cols == (0, 256)
        assert win_rows == (0, 256)

        win_rows, win_cols = _north_up_window(center_col=256, center_row=256, patch_size=256)
        assert win_cols == (128, 384)
        assert win_rows == (10980 - 1 - 383, 10980 - 1 - 383 + 256)

    def test_north_up_window_out_of_tile_raises(self):
        from scripts.prepare_plastic_l2a import _north_up_window

        with pytest.raises(ValueError):
            _north_up_window(center_col=-50, center_row=300, patch_size=256)

    def test_north_up_transform_origin(self):
        from scripts.prepare_plastic_l2a import _north_up_transform

        tr = _north_up_transform(center_col=5893, center_row=7875, patch_size=256)
        assert tr.c == 399960.0 + (5893 - 128) * 10.0
        assert tr.f == 4400040.0 - (7875 - 128) * 10.0
        assert tr.a == 10.0 and tr.e == -10.0


class TestPrepareIntegration:
    def test_prepared_artifacts_georeferenced(self, real_root):
        if real_root is None:
            pytest.skip("plastic_l2a artifacts not prepared")
        manifest = real_root / "patches" / f"{S2_PREFIX}26-6-21_T35SMD" / "manifest.json"
        assert manifest.exists(), "run scripts/prepare_plastic_l2a.py first"
        import json

        data = json.loads(manifest.read_text())
        assert data["crs"] == "EPSG:32635"
        assert data["bands"] == [f"rhos_{w}" for w in BANDS_NM]
        assert data["ground_truth"] == "pending"
        assert len(data["patches"]) >= 1


class TestDashboardSeam:
    def test_plastic_l2a_registered_in_source_codes(self):
        from dashboard.app import SOURCE_CODES

        assert SOURCE_CODES.get("Plastic L2A (Sentinel-2, 2021-06-26)") == "plastic_l2a"

    def test_plastic_l2a_is_georeferenced_source_kind(self):
        from dashboard import app

        assert app.get_pixel_area({"geo": {"transform": (10, 0, 0, 0, -10, 0)}},
                                  "plastic_l2a", None) == 100.0
        meta = app.get_geo({"geo": {"transform": (10, 0, 457610, 0, -10, 4322570),
                                    "crs": "EPSG:32635", "width": 256, "height": 256}},
                           "plastic_l2a", None)
        assert meta is not None and meta["crs"] == "EPSG:32635"

    def test_analyze_detection_plastic_l2a_with_dummy_model(self, tmp_path):
        import torch

        from dashboard import app

        if (ROOT / "artifacts" / "plastic_l2a" / "patches").exists() is False:
            pytest.skip("plastic_l2a artifacts not prepared")

        class _DummyModel(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 11, x.shape[2], x.shape[3])
                logits[:, 0] = 3.0
                return logits

        app.load_model = lambda variant: (None, None)  # force untrained fallback
        app.load_untrained_model = lambda variant: _DummyModel().eval()

        det = app.analyze_detection("plastic_l2a", "baseline", b"", "26-6-21_T35SMD_4",
                                    2.0, debris_threshold=0.05)
        assert det["label"] == "26-6-21_T35SMD_4"
        assert det["geo"]["crs"] == "EPSG:32635"
        assert det["ground_truth_pending"] is True
        assert det["rgb"].shape == (256, 256, 3)