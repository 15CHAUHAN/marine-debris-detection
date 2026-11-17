"""Tests for src/data/ingestion.py (patch reading and masking)."""
from pathlib import Path

import numpy as np
import pytest

from src.data.ingestion import MaridaDataset, scene_from_patch_id

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dataset():
    return MaridaDataset(ROOT)


@pytest.mark.requires_data
def test_dataset_discovers_all_patches(dataset):
    assert len(dataset) == 1381
    assert len(dataset.scene_ids) == 63


@pytest.mark.requires_data
def test_patch_index_resolves_all_ids(dataset):
    assert dataset.resolved_count() == 1381


@pytest.mark.requires_data
def test_image_shape_and_range(dataset):
    image = dataset.read_image(dataset.ids[0])
    assert image.shape == (11, 256, 256)
    assert image.dtype == np.float32
    assert image.min() >= 0.0 and image.max() <= 1.0


@pytest.mark.requires_data
def test_mask_shape_and_value_range(dataset):
    mask = dataset.read_mask(dataset.ids[0])
    assert mask.shape == (256, 256)
    assert mask.dtype == np.int64
    assert mask.min() >= 0 and mask.max() <= 11


@pytest.mark.requires_data
def test_meta_contains_georeferencing(dataset):
    meta = dataset.read_meta(dataset.ids[0])
    assert meta.band_count == 11
    assert meta.width == 256 and meta.height == 256
    assert meta.crs.startswith("EPSG:")


@pytest.mark.requires_data
def test_all_masks_are_encoded_within_range(dataset):
    for patch_id in dataset.ids:
        mask = dataset.read_mask(patch_id)
        assert mask.min() >= 0 and mask.max() <= 11, patch_id


def test_scene_extraction_helpers():
    assert scene_from_patch_id("1-12-19_48MYU_0") == "1-12-19_48MYU"
    assert scene_from_patch_id("11-6-18_16PCC_47") == "11-6-18_16PCC"
    with pytest.raises(ValueError):
        scene_from_patch_id("not-a-valid-id")


@pytest.mark.requires_data
def test_missing_patch_id_raises():
    dataset = MaridaDataset(ROOT)
    with pytest.raises(KeyError):
        dataset.read_image("99-99-99_ZZZZZ_0")