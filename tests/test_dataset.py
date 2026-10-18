"""Tests for src/data/augment.py and src/data/dataset.py."""
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.augment import TrainTransform, build_train_transforms, identity, wrap
from src.data.dataset import MaridaSegmentationDataset
from src.data.ingestion import MaridaDataset

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def marida():
    return MaridaDataset(ROOT)


def test_train_transforms_are_stable_shapes(marida):
    patch_id = marida.ids[0]
    image = marida.read_image(patch_id)
    mask = marida.read_mask(patch_id)
    image_hwc = np.transpose(image, (1, 2, 0))
    pipeline = TrainTransform()
    for _ in range(5):
        aug_image, aug_mask = wrap(pipeline)(image_hwc, mask)
        assert aug_image.shape == image_hwc.shape
        assert aug_mask.shape == mask.shape
        assert set(np.unique(aug_mask)).issubset(set(np.unique(mask)))


def test_clahe_is_applied_only_to_the_pipeline(marida):
    patch_id = marida.ids[0]
    image = marida.read_image(patch_id)
    image_hwc = np.transpose(image, (1, 2, 0))
    from src.data.augment import apply_clahe
    enhanced = apply_clahe(image_hwc)
    assert enhanced.shape == image_hwc.shape
    assert enhanced.dtype == np.float32


def test_clahe_sanitizes_nan_pixels():
    from src.data.augment import apply_clahe
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[2, 2, :] = np.nan
    image[4, 4, :] = np.inf
    enhanced = apply_clahe(image)
    assert np.isfinite(enhanced).all()
    assert enhanced.dtype == np.float32


def test_identity_transform_is_passthrough(marida):
    patch_id = marida.ids[0]
    image = marida.read_image(patch_id)
    mask = marida.read_mask(patch_id)
    image_hwc = np.transpose(image, (1, 2, 0))
    out_image, out_mask = identity(image_hwc, mask)
    np.testing.assert_array_equal(out_image, image_hwc)
    np.testing.assert_array_equal(out_mask, mask)


def test_dataset_batch_shapes(marida):
    ids = list(marida.ids[:16])
    ds = MaridaSegmentationDataset(marida, ids, transform_fn=identity)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    image, mask = next(iter(loader))
    assert tuple(image.shape) == (8, 11, 256, 256)
    assert tuple(mask.shape) == (8, 256, 256)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.int64


def test_dataset_with_train_transforms(marida):
    ids = list(marida.ids[:8])
    ds = MaridaSegmentationDataset(marida, ids, transform_fn=wrap(TrainTransform()))
    image, mask = ds[0]
    assert tuple(image.shape) == (11, 256, 256)
    assert mask.min().item() >= 0 and mask.max().item() <= 11


def test_dataset_count(marida):
    ids = list(marida.ids[:100])
    ds = MaridaSegmentationDataset(marida, ids)
    assert len(ds) == 100