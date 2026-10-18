"""Tests for src/data/splits.py."""
from pathlib import Path

import pytest

from src.data.ingestion import MaridaDataset
from src.data.splits import (
    assert_scene_disjoint,
    load_official_splits,
    scene_grouped_folds,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dataset():
    return MaridaDataset(ROOT)


@pytest.fixture(scope="module")
def splits(dataset):
    official = load_official_splits(ROOT / "splits")
    assert_scene_disjoint(official, dataset)
    return official


def test_official_split_counts(splits):
    assert len(splits["train"]) == 694
    assert len(splits["val"]) == 328
    assert len(splits["test"]) == 359


def test_splits_cover_every_patch(dataset, splits):
    union = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(union) == 1381
    assert set(dataset.ids) == union


def test_no_patch_leakage_between_splits(splits):
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))


def test_all_ids_resolve_in_dataset(dataset, splits):
    for split in splits.values():
        for patch_id in split:
            assert patch_id in dataset


def test_scene_grouped_folds_are_scene_disjoint(dataset):
    ids = dataset.ids
    folds = scene_grouped_folds(list(ids), dataset, n_folds=10, seed=42)
    assert len(folds) == 10
    scene_train, scene_val = set(), set()
    for train_idx, val_idx in folds:
        train_scenes = {dataset.read_scene_id(ids[i]) for i in train_idx}
        val_scenes = {dataset.read_scene_id(ids[i]) for i in val_idx}
        assert not (train_scenes & val_scenes), "scene leakage inside a fold"
        scene_train |= train_scenes
        scene_val |= val_scenes
    assert len(scene_train) == 63 and len(scene_val) == 63, "all scenes eventually used"


def test_scene_grouped_folds_cover_all_indices(dataset):
    ids = dataset.ids
    folds = scene_grouped_folds(list(ids), dataset, n_folds=10, seed=42)
    seen = set()
    for train_idx, val_idx in folds:
        seen |= set(train_idx.tolist()) | set(val_idx.tolist())
    assert seen == set(range(len(ids)))