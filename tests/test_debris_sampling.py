"""Tests for debris-balanced sampling and dominance filtering (src/data/)."""
import random
from pathlib import Path

import numpy as np
import pytest

from src.data.augment import identity
from src.data.dataset import DebrisBalancedBatchSampler, MaridaSegmentationDataset
from src.data.ingestion import MaridaDataset

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def marida():
    return MaridaDataset(ROOT)


@pytest.fixture(scope="module")
def clean_ds(marida):
    return MaridaSegmentationDataset(marida, list(marida.ids), transform_fn=identity)


@pytest.fixture(scope="module")
def filtered_ds(marida):
    return MaridaSegmentationDataset(
        marida, list(marida.ids), transform_fn=identity, filter_empty_dominance=0.95
    )


# ---------------------------------------------------------------------------
# DebrisBalancedBatchSampler
# ---------------------------------------------------------------------------

def test_sampler_enforces_min_debris_ratio_every_batch():
    sampler = DebrisBalancedBatchSampler(
        data_len=100,
        positive_indices=list(range(20)),
        batch_size=8,
        min_debris_ratio=0.45,
        generator=random.Random(7),
    )
    assert sampler.n_pos_in_batch == 4
    assert len(sampler) == 13
    for batch in sampler:
        assert len(batch) == 8
        positives = sum(1 for idx in batch if idx < 20)
        assert positives >= 4


def test_sampler_seeded_is_deterministic():
    def make():
        return DebrisBalancedBatchSampler(
            data_len=50,
            positive_indices=[0, 5, 9],
            batch_size=6,
            min_debris_ratio=0.5,
            generator=random.Random(123),
        )
    first = [list(b) for b in make()]
    second = [list(b) for b in make()]
    assert first == second


def test_sampler_sample_without_replacement_when_pool_large():
    pool = list(range(50))
    sampler = DebrisBalancedBatchSampler(
        data_len=50,
        positive_indices=pool,
        batch_size=8,
        min_debris_ratio=0.5,
        replacement=False,
    )
    drawn = sampler._sample_indices(pool, 4)
    assert len(drawn) == 4
    assert len(set(drawn)) == 4
    assert all(i in pool for i in drawn)


def test_sampler_falls_back_to_replacement_when_pool_too_small():
    pool = [0, 1]
    sampler = DebrisBalancedBatchSampler(
        data_len=10,
        positive_indices=pool,
        batch_size=8,
        min_debris_ratio=0.5,
        replacement=False,
    )
    drawn = sampler._sample_indices(pool, 4)
    assert len(drawn) == 4
    assert set(drawn).issubset({0, 1})
    assert len(set(drawn)) < 4  # duplicates required


def test_sampler_all_positive_dataset_tops_up_negatives_from_positives():
    sampler = DebrisBalancedBatchSampler(
        data_len=6,
        positive_indices=list(range(6)),
        batch_size=4,
        min_debris_ratio=0.5,
        generator=random.Random(1),
    )
    for batch in sampler:
        assert len(batch) == 4
        assert all(0 <= i < 6 for i in batch)
        assert len(set(batch)) >= 1


def test_sampler_rejects_duplicate_positive_indices():
    with pytest.raises(ValueError):
        DebrisBalancedBatchSampler(
            data_len=10,
            positive_indices=[1, 1, 2],
            batch_size=4,
            min_debris_ratio=0.5,
        )


# ---------------------------------------------------------------------------
# Dominance filtering
# ---------------------------------------------------------------------------

def _id_to_stats(ds):
    ds._ensure_analysis()
    return {pid: (nd, dom) for pid, nd, dom in zip(ds.ids, ds._debris_counts, ds._dominance)}


@pytest.mark.requires_data
def test_filter_is_disabled_when_threshold_zero_or_below(marida, clean_ds):
    for thr in (0.0, -1.0):
        ds = MaridaSegmentationDataset(
            marida, list(marida.ids), transform_fn=identity, filter_empty_dominance=thr
        )
        assert len(ds) == len(clean_ds)
        assert ds.n_dropped == 0


@pytest.mark.requires_data
def test_dominance_filter_only_drops_zero_debris_high_dominance(clean_ds, filtered_ds):
    clean_ids = set(clean_ds.ids)
    dropped_ids = clean_ids - set(filtered_ds.ids)
    stats = _id_to_stats(clean_ds)
    assert len(dropped_ids) == filtered_ds.n_dropped
    for pid in dropped_ids:
        nd, dom = stats[pid]
        assert nd == 0
        assert dom > 0.95
    assert len(dropped_ids) > 0


@pytest.mark.requires_data
def test_dominance_filter_preserves_all_debris_positive_patches(clean_ds, filtered_ds):
    assert filtered_ds.n_debris_positive == clean_ds.n_debris_positive
    kept_pos = set(filtered_ds.ids) & set(np.asarray(clean_ds.ids)[clean_ds.debris_positive_mask])
    assert len(kept_pos) == clean_ds.n_debris_positive
    assert filtered_ds.n_debris_positive + filtered_ds.n_debris_negative == len(filtered_ds)