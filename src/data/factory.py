"""DataLoader factory: the seam between data modules and training."""
from __future__ import annotations

import random
from typing import Callable

import numpy as np
from torch.utils.data import DataLoader

from .augment import identity, wrap
from .dataset import DEFAULT_MIN_DEBRIS_RATIO, DebrisBalancedBatchSampler, MaridaSegmentationDataset
from .ingestion import MaridaDataset
from .preprocessing import clahe_multispectral_chw


def _clahe_transform(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply CLAHE to validation/test images (H, W, C) -> (H, W, C)."""
    return clahe_multispectral_chw(image.transpose(2, 0, 1)).transpose(1, 2, 0), mask


def build_loaders(
    marida: MaridaDataset,
    splits: dict[str, list[str]],
    batch_size: int,
    num_workers: int = 0,
    train_transform: Callable | None = None,
    shuffle: bool = True,
    seed: int = 42,
    min_debris_ratio: float | None = None,
    filter_empty_dominance: float | None = None,
    apply_clahe_to_val_test: bool = False,
) -> dict[str, DataLoader]:
    """Return {'train', 'val', 'test'} DataLoaders.

    train_transform is wrapped; val and test use raw percentile-normalized
    inputs without CLAHE (to match pure inference environment). Debris-balancing
    options apply to the train loader only:

      * filter_empty_dominance: drop patches with > X% single-class dominance
        and zero debris pixels from the training set. A value <= 0 disables
        filtering.
      * min_debris_ratio: enforce at least this fraction of debris-positive
        patches in EVERY training batch, otherwise training falls back to a
        plain shuffled loader (the final batch may be smaller).
    """
    train_ds = MaridaSegmentationDataset(
        marida,
        splits["train"],
        transform_fn=wrap(train_transform),
        filter_empty_dominance=None if filter_empty_dominance is None or filter_empty_dominance <= 0 else filter_empty_dominance,
    )
    val_transform = _clahe_transform if apply_clahe_to_val_test else None
    test_transform = _clahe_transform if apply_clahe_to_val_test else None
    val_ds = MaridaSegmentationDataset(marida, splits["val"], transform_fn=val_transform)
    test_ds = MaridaSegmentationDataset(marida, splits["test"], transform_fn=test_transform)

    def _loader(ds, do_shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=do_shuffle,
            num_workers=num_workers,
            persistent_workers=False,
        )

    if min_debris_ratio is not None and min_debris_ratio > 0 and train_ds.n_debris_positive > 0:
        sampler = DebrisBalancedBatchSampler(
            data_len=len(train_ds),
            positive_indices=train_ds.debris_indices,
            batch_size=batch_size,
            min_debris_ratio=min_debris_ratio,
            generator=random.Random(seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            persistent_workers=False,
        )
    else:
        train_loader = _loader(train_ds, shuffle)

    return {
        "train": train_loader,
        "val": _loader(val_ds, False),
        "test": _loader(test_ds, False),
    }