"""PyTorch Dataset bridging MARIDA files to training batches."""
from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .encoding import N_CLASSES, to_train_targets
from .ingestion import MaridaDataset

# 11-class value 1 is Marine Debris (CONTEXT.md class mapping).
CLASS1_11 = 1
DEFAULT_MIN_DEBRIS_RATIO = 0.45


class MaridaSegmentationDataset(Dataset):
    """Returns (image: float32[11, 256, 256] in [0,1], target: int64[256, 256]).

    apply_transform takes (H, W, C) image and (H, W) target and returns the
    same shapes. Targets use the training encoding: classes 1..11 become
    0..10 and unlabeled pixels become IGNORE_INDEX (11).

    Optional debris-targeted behaviour (train-only):
      * filter_empty_dominance (float | None): drop patches whose labeled
        pixels are > `filter_empty_dominance` of a single class AND that
        contain ZERO Marine Debris pixels. Only uninformative negative patches
        are removed; no class-encoding semantics change.
      * Min-debris-ratio batch oversampling via `debris_indices` (lazily
        computed by reading every mask once) combined with a
        DebrisBalancedBatchSampler.
    """

    def __init__(
        self,
        marida: MaridaDataset,
        patch_ids: list[str],
        transform_fn: Callable | None = None,
        filter_empty_dominance: float | None = None,
    ) -> None:
        self.marida = marida
        self.ids = list(patch_ids)
        self.transform_fn = transform_fn
        self.filter_empty_dominance = filter_empty_dominance
        self._analyzed: bool | None = None
        self._is_debris: np.ndarray | None = None
        self._debris_counts: np.ndarray | None = None
        self._dominance: np.ndarray | None = None
        self._n_labeled: np.ndarray | None = None
        self._n_dropped = 0
        if filter_empty_dominance is not None:
            self._ensure_analysis()

    def __len__(self) -> int:
        return len(self.ids)

    def _ensure_analysis(self) -> None:
        """Inspect each patch mask once and cache per-patch statistics.

        If filter_empty_dominance is set (and > 0), patches with >
        filter_empty_dominance single-class dominance and zero debris pixels are
        dropped from self.ids. A value <= 0 disables filtering.
        """
        filter_thr = self.filter_empty_dominance
        filter_active = filter_thr is not None and filter_thr > 0
        if self._analyzed is not None:
            return
        original_ids = list(self.ids)
        is_debris: list[bool] = []
        debris_counts: list[int] = []
        dominance: list[float] = []
        n_labeled: list[int] = []
        keep: list[bool] = []
        for pid in original_ids:
            mask = self.marida.read_mask(pid)  # 11-class values 0..11
            labeled = mask[mask > 0]
            nlab = int(labeled.size)
            nd = int(np.count_nonzero(labeled == CLASS1_11))
            dom = 0.0
            if nlab > 0:
                counts = np.bincount(labeled, minlength=N_CLASSES + 1)
                dom = float(counts.max()) / float(nlab)
            is_debris.append(nd > 0)
            debris_counts.append(nd)
            dominance.append(dom)
            n_labeled.append(nlab)
            drop = filter_active and nd == 0 and dom > filter_thr
            keep.append(not drop)
        keep_mask = np.asarray(keep, dtype=bool)
        self.ids = [pid for pid, k in zip(original_ids, keep_mask) if k]
        self._is_debris = np.asarray(is_debris, dtype=bool)[keep_mask]
        self._debris_counts = np.asarray(debris_counts, dtype=np.int64)[keep_mask]
        self._dominance = np.asarray(dominance, dtype=np.float64)[keep_mask]
        self._n_labeled = np.asarray(n_labeled, dtype=np.int64)[keep_mask]
        self._n_dropped = int((~keep_mask).sum())
        self._analyzed = True

    @property
    def _analyzed_masks(self) -> bool:
        return self._analyzed is not None

    @property
    def n_dropped(self) -> int:
        self._ensure_analysis()
        return self._n_dropped

    @property
    def debris_counts(self) -> np.ndarray:
        self._ensure_analysis()
        return self._debris_counts

    @property
    def debris_positive_mask(self) -> np.ndarray:
        self._ensure_analysis()
        return self._is_debris

    @property
    def n_debris_positive(self) -> int:
        self._ensure_analysis()
        return int(self._is_debris.sum())

    @property
    def n_debris_negative(self) -> int:
        self._ensure_analysis()
        return int((~self._is_debris).sum())

    @property
    def debris_indices(self) -> list[int]:
        """Indices (into len(self)) of debris-positive patches."""
        self._ensure_analysis()
        return [i for i in range(len(self.ids)) if self._is_debris[i]]

    @property
    def debris_negative_indices(self) -> list[int]:
        """Indices (into len(self)) of debris-negative background patches."""
        self._ensure_analysis()
        return [i for i in range(len(self.ids)) if not self._is_debris[i]]

    def debris_balanced_sampler(
        self,
        batch_size: int,
        min_debris_ratio: float = DEFAULT_MIN_DEBRIS_RATIO,
        replacement: bool = True,
        generator: random.Random | None = None,
    ) -> "DebrisBalancedBatchSampler":
        return DebrisBalancedBatchSampler(
            data_len=len(self),
            positive_indices=self.debris_indices,
            batch_size=batch_size,
            min_debris_ratio=min_debris_ratio,
            replacement=replacement,
            generator=generator,
        )

    def __getitem__(self, index: int):
        patch_id = self.ids[index]
        image = self.marida.read_image(patch_id)
        mask = to_train_targets(self.marida.read_mask(patch_id))
        image_hwc = np.transpose(image, (1, 2, 0))
        if self.transform_fn is not None:
            image_hwc, mask = self.transform_fn(image_hwc, mask)
        mask = mask.astype(np.int64)
        image_chw = np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)), dtype=np.float32)
        return torch.from_numpy(image_chw), torch.from_numpy(np.ascontiguousarray(mask))


class DebrisBalancedBatchSampler:
    """Yields batches that always contain >= min_debris_ratio debris-positive samples.

    Each batch draws n_pos = ceil(min_debris_ratio * batch_size) samples from
    the debris-positive indices and the remainder from the debris-negative
    indices, sampling with replacement (default). This enforces the debris ratio
    per batch rather than only in expectation.
    """

    def __init__(
        self,
        data_len: int,
        positive_indices: list[int],
        batch_size: int,
        min_debris_ratio: float = DEFAULT_MIN_DEBRIS_RATIO,
        replacement: bool = True,
        generator: random.Random | None = None,
    ) -> None:
        self.data_len = int(data_len)
        self.batch_size = int(batch_size)
        self.replacement = replacement
        self.rng = generator if generator is not None else random.Random()
        positive_set = set(int(i) for i in positive_indices)
        if len(positive_set) != len(positive_indices):
            raise ValueError("positive_indices must be unique")
        self.positive_indices = [int(i) for i in positive_indices]
        self.negative_indices = [i for i in range(self.data_len) if i not in positive_set]
        n_pos = len(self.positive_indices)
        ratio = float(min_debris_ratio)
        self.min_debris_ratio = ratio
        self.n_pos_in_batch = int(math.ceil(ratio * self.batch_size)) if n_pos > 0 else 0
        self.n_pos_in_batch = min(self.n_pos_in_batch, self.batch_size, n_pos)
        self.n_neg_in_batch = self.batch_size - self.n_pos_in_batch
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.n_batches = int(math.ceil(self.data_len / self.batch_size))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            if self.n_pos_in_batch > 0:
                pos = self._sample_indices(
                    self.positive_indices,
                    self.n_pos_in_batch,
                )
            else:
                pos = []
            if self.negative_indices:
                neg = self._sample_indices(
                    self.negative_indices,
                    self.n_neg_in_batch,
                )
            elif self.n_neg_in_batch > 0:
                neg = (pos * ((self.n_neg_in_batch + len(pos) - 1) // max(len(pos), 1)))[: self.n_neg_in_batch] if pos else []
            else:
                neg = []
            batch = pos + neg
            self.rng.shuffle(batch)
            yield batch

    def _sample_indices(self, pool: list[int], k: int) -> list[int]:
        if self.replacement or len(pool) < k:
            sampled = self.rng.choices(pool, k=k)
        else:
            sampled = self.rng.sample(pool, k=k)
        return list(sampled)


class SyntheticSegmentationDataset(Dataset):
    """In-memory segmentation dataset (e.g. GAN pseudo-labeled samples).

    images: (N, C, H, W) float32 in [0, 1]; targets: (N, H, W) int64 training
    targets ({0..10}, unlabeled already mapped to IGNORE_INDEX). transform_fn
    follows the same (H, W, C) convention as MaridaSegmentationDataset.
    """

    def __init__(
        self,
        images: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        transform_fn: Callable | None = None,
    ) -> None:
        self.images = images if isinstance(images, np.ndarray) else images.numpy()
        self.targets = targets if isinstance(targets, np.ndarray) else targets.numpy()
        if self.images.shape[0] != self.targets.shape[0]:
            raise ValueError("images and targets must have the same leading dimension")
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        image = self.images[index]
        mask = self.targets[index].astype(np.int64)
        image_hwc = np.transpose(image, (1, 2, 0))
        if self.transform_fn is not None:
            image_hwc, mask = self.transform_fn(image_hwc, mask)
        image_chw = np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)), dtype=np.float32)
        return torch.from_numpy(image_chw), torch.from_numpy(np.ascontiguousarray(mask))