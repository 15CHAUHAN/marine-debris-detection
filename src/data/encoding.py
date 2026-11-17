"""15-class to 11-class encoding for MARIDA class masks.

The full MARIDA label set ships 15 thematic classes. The benchmark set
aggregates Waves (12), Cloud Shadows (13), Wakes (14) and Mixed Water (15)
into the Marine Water (7) super-class. Class 0 (unlabeled) stays 0.
See CONTEXT.md for the full mapping table.
"""
from __future__ import annotations

import numpy as np

N_CLASSES = 11
_N_CLASSES_15 = 15

# Unlabeled pixels are flagged with this value in training targets. The model
# outputs 11 logits (classes 0..10); unlabeled pixels therefore use index 11
# which is skipped by the losses via ignore_index.
IGNORE_INDEX = 11

# Maps 11-class value (0..11, where 0 means unlabeled) to a training target
# in {0..10, IGNORE_INDEX}. Class c in 1..11 becomes c - 1.
CLASS11_TO_TRAIN = np.full(N_CLASSES + 1, IGNORE_INDEX, dtype=np.int64)
CLASS11_TO_TRAIN[1:] = np.arange(N_CLASSES)

# Index i holds the 11-class value for 15-class digital number i (0..15).
CLASS15_TO_11 = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 7, 7, 7, 7],
    dtype=np.int64,
)

CLASS_NAMES = {
    1: "Marine Debris",
    2: "Dense Sargassum",
    3: "Sparse Sargassum",
    4: "Natural Organic Material",
    5: "Ship",
    6: "Clouds",
    7: "Marine Water",
    8: "Sediment-Laden Water",
    9: "Foam",
    10: "Turbid Water",
    11: "Shallow Water",
}


def encode_15to11(mask: np.ndarray) -> np.ndarray:
    """Map a 15-class mask (whole-number values 0..15) to the 11-class set.

    Accepts any numeric dtype; floating inputs are rounded first. Returns an
    int64 mask with the same shape and values in {0..11}.
    """
    if mask.dtype.kind == "f":
        values = np.round(mask).astype(np.int64)
    else:
        values = mask.astype(np.int64)
    np.clip(values, 0, _N_CLASSES_15, out=values)
    return CLASS15_TO_11[values]


def to_train_targets(mask11: np.ndarray) -> np.ndarray:
    """Convert an 11-class mask (0 unlabeled, 1..11 classes) to training
    targets: classes become 0..10 and unlabeled pixels become IGNORE_INDEX
    (11). Losses and metrics skip IGNORE_INDEX pixels."""
    return CLASS11_TO_TRAIN[mask11]


def class_weights(mask11: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Inverse-frequency class weights (classes 0..10) for the CE loss.

    Weights use Laplace-smoothed inverse frequencies and are normalized so the
    most frequent labeled class gets weight 1.0. power controls the amount of
    rebalancing. Unlabeled pixels (raw value 0) are excluded from the counts.
    """
    counts = np.bincount(mask11.reshape(-1), minlength=N_CLASSES + 1)[: N_CLASSES + 1]
    labeled = counts[1:].sum()
    if labeled == 0:
        return np.ones(N_CLASSES, dtype=np.float32)
    inv = (labeled + 1.0) / (counts[1:] + 1.0)
    weights = inv ** power
    weights = weights / weights.min()
    return weights.astype(np.float32)