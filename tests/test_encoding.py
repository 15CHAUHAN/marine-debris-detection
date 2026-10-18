"""Tests for src/data/encoding.py (15-to-11 class encoding)."""
import numpy as np
import pytest

from src.data.encoding import (
    CLASS_NAMES,
    N_CLASSES,
    class_weights,
    encode_15to11,
    to_train_targets,
)

AGGREGATION_CASES = [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
    (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
    (12, 7), (13, 7), (14, 7), (15, 7), (0, 0),
]


@pytest.mark.parametrize("raw, expected", AGGREGATION_CASES)
def test_individual_value_mapping(raw, expected):
    mask = np.array([[raw]], dtype=np.float32)
    assert encode_15to11(mask)[0, 0] == expected


def test_whole_vector_mapping():
    raw = np.arange(0, 16, dtype=np.float32).reshape(1, 16)
    encoded = encode_15to11(raw)
    expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 7, 7, 7, 7], dtype=np.int64)
    np.testing.assert_array_equal(encoded.ravel(), expected)


def test_float_inputs_round_to_whole_numbers():
    mask = np.array([[5.4, 5.5, 5.6]], dtype=np.float32)
    encoded = encode_15to11(mask)
    np.testing.assert_array_equal(encoded[0], [5, 6, 6])


def test_output_dtype_is_int64():
    encoded = encode_15to11(np.zeros((8, 8), dtype=np.float32))
    assert encoded.dtype == np.int64


def test_out_of_range_values_are_clipped_to_domain():
    # [-3, 20] clips to [0, 15], then encodes: CLASS15_TO_11[0] = 0, [15] = 7.
    mask = np.array([[-3, 20]], dtype=np.float32)
    encoded = encode_15to11(mask)
    np.testing.assert_array_equal(encoded[0], [0, 7])


def test_class_count_and_names():
    assert N_CLASSES == 11
    assert set(CLASS_NAMES) == set(range(1, 12))
    assert CLASS_NAMES[1] == "Marine Debris"
    assert CLASS_NAMES[7] == "Marine Water"


def test_class_weights_are_aligned_with_train_targets():
    mask = np.zeros((4, 5), dtype=np.int64)
    mask[0, :] = 7    # Marine Water (train target 6): 5 px
    mask[1, :] = 7    # Marine Water: 5 px
    mask[2, :5] = 7   # Marine Water: 5 px (15 px total, common class)
    mask[3, 0:2] = 1  # Marine Debris (train target 0): 2 px, rare class
    weights = class_weights(mask, power=1.0)
    assert weights.shape == (N_CLASSES,)
    assert weights.dtype == np.float32
    assert weights.min() == 1.0, "most frequent labeled class scaled to 1.0"
    assert weights[6] == 1.0, "most frequent target (Marine Water) gets weight 1.0"
    assert weights[0] > weights[6], "rarer target (Marine Debris) gets higher weight"
    zero_excluded = class_weights(np.zeros((8, 8), dtype=np.int64))
    np.testing.assert_array_equal(zero_excluded, np.ones(N_CLASSES, dtype=np.float32))


def test_unlabeled_pixels_map_to_ignore_index():
    mask = np.array([[0, 1, 11]], dtype=np.int64)
    targets = to_train_targets(mask)
    np.testing.assert_array_equal(targets[0], [11, 0, 10])