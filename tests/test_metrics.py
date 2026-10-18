"""Tests for src/evaluate/metrics.py."""
import numpy as np
import pytest
import torch

from src.data.encoding import IGNORE_INDEX
from src.evaluate.metrics import aggregate_metrics, evaluate_predictions


def test_perfect_prediction_scores_perfect_metrics():
    target = torch.zeros(2, 8, 8, dtype=torch.int64)
    target[0, :, :] = 0  # marine debris channel
    target[1, :, :] = 6  # marine water channel
    logits = torch.randn(2, 11, 8, 8) * 0.01
    for batch in range(2):
        logits[batch, target[batch, 0, 0], :, :] = 10.0
    metrics = evaluate_predictions(logits, target)
    assert metrics["accuracy"] == 1.0
    assert metrics["mean_iou"] == 1.0
    assert metrics["f1_class_1_marine_debris"] == 1.0


def test_unlabeled_pixels_are_ignored():
    target = torch.full((4, 8, 8), IGNORE_INDEX, dtype=torch.int64)
    target[0, 0:4, 0:4] = 0  # only 16 labeled pixels on batch 0
    target[1, 0:2, 0:2] = 6  # only 4 labeled pixels on batch 1
    logits = torch.zeros(4, 11, 8, 8)
    logits[0, 0, 0:4, 0:4] = 10.0
    logits[1, 6, 0:2, 0:2] = 10.0
    metrics = evaluate_predictions(logits, target)
    assert metrics["accuracy"] == 1.0, "unlabeled pixels must not drag the score down"


def test_aggregate_sums_confusion_matrices():
    target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.int64)
    target[0, :, :] = 4
    logits = torch.randn(1, 11, 4, 4) * 0.01
    logits[0, 4, :, :] = 10.0
    m1 = evaluate_predictions(logits, target)
    m2 = evaluate_predictions(logits, target)
    aggregated = aggregate_metrics([m1, m2])
    assert aggregated["confusion_matrix"].shape == (11, 11)
    total = m1["confusion_matrix"] + m2["confusion_matrix"]
    np.testing.assert_array_equal(aggregated["confusion_matrix"], total)
    assert aggregated["accuracy"] == 1.0


def test_metrics_sanity_ranges():
    rng = np.random.default_rng(0)
    logits = torch.from_numpy(rng.normal(size=(2, 11, 16, 16))).float()
    target = torch.from_numpy(rng.integers(0, 12, size=(2, 16, 16))).long()
    metrics = evaluate_predictions(logits, target)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["f1_class_1_marine_debris"] != float("nan")


def test_debris_precision_recall_iou_perfect():
    target = torch.zeros(1, 4, 4, dtype=torch.int64)
    logits = torch.randn(1, 11, 4, 4) * 0.01
    logits[0, 0, :, :] = 10.0
    metrics = evaluate_predictions(logits, target)
    assert metrics["precision_class_1_marine_debris"] == 1.0
    assert metrics["recall_class_1_marine_debris"] == 1.0
    assert metrics["iou_class_1_marine_debris"] == 1.0


def test_debris_precision_recall_iou_known_values():
    target = torch.full((1, 4, 4), 6, dtype=torch.int64)
    target[0, 0:2, 0:2] = 0
    logits = torch.full((1, 11, 4, 4), -1.0)
    logits[0, 6] = 10.0
    pred_debris = [(0, 0), (0, 1), (1, 0), (1, 2)]
    for r, c in pred_debris:
        logits[0, 6, r, c] = 0.0
        logits[0, 0, r, c] = 10.0
    metrics = evaluate_predictions(logits, target)
    # tp = 3 ((0,0),(0,1),(1,0)), fp = 1 ((1,2)), fn = 1 ((1,1) missed)
    assert metrics["precision_class_1_marine_debris"] == pytest.approx(3 / 4, abs=1e-3)
    assert metrics["recall_class_1_marine_debris"] == pytest.approx(3 / 4, abs=1e-3)
    assert metrics["iou_class_1_marine_debris"] == pytest.approx(3 / 5, abs=1e-3)


def test_debris_metrics_offered_but_absent_is_nan():
    target = torch.full((1, 4, 4), 6, dtype=torch.int64)
    logits = torch.full((1, 11, 4, 4), -1.0)
    logits[0, 6] = 10.0
    metrics = evaluate_predictions(logits, target)
    assert np.isnan(metrics["precision_class_1_marine_debris"])
    assert np.isnan(metrics["recall_class_1_marine_debris"])
    assert np.isnan(metrics["iou_class_1_marine_debris"])