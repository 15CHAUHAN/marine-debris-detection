"""Tests for the cross-validation helper functions in scripts/."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import run_all_cv
from scripts.run_cross_validation import (
    _aggregate,
    build_fold_record,
    aggregate_fold_metrics,
    to_json_safe,
)


def _result(best_iou=0.5, epoch=12, f1=0.4, iou=0.3, p=0.6, r=0.2):
    return {
        "best_epoch": epoch,
        "best_mean_iou": best_iou,
        "best_metrics": {
            "f1_class_1_marine_debris": f1,
            "iou_class_1_marine_debris": iou,
            "precision_class_1_marine_debris": p,
            "recall_class_1_marine_debris": r,
        },
    }


def test_build_fold_record_maps_best_metrics():
    record = build_fold_record(3, 100, 20, _result())
    assert record["fold"] == 3
    assert record["train_size"] == 100
    assert record["val_size"] == 20
    assert record["best_epoch"] == 12
    assert record["best_mean_iou"] == 0.5
    assert record["val_debris_f1"] == 0.4
    assert record["val_debris_iou"] == 0.3
    assert record["val_precision"] == 0.6
    assert record["val_recall"] == 0.2


def test_build_fold_record_missing_best_metrics_yields_none():
    record = build_fold_record(1, 50, 10, {"best_epoch": 5, "best_mean_iou": 0.2})
    assert record["best_mean_iou"] == 0.2
    assert record["val_debris_f1"] is None


def test_aggregate_fold_metrics_nan_aware():
    fold_results = [
        {"best_mean_iou": 0.5, "val_debris_f1": 0.4, "val_debris_iou": 0.3,
         "val_precision": 0.6, "val_recall": 0.2},
        {"best_mean_iou": 0.6, "val_debris_f1": float("nan"), "val_debris_iou": 0.5,
         "val_precision": 0.8, "val_recall": 0.4},
        {"best_mean_iou": 0.7, "val_debris_f1": 0.8, "val_debris_iou": 0.7,
         "val_precision": 0.9, "val_recall": 0.6},
    ]
    s = aggregate_fold_metrics(fold_results)
    assert s["mean_val_mIoU"] == pytest.approx(0.6)
    assert s["std_val_mIoU"] == pytest.approx(np.std([0.5, 0.6, 0.7]), abs=1e-5)
    assert s["min_val_mIoU"] == pytest.approx(0.5)
    assert s["max_val_mIoU"] == pytest.approx(0.7)
    assert s["mean_val_debris_f1"] == pytest.approx(0.6)
    assert s["n_nan_val_debris_f1"] == 1
    assert s["min_val_debris_f1"] == pytest.approx(0.4)
    assert s["max_val_debris_f1"] == pytest.approx(0.8)
    assert s["n_nan_val_debris_iou"] == 0


def test_aggregate_all_nan_reports_nan_mean():
    agg = _aggregate(np.array([np.nan, np.nan]))
    assert np.isnan(agg["mean"])
    assert np.isnan(agg["std"])
    assert agg["n_nan"] == 2


def test_to_json_safe_converts_nan_to_none():
    safe = to_json_safe({"a": float("nan"), "b": {"c": float("nan")}, "d": 1, "e": [float("nan")]})
    assert safe["a"] is None
    assert safe["b"]["c"] is None
    assert safe["e"] == [None]
    assert safe["d"] == 1


def _fake_summary(values):
    summary = {}
    for stem, value in values.items():
        summary[f"mean_val_{stem}"] = value
        summary[f"std_val_{stem}"] = 0.05
        summary[f"min_val_{stem}"] = value - 0.1
        summary[f"max_val_{stem}"] = value + 0.1
        summary[f"n_nan_val_{stem}"] = 1 if value != value else 0
    return summary


def test_build_comparison_table():
    base = {
        "mIoU": 0.55, "debris_f1": 0.4, "debris_iou": 0.3,
        "precision": 0.6, "recall": 0.2,
    }
    summaries = {
        "baseline": _fake_summary(base),
        "attention": _fake_summary({**base, "mIoU": 0.6, "debris_f1": 0.5}),
    }
    cmp = run_all_cv.build_comparison(summaries)
    assert cmp["table"][0] == ["variant", "val mIoU", "val debris_f1", "val debris_iou",
                               "val precision", "val recall"]
    assert cmp["table"][1][0] == "baseline"
    assert cmp["table"][2][0] == "attention"
    assert "0.5500" in cmp["table"][1][1]
    assert "0.5000" in cmp["table"][2][2]


def test_build_comparison_nan_cell():
    base = {
        "mIoU": 0.55, "debris_f1": float("nan"), "debris_iou": 0.3,
        "precision": 0.6, "recall": 0.2,
    }
    summaries = {"baseline": _fake_summary(base)}
    cmp = run_all_cv.build_comparison(summaries)
    assert cmp["table"][1][2] == "N/A"