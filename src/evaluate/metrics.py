"""Pixel-level segmentation metrics (IoU, F1, accuracy, confusion matrix)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from src.data.encoding import CLASS_NAMES, IGNORE_INDEX, N_CLASSES


def _flatten_valid_pixels(
    pred_labels: np.ndarray, target_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only pixels where the ground truth is labeled (target != IGNORE_INDEX)."""
    valid = target_labels != IGNORE_INDEX
    return pred_labels[valid], target_labels[valid]


def confusion_matrix(
    pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int = N_CLASSES
) -> np.ndarray:
    """num_classes x num_classes confusion matrix over labeled pixels only."""
    pred_labels = torch.argmax(pred_logits, dim=1).detach().cpu().numpy().astype(np.int64)
    target_labels = target.detach().cpu().numpy().astype(np.int64)
    pred_valid, target_valid = _flatten_valid_pixels(pred_labels, target_labels)
    return sklearn_confusion_matrix(
        target_valid, pred_valid, labels=list(range(num_classes))
    )


def _from_confusion_matrix(cm: np.ndarray) -> dict:
    """Derive per-class and aggregate metrics from a confusion matrix."""
    num_classes = cm.shape[0]
    iou_per_class: dict[str, float] = {}
    f1_per_class: dict[str, float] = {}
    precision_per_class: dict[str, float] = {}
    recall_per_class: dict[str, float] = {}
    for channel in range(num_classes):
        row = cm[channel]
        col = cm[:, channel]
        tp = float(row[channel])
        fn = float(row.sum() - tp)  # Row sum minus TP = ground-truth false negatives
        fp = float(col.sum() - tp)  # Column sum minus TP = predicted false positives
        union = tp + fp + fn
        iou = tp / union if union > 0 else float("nan")
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        name = CLASS_NAMES[channel + 1]
        iou_per_class[name] = round(iou, 4)
        f1_per_class[name] = round(f1, 4)
        precision_per_class[name] = round(precision, 4)
        recall_per_class[name] = round(recall, 4)
    present = {k: v for k, v in iou_per_class.items() if not np.isnan(v)}
    accuracy = float(cm.diagonal().sum()) / float(cm.sum()) if cm.sum() > 0 else float("nan")
    return {
        "mean_iou": round(float(np.mean(list(present.values()))) if present else float("nan"), 4),
        "iou_per_class": iou_per_class,
        "f1_per_class": f1_per_class,
        "f1_class_1_marine_debris": f1_per_class["Marine Debris"],
        "iou_class_1_marine_debris": iou_per_class["Marine Debris"],
        "precision_class_1_marine_debris": precision_per_class["Marine Debris"],
        "recall_class_1_marine_debris": recall_per_class["Marine Debris"],
        "accuracy": round(accuracy, 4),
    }


def evaluate_predictions(
    pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int = N_CLASSES
) -> dict:
    """Evaluate one batch of predictions against ground truth."""
    cm = confusion_matrix(pred_logits, target, num_classes=num_classes)
    derived = _from_confusion_matrix(cm)
    derived["confusion_matrix"] = cm
    return derived


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    """Sum per-batch confusion matrices, then derive a single metric summary."""
    total_cm = sum(metrics["confusion_matrix"] for metrics in metrics_list)
    summary = _from_confusion_matrix(total_cm)
    summary["confusion_matrix"] = total_cm
    return summary