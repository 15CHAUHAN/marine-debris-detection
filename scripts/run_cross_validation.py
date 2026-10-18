"""Run N-fold scene-disjoint cross-validation on any segmentation variant.

Uses GroupKFold (group=scene) on train+val patches; the official test split
is held out entirely and never touched during CV.

Per fold:
  1. Build train/val loaders using build_loaders()
  2. Train the variant for N epochs using the Trainer
  3. Record the best-epoch validation metrics: mean IoU plus the Marine
     Debris class metrics (F1, IoU, precision, recall)

At the end aggregates mean/std/min/max across all folds and writes:
  artifacts/cv/<variant>/cv_summary.json

The best fold (by val_debris_f1, then best_mean_iou) is exported to
artifacts/{variant}/best.pth for production deployment, and a manifest
is written to artifacts/{variant}/model_selection_manifest.json.

Usage:
  python scripts/run_cross_validation.py --variant baseline --epochs 20
  python scripts/run_cross_validation.py --variant attention --epochs 20 --folds 5
  python scripts/run_cross_validation.py --variant yolo_vit_lstm --epochs 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.augment import TrainTransform
from src.data.factory import build_loaders
from src.data.ingestion import MaridaDataset
from src.data.splits import assert_scene_disjoint, load_official_splits, scene_grouped_folds
from src.models.registry import VARIANTS, resolve_variant
from src.train.trainer import TrainConfig, Trainer

# fold-result column -> aggregation key stem
# "best_mean_iou" keeps the legacy stem mIoU (dashboard compatibility).
METRIC_STEMS = {
    "best_mean_iou": "mIoU",
    "val_debris_f1": "debris_f1",
    "val_debris_iou": "debris_iou",
    "val_precision": "precision",
    "val_recall": "recall",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--encoder-weights", choices=["imagenet_physics", "imagenet_replicate", "seco_s2", "random"], default="imagenet_physics",
                   help="First-convolution weight initialization strategy for pretrained encoders")
    p.add_argument("--scheduler", choices=["cosine_warmup", "plateau", "cosine", "none"],
                   default="cosine_warmup",
                   help="LR scheduler. Default: cosine_warmup (3-epoch warmup + cosine decay).")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--lambda-lovasz", type=float, default=0.0, help="Weight for Lovász-Softmax loss term in compound loss")
    return p.parse_args()


def compute_class_weights(marida: MaridaDataset, patch_ids: list[str]) -> torch.Tensor:
    counts = np.zeros(12, dtype=np.int64)
    for pid in patch_ids:
        counts += np.bincount(marida.read_mask(pid).reshape(-1), minlength=12)[:12]
    labeled = counts[1:].sum()
    weights = ((labeled + 1.0) / (counts[1:] + 1.0)) ** 0.5
    weights = weights / weights.min()
    return torch.from_numpy(weights.astype(np.float32))


def build_fold_record(
    fold_idx: int,
    train_size: int,
    val_size: int,
    result: dict,
) -> dict:
    """Map a Trainer.run() result to one JSON-serializable fold record."""
    best_metrics = result.get("best_metrics") or {}
    return {
        "fold": fold_idx,
        "train_size": train_size,
        "val_size": val_size,
        "best_epoch": result["best_epoch"],
        # legacy dashboard key; equals val mean IoU at the best epoch.
        "best_mean_iou": result["best_mean_iou"],
        "val_debris_f1": best_metrics.get("f1_class_1_marine_debris"),
        "val_debris_iou": best_metrics.get("iou_class_1_marine_debris"),
        "val_precision": best_metrics.get("precision_class_1_marine_debris"),
        "val_recall": best_metrics.get("recall_class_1_marine_debris"),
    }


def _aggregate(values: np.ndarray) -> dict:
    """mean/std/min/max with nan-awareness plus a nan count."""
    as_float = np.asarray(values, dtype=np.float64)
    finite = as_float[~np.isnan(as_float)]
    if finite.size == 0:
        return {
            "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"), "n_nan": int(np.isnan(as_float).sum()),
        }
    return {
        "mean": round(float(np.mean(finite)), 5),
        "std": round(float(np.std(finite)), 5),
        "min": round(float(np.min(finite)), 5),
        "max": round(float(np.max(finite)), 5),
        "n_nan": int(np.isnan(as_float).sum()),
    }


def to_json_safe(obj):
    """Recursively replace NaN floats with None so cv_summary.json stays strict JSON."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj


def aggregate_fold_metrics(fold_results: list[dict]) -> dict:
    """mean/std/min/max per metric column plus per-column nan counts.

    Keys follow ``mean_val_<stem>`` / ``std_val_<stem>`` / ... with stems from
    METRIC_STEMS. ``mean_val_mIoU`` stays the legacy dashboard key.
    """
    summary: dict = {}
    for column, stem in METRIC_STEMS.items():
        values = np.array([r.get(column, float("nan")) for r in fold_results], dtype=np.float64)
        agg = _aggregate(values)
        summary[f"mean_val_{stem}"] = agg["mean"]
        summary[f"std_val_{stem}"] = agg["std"]
        summary[f"min_val_{stem}"] = agg["min"]
        summary[f"max_val_{stem}"] = agg["max"]
        summary[f"n_nan_val_{stem}"] = agg["n_nan"]
    return summary


def run_cv(
    variant: str,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    folds: int = 10,
    device: torch.device | None = None,
    out_dir: Path | None = None,
    encoder_weights: str = "imagenet_physics",
    scheduler: str = "cosine_warmup",
    warmup_epochs: int = 3,
    min_lr: float = 1e-6,
    lambda_lovasz: float = 0.5,
) -> dict:
    """Run scene-disjoint CV for one variant and return the summary dict.

    Writes ``cv_summary.json`` into ``out_dir`` (default
    ``artifacts/cv/<variant>``) as a side effect.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir) if out_dir else ROOT / "artifacts" / "cv" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}, variant={variant}, folds={folds}, epochs={epochs}")
    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    assert_scene_disjoint(splits, marida)

    trainval_ids = splits["train"] + splits["val"]
    print(f"CV on {len(trainval_ids)} train+val patches ({len(splits['test'])} test held out)")

    cv_folds = scene_grouped_folds(trainval_ids, marida, n_folds=folds, seed=42)
    print(f"Generated {len(cv_folds)} scene-disjoint folds")
    for i, (tr, va) in enumerate(cv_folds):
        print(f"  fold {i+1}: train={len(tr)}, val={len(va)}")

    fold_results: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        fold_train_ids = [trainval_ids[i] for i in train_idx]
        fold_val_ids = [trainval_ids[i] for i in val_idx]
        print(f"\n=== Fold {fold_idx + 1}/{len(cv_folds)} ===")
        print(f"  train={len(fold_train_ids)}, val={len(fold_val_ids)}")

        class_weights = compute_class_weights(marida, fold_train_ids)
        config = TrainConfig(
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            run_dir=out_dir,
            run_name=f"fold_{fold_idx + 1}",
            class_weights=class_weights,
            scheduler=scheduler,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
            lambda_lovasz=lambda_lovasz,
        )
        model = resolve_variant(variant, weight_init_strategy=encoder_weights)
        loaders = build_loaders(
            marida,
            {"train": fold_train_ids, "val": fold_val_ids, "test": []},
            batch_size=batch_size,
            train_transform=TrainTransform(),
        )
        trainer = Trainer(model, config, device)
        result = trainer.run(loaders["train"], loaders["val"])
        record = build_fold_record(
            fold_idx + 1, len(fold_train_ids), len(fold_val_ids), result
        )
        fold_results.append(record)
        print(
            f"  fold {fold_idx+1}: best_epoch={record['best_epoch']}, "
            f"val mIoU={record['best_mean_iou']:.4f}, debris F1={record['val_debris_f1']}, "
            f"debris IoU={record['val_debris_iou']}, P={record['val_precision']}, R={record['val_recall']}"
        )

    summary = aggregate_fold_metrics(fold_results)
    summary["variant"] = variant
    summary["folds"] = len(cv_folds)
    summary["epochs"] = epochs
    summary["fold_results"] = fold_results

    # Identify best fold: primary=val_debris_f1, secondary=best_mean_iou
    valid_folds = [r for r in fold_results if r.get("val_debris_f1") is not None]
    if valid_folds:
        best_fold = max(valid_folds, key=lambda f: (f.get("val_debris_f1", 0.0), f.get("best_mean_iou", 0.0)))
        best_fold_idx = best_fold["fold"]

        # Write model selection manifest (CV only; do NOT overwrite production best.pth)
        manifest = {
            "source": "10-fold-cross-validation",
            "selected_fold": best_fold_idx,
            "selection_metric": "val_debris_f1",
            "val_debris_f1": best_fold.get("val_debris_f1"),
            "val_mean_iou": best_fold.get("best_mean_iou"),
            "fold_checkpoint": str(fold_ckpt.relative_to(ROOT)),
            "cv_mean_f1": summary.get("mean_val_debris_f1"),
            "cv_std_f1": summary.get("std_val_debris_f1"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        manifest_path = ROOT / "artifacts" / variant / "model_selection_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        print(
            f"\nBest fold: {best_fold_idx} "
            f"(Debris F1={best_fold.get('val_debris_f1', 0.0):.4f})"
        )
        print(f"Manifest saved to {manifest_path}")
        print("Note: Production checkpoint must be trained via scripts/train_one.py on the full official train split.")

    summary_path = out_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(to_json_safe(summary), indent=2))
    print(
        f"\nCV complete ({variant}): mIoU = {summary['mean_val_mIoU']} ± {summary['std_val_mIoU']}, "
        f"debris F1 = {summary['mean_val_debris_f1']} ± {summary['std_val_debris_f1']}"
    )
    print(f"Summary saved to {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    run_cv(args.variant, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, folds=args.folds, encoder_weights=args.encoder_weights, scheduler=args.scheduler, warmup_epochs=args.warmup_epochs, min_lr=args.min_lr, lambda_lovasz=args.lambda_lovasz)


if __name__ == "__main__":
    main()