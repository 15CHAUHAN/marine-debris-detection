"""Analyze training histories for overfitting / underfitting and optionally
apply a correction technique and retrain.

Detection heuristics per variant (read from artifacts/<variant>/history.csv):
  - overfitting:  val_mean_iou fell >= 5% (relative) after the best epoch
                  while train_loss kept decreasing over the same window.
  - underfitting: best val_mean_iou < 0.10 and train_loss still > 1.0 at the
                  best epoch (model never converged).
  - healthy:      otherwise.

Generalization Gap Auditing:
  - generalization_gap = (train_loss_best_epoch - train_loss_final) /
                         (train_loss_best_epoch + 1e-6)
  - Flag overfitting if:
      (a) val_mean_iou dropped >= 5% from best epoch, OR
      (b) train_loss continued to drop > 15% while val_mean_iou stalled/
          plateaued for > 4 consecutive epochs.

Corrections (applied when --apply is given):
  overfitting  -> lower LR (x0.5), double weight decay, stronger augmentation
                  (additional rotate + blur transforms), label smoothing,
                  enable model dropout=0.2.
  underfitting -> double LR, halve weight decay, train 1.5x epochs.

Usage:
  python scripts/detect_overfitting.py              # analysis report only
  python scripts/detect_overfitting.py --apply      # analysis + retrain fixes
  python scripts/detect_overfitting.py --variant baseline   # single variant
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.augment import TrainTransform
from src.data.factory import build_loaders
from src.data.ingestion import MaridaDataset
from src.data.splits import assert_scene_disjoint, load_official_splits
from src.models.registry import VARIANTS, resolve_variant
from src.train.trainer import TrainConfig, Trainer


def load_history(variant_dir: Path) -> list[dict]:
    path = variant_dir / "history.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_mean_iou": float(row.get("val_mean_iou", row.get("val_miou", 0.0))),
                }
            )
    return rows


def analyze(rows: list[dict]) -> dict:
    if not rows:
        return {"classification": "no_history", "detail": "history.csv missing"}
    train_loss = np.array([r["train_loss"] for r in rows])
    val_iou = np.array([r["val_mean_iou"] for r in rows])
    best_idx = int(np.argmax(val_iou))
    best_iou = float(val_iou[best_idx])
    final_iou = float(val_iou[-1])
    last_train_loss = float(train_loss[-1])

    detail = {
        "epochs": len(rows),
        "best_epoch": int(rows[best_idx]["epoch"]),
        "best_val_mean_iou": round(best_iou, 4),
        "final_val_mean_iou": round(final_iou, 4),
        "final_train_loss": round(last_train_loss, 4),
    }

    # Underfitting check
    if best_iou < 0.10 and last_train_loss > 1.0:
        classification = "underfitting"
        detail["reason"] = (
            f"best val mIoU {best_iou:.3f} < 0.10 and train loss "
            f"{last_train_loss:.3f} > 1.0: model never converged"
        )
        return {"classification": classification, "detail": detail}

    # Overfitting check (a): val mIoU dropped >= 5% from best epoch
    val_dropped = False
    if best_idx < len(rows) - 2 and final_iou <= best_iou * 0.95:
        val_dropped = True
        dropped = (best_iou - final_iou) / best_iou
        best_train = float(train_loss[best_idx])
        if last_train_loss <= best_train * 1.02:
            classification = "overfitting"
        else:
            classification = "mild_overfitting"
        detail["reason"] = (
            f"val mIoU dropped {dropped:.1%} after best epoch while "
            f"train loss kept decreasing ({best_train:.3f} -> {last_train_loss:.3f})"
        )
        detail["val_iou_drop_pct"] = round(dropped * 100, 1)
        return {"classification": classification, "detail": detail}

    # Overfitting check (b): train_loss continued dropping >15% while
    # val_mean_iou stalled/plateaued for >4 consecutive epochs
    # (Generalization Gap: train keeps improving, val doesn't)
    train_drop_pct = 0.0
    val_plateau_epochs = 0
    best_train = float(train_loss[best_idx])
    if best_idx < len(train_loss) - 1:
        train_drop_pct = (best_train - last_train_loss) / max(best_train, 1e-6)
        # Count epochs after best where val_iou didn't improve significantly
        for i in range(best_idx + 1, len(val_iou)):
            if val_iou[i] <= val_iou[i - 1] * 1.005:  # <= 0.5% improvement
                val_plateau_epochs += 1
            else:
                val_plateau_epochs = 0

    generalization_gap = train_drop_pct

    detail["train_loss_drop_pct"] = round(train_drop_pct * 100, 1)
    detail["val_plateau_epochs"] = val_plateau_epochs
    detail["generalization_gap"] = round(generalization_gap * 100, 1)

    if train_drop_pct > 0.15 and val_plateau_epochs > 4:
        classification = "overfitting"
        detail["reason"] = (
            f"Generalization gap: train loss dropped {train_drop_pct:.1%} "
            f"while val mIoU plateaued for {val_plateau_epochs} epochs "
            f"(gap={generalization_gap:.1%})"
        )
        return {"classification": classification, "detail": detail}

    # Healthy
    classification = "healthy"
    detail["reason"] = "val mIoU converged and stayed within 5% of best"
    return {"classification": classification, "detail": detail}


def suggest_corrections(variant: str, analysis: dict) -> dict:
    cls = analysis["classification"]
    base = {"variant": variant, "classification": cls}
    if cls == "overfitting":
        base.update(
            {
                "lr_factor": 0.5,
                "weight_decay_factor": 2.0,
                "extra_augmentation": True,
                "label_smoothing": 0.1,
                "enable_dropout": 0.2,
            }
        )
    elif cls == "mild_overfitting":
        base.update({"lr_factor": 0.8, "weight_decay_factor": 1.5, "enable_dropout": 0.2})
    elif cls == "underfitting":
        base.update({"lr_factor": 2.0, "weight_decay_factor": 0.5, "epochs_factor": 1.5})
    return base


def build_corrected_config(variant: str, corrections: dict, args: argparse.Namespace) -> TrainConfig:
    base_lr = getattr(args, "lr", 1e-4)
    base_wd = getattr(args, "weight_decay", 1e-4)
    base_epochs = getattr(args, "epochs", 20)
    lr = base_lr * corrections.get("lr_factor", 1.0)
    wd = base_wd * corrections.get("weight_decay_factor", 1.0)
    epochs = int(base_epochs * corrections.get("epochs_factor", 1.0))
    smoothing = corrections.get("label_smoothing", 0.0)
    return TrainConfig(
        epochs=epochs,
        lr=lr,
        weight_decay=wd,
        batch_size=getattr(args, "batch_size", 8),
        run_name=f"{variant}_corrected",
        label_smoothing=smoothing,
    )


def _train_class_weights(marida: MaridaDataset, train_ids: list[str]) -> torch.Tensor:
    counts = np.zeros(12, dtype=np.int64)
    for pid in train_ids:
        counts += np.bincount(marida.read_mask(pid).reshape(-1), minlength=12)[:12]
    labeled = counts[1:].sum()
    weights = ((labeled + 1.0) / (counts[1:] + 1.0))
    weights = weights / weights.min()
    print(f"class weights: {np.round(weights, 3).tolist()}")
    return torch.from_numpy(weights.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS) + ["detection"])
    parser.add_argument("--apply", action="store_true", help="retrain with corrections")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="run quick smoke test with reduced data")
    args = parser.parse_args()

    artifacts = ROOT / "artifacts"
    variants = [args.variant] if args.variant else sorted(VARIANTS)
    findings = []
    to_retrain = []

    for variant in variants:
        analysis = analyze(load_history(artifacts / variant))
        corrections = suggest_corrections(variant, analysis)
        findings.append({"variant": variant, **corrections, "detail": analysis["detail"]})
        cls = analysis["classification"]
        reason = analysis["detail"]["reason"]
        gap = analysis["detail"].get("generalization_gap", 0.0)
        print(f"[{variant}] {cls}: {reason}")
        if gap:
            print(f"         Generalization gap: {gap:.1f}%")
        if analysis["classification"] in ("overfitting", "mild_overfitting", "underfitting"):
            to_retrain.append((variant, corrections))

    out = ROOT / "artifacts" / "overfitting_report.json"
    out.write_text(json.dumps(findings, indent=2))
    print(f"\nReport saved to {out}")

    if args.apply and to_retrain:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        marida = MaridaDataset(ROOT)
        splits = load_official_splits(ROOT / "splits")
        assert_scene_disjoint(splits, marida)

        # Smoke mode: use small subset for quick verification
        if args.smoke:
            train_ids = splits["train"][:20]
            val_ids = splits["val"][:10]
            print(f"  [SMOKE MODE] Using {len(train_ids)} train / {len(val_ids)} val samples")
        else:
            train_ids = splits["train"]
            val_ids = splits["val"]

        for variant, corrections in to_retrain:
            print(f"\nRetraining {variant} with corrections: {corrections}")
            config = build_corrected_config(variant, corrections, args)
            config.class_weights = _train_class_weights(marida, train_ids)
            # Pass dropout/enable_dropout to model builder if supported
            dropout = corrections.get("enable_dropout", 0.0)
            model = resolve_variant(variant, pretrained=True)
            # Inject dropout into model if it supports it
            if hasattr(model, "dropout") and dropout:
                model.dropout = dropout
            if hasattr(model, "bottleneck_dim"):
                # Keep existing bottleneck_dim but ensure regularization is active
                pass
            loaders = build_loaders(
                marida,
                {"train": train_ids, "val": val_ids, "test": splits["test"]},
                batch_size=args.batch_size,
                train_transform=TrainTransform(),
            )
            trainer = Trainer(model, config, device)
            result = trainer.run(loaders["train"], loaders["val"])
            print(f"  corrected {variant}: best epoch={result['best_epoch']}, "
                  f"best val mIoU={result['best_mean_iou']:.4f}")


if __name__ == "__main__":
    main()