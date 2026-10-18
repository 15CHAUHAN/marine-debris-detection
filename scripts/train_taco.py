"""Fine-tune YOLOv8s on TACO for high-precision RGB debris detection.

Improvements over the original YOLOv8n run (mAP50=0.33):
- Uses YOLOv8s (small, not nano) — 3x more parameters for better accuracy
- Higher epochs (100) with early stopping (patience=15)
- cls=2.0 to penalize misclassification more (boosts precision)
- box=10.0 for tighter bounding boxes
- Mosaic and MixUp augmentations enabled
- copy_paste=0.2 for improved small-object detection
- warmup_epochs=5 for stable initial training
- Saves test_metrics.json for dashboard display

Run-dir fix (this script used to hardcode ``yolov8s_taco`` while Ultralytics
auto-incremented the real dir to ``yolov8s_taco-2``, so the final test
evaluation ran against an old ~1-epoch checkpoint from a crashed run and the
dashboard showed mAP50=0.118 instead of the real model). The trainer's ACTUAL
``save_dir`` and ``best`` weights are now resolved and copied to the canonical
paths before test evaluation, so test_metrics.json always reflects the model
just trained.

Usage:
    python scripts/train_taco.py --epochs 100 --batch 16 --imgsz 640
    python scripts/train_taco.py --smoke   # 2-epoch sanity run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Canonical location consumed by dashboard.pipeline.TACO_MODEL_PATH.
RUN_DIR = ROOT / "artifacts" / "taco_training" / "yolov8s_taco"
BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"
# Keep backward-compatible path for the dashboard (train_taco.py copies to it).
DASHBOARD_WEIGHTS = ROOT / "artifacts" / "taco_training" / "yolov8n_taco" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="Use 8 if you hit CUDA out-of-memory errors.")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--smoke", action="store_true", help="2-epoch sanity run")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Base model: yolov8n.pt (fast) or yolov8s.pt (accurate)")
    parser.add_argument("--name", default="yolov8s_taco",
                        help="Ultralytics run name (default: yolov8s_taco)")
    parser.add_argument("--exist-ok", action="store_true",
                        help="overwrite the run dir instead of letting Ultralytics "
                             "auto-increment it to '-2', '-3', ...")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_yaml = ROOT / "artifacts" / "taco_dataset" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            "Missing data.yaml! Run `python scripts/prepare_taco.py` first."
        )

    # Count training images to verify dataset is complete
    train_dir = ROOT / "artifacts" / "taco_dataset" / "images" / "train"
    n_train = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
    print(f"Training on {n_train} images in {train_dir}")
    if n_train < 500:
        print(
            f"WARNING: Only {n_train} training images found. "
            "Expected 1000+. Re-run prepare_taco.py with --download-images "
            "or use the TACO-master download.py to fetch all batches first."
        )

    epochs = 2 if args.smoke else args.epochs
    print(f"Model: {args.model} | Epochs: {epochs} | Batch: {args.batch} | imgsz: {args.imgsz}")

    model = YOLO(args.model)  # YOLOv8s pretrained on COCO as starting point
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "artifacts" / "taco_training"),
        name=args.name,
        exist_ok=args.exist_ok,
        workers=0,            # Safe on Windows (avoids DataLoader spawn issues)
        seed=42,
        deterministic=True,   # Reproducible across identical runs
        amp=True,             # Mixed precision — required for RTX 4050 6 GB
        # ---- Early stopping ----
        patience=15,
        # ---- Loss weights (boost precision) ----
        box=10.0,             # Tighter localization loss
        cls=2.0,              # Heavier classification penalty → higher precision
        # ---- Augmentation ----
        mosaic=1.0,           # Mosaic augmentation (4-image composite) — always on
        mixup=0.15,           # MixUp blending between training images
        copy_paste=0.2,       # Copy-paste augmentation for small litter objects
        fliplr=0.5,           # Horizontal flip
        flipud=0.1,           # Vertical flip
        degrees=10.0,         # Random rotation ±10°
        translate=0.1,        # Random translate ±10%
        scale=0.5,            # Random scale ±50%
        hsv_h=0.015,          # Hue shift
        hsv_s=0.7,            # Saturation shift
        hsv_v=0.4,            # Value/brightness shift
        # ---- Training stability ----
        warmup_epochs=5,      # Gradual LR warmup from 0
        close_mosaic=min(10, max(0, epochs // 2)),  # Disable mosaic in final epochs for fine-tuning
        # ---- Checkpointing ----
        save=True,
        save_period=10,       # Save checkpoint every 10 epochs
        verbose=True,
    )

    # Resolve the ACTUAL run directory and best weights produced by the
    # trainer. Ultralytics auto-increments the run name ('-2', '-3', ...) when
    # --name already exists and exist_ok=False, so never rely on the literal
    # RUN_DIR: it may still hold a stale checkpoint from an older attempt.
    trainer = getattr(model, "trainer", None)
    save_dir = Path(trainer.save_dir) if trainer is not None and getattr(trainer, "save_dir", None) else RUN_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    reported_best = Path(trainer.best) if trainer is not None and getattr(trainer, "best", None) else None
    run_best = (reported_best if reported_best is not None and reported_best.exists()
                else save_dir / "weights" / "best.pt")
    trainer_epoch = getattr(trainer, "epoch", None) if trainer is not None else None
    epochs_done = (int(trainer_epoch) + 1) if trainer_epoch is not None else epochs  # 0-indexed

    if not run_best.exists():
        raise SystemExit(
            f"Training finished but no best weights found at {run_best}\n"
            "Check the run directory manually."
        )

    # Copy the freshly trained best weights to the canonical path (always
    # refresh it so the constant below stays trustworthy).
    BEST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if run_best.resolve() != BEST_WEIGHTS.resolve():
        shutil.copy2(run_best, BEST_WEIGHTS)
    print(f"Best weights from actual run dir {save_dir.name} -> {BEST_WEIGHTS}")

    # Also copy to the dashboard's expected yolov8n_taco path for compatibility
    DASHBOARD_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BEST_WEIGHTS, DASHBOARD_WEIGHTS)
    print(f"Copied best weights -> {DASHBOARD_WEIGHTS} (dashboard-compatible path)")

    # Validate on held-out test split
    print("\nEvaluating on test split...")
    eval_model = YOLO(str(BEST_WEIGHTS))
    val_metrics = eval_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    summary = {
        "mAP50": float(val_metrics.box.map50),
        "mAP50_95": float(val_metrics.box.map),
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
        "weights_path": str(BEST_WEIGHTS),
        "run_dir": str(save_dir),
        "model": args.model,
        "epochs_trained": epochs_done,
        "n_train_images": n_train,
    }
    metrics_file = ROOT / "artifacts" / "taco_training" / "test_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"TACO Training Complete!")
    print(f"  Best model  : {BEST_WEIGHTS}")
    print(f"  Dashboard   : {DASHBOARD_WEIGHTS}")
    print(f"  mAP50       : {summary['mAP50']:.4f}")
    print(f"  mAP50-95    : {summary['mAP50_95']:.4f}")
    print(f"  Precision   : {summary['precision']:.4f}")
    print(f"  Recall      : {summary['recall']:.4f}")
    print(f"{'='*60}")
    if summary["mAP50"] < 0.40:
        print(
            "\nWARNING: mAP50 is still low. Possible causes:\n"
            "  1. Incomplete dataset — re-check that all TACO batches were downloaded\n"
            "  2. Too few epochs — try --epochs 150\n"
            "  3. Too small a model — try --model yolov8m.pt\n"
            "  4. Run `python scripts/prepare_taco.py --download-images` again\n"
        )


if __name__ == "__main__":
    main()