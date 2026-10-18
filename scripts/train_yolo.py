"""Fine-tune YOLOv8n for Marine Debris detection and evaluate on test.

Tiny debris blobs need (a) a class-balanced training set and (b) an upscaled
input so blobs cover enough anchors to be learned.

Usage:
  python scripts/train_yolo.py --epochs 100 --imgsz 512 --balanced
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--balanced", action="store_true", help="use the class-balanced train split")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    yaml_name = "data_balanced.yaml" if args.balanced else "data.yaml"
    data_yaml = ROOT / "artifacts" / "detection" / yaml_name
    if not data_yaml.exists():
        raise SystemExit("run scripts/prepare_yolo.py first")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=1 if args.smoke else args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "artifacts" / "detection" / "runs"),
        name="debris_balanced" if args.balanced else "debris",
        workers=0,
        seed=42,
        close_mosaic=5,
        verbose=True,
    )

    best = model.trainer.best if hasattr(model, "trainer") else None
    eval_model = YOLO(best) if best else model
    metrics = eval_model.val(
        data=str(data_yaml), split="test", imgsz=args.imgsz, workers=0, conf=0.001
    )
    print(f"test mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")
    print(f"test P={metrics.box.mp:.4f}, R={metrics.box.mr:.4f}")
    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "balanced": args.balanced,
        "imgsz": args.imgsz,
    }
    (ROOT / "artifacts" / "detection" / "test_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"saved {ROOT / 'artifacts' / 'detection' / 'test_metrics.json'}")


if __name__ == "__main__":
    main()