---
name: yolo-ultralytics-custom
description: Standards for training, fine-tuning, and evaluating YOLOv8 models on the TACO trash/litter dataset.
---

# YOLOv8 TACO Training & Inference Standards

1. **Dataset Format & data.yaml:**
   - Labels must be normalized bounding boxes in YOLO format: `<class_id> <x_center> <y_center> <width> <height>` (all values in [0.0, 1.0]).
   - `data.yaml` must point to absolute paths or paths relative to `D:/TECH-ATHON/artifacts/taco_dataset`.
   - The 7 macro classes:
     0: Bottle, 1: Can, 2: Plastic bag, 3: Cup, 4: Wrapper_Sachet, 5: Container, 6: Other_Trash.

2. **Training Configurations for High Precision:**
   - Model: Use `yolov8s.pt` or `yolov8m.pt` (do not downgrade to nano unless doing a smoke test).
   - Loss balance: Set `cls=2.0` and `box=10.0` to heavily penalize false positives and tighten localization.
   - Augmentations: Enable `mosaic=1.0`, `mixup=0.15`, and `copy_paste=0.2` to help with small debris items.
   - Disable mosaic in late epochs: `close_mosaic=10`.
   - Early stopping: `patience=15`.

3. **Weight Export & Dashboard Contract:**
   - The trained weights must be saved or copied to:
     `D:\TECH-ATHON\artifacts\taco_training\yolov8n_taco\weights\best.pt`
     (this ensures backward compatibility with `dashboard/pipeline.py`).
   - Write test metrics to:
     `D:\TECH-ATHON\artifacts\taco_training\test_metrics.json`
     containing `{"mAP50": ..., "precision": ..., "recall": ...}`.
