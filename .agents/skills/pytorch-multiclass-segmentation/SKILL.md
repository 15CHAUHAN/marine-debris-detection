---
name: pytorch-multiclass-segmentation
description: Architecture rules, loss function choices, and evaluation metrics for MARIDA 11-class marine debris segmentation.
---

# PyTorch Multiclass Segmentation Standards

1. **Target Dimensions & Shapes:**
   - Input Tensor: `(B, 11, 256, 256)` float32.
   - Ground Truth Mask: `(B, 256, 256)` int64 with values 0 to 11.
   - Ignore Index: Index 0 is unannotated/nodata. Always set `ignore_index=0` in loss functions and evaluation metrics.
   - Class 1 is **Marine Debris**.

2. **Severe Class Imbalance Strategy:**
   - Avoid plain Cross-Entropy without weights.
   - Use Compound Loss:
     `Total_Loss = Focal_Loss(gamma=2.0, alpha=weights) + 0.5 * Multiclass_Dice_Loss(smooth=1.0)`
   - Never let majority background (Class 6: Marine Water, Class 11: Clouds) dominate gradients.

3. **Metrics Evaluation:**
   - Evaluate model quality primarily on **Class 1 (Marine Debris) F1-Score and IoU**, not just overall mean IoU.
   - Calculate precision, recall, F1, and IoU per class using confusion matrix accumulation across test batches.
   - Checkpoint selection: Save `best.pth` based on highest validation Marine Debris F1 score.
