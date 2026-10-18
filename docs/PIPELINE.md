# Pipeline Data Flow

This document explains how data moves through the system, file by file, and how modules connect. Read this before modifying any module.

## Data Flow Diagram (conceptual)

```
patches/
  S2_<scene>/
    *_0.tif      (11-band GeoTiff)     ----+                    _CONF.tif
    *_0_cl.tif   (class mask, 1-15)    ----+---> [ingestion] -> [encoding] -> tensors
    *_0_conf.tif (confidence, 1-3)     ----+                              |
                                                                [splits]
                                                                [augment]
                                                                   |
                                                          train/val/test sets
                                                                   |
                                                        [unet.py] / [unet_attention.py]
                                                                   |
                                                             [trainer]
                                                                   |
                                                       checkpoints + loss curves
                                                                   |
                                                       [metrics] -> IoU, F1, mAP
                                                                   |
                                                        [heatmap] -> [georeference]
                                                                   |
                                                        pollution GeoTiff
splits/train_X.txt, val_X.txt, test_X.txt -> [splits.py]
labels_mapping.txt -> multi-label checks in metrics
```

## Module Contracts

### 1. Ingestion (`src/data/ingestion.py`)

- Input: absolute path to a patch .tif (e.g., `patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif`)
- Output: `(image: np.ndarray[float32, (11, 256, 256)], meta: dict)`
- `meta` contains: CRS, transform, resolution, band count, original dtype
- Normalization: each band scaled to [0, 1] using band-wise min/max or percentile clip (configurable)
- Notes: returns 11 channels as stored in the GeoTiff; no interpolation here. The stored ordering is assumed B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (no descriptions embedded in the files)

### 2. Encoding (`src/data/encoding.py`)

- Input: class mask array `(256, 256)` with values in {0..15} (0 = unlabeled)
- Output: `(256, 256)` with values in {0..11} per the aggregation table in CONTEXT.md (0 = unlabeled)
- Implementation: vectorized array lookup (`CLASS15_TO_11[values]`) over the deterministic mapping list; zero values in the raw mask are kept as 0 to signal unlabeled pixels
- Contract: `encode_15to11(mask) -> encoded_mask` is a pure function, unit tested

### 3. Splits (`src/data/splits.py`)

- Input: `splits/train_X.txt`, `splits/val_X.txt`, `splits/test_X.txt`
- Output: three lists of patch IDs (e.g., `"1-12-19_48MYU_0"`)
- Derived: scene ID = patch ID minus trailing `_<index>`; enables scene grouping
- Contract: official test split is never used for model selection

### 4. Augment (`src/data/augment.py`)

- Input: image tensor `(11, 256, 256)`, mask `(256, 256)`, flag `is_train: bool`
- CLAHE: only when `is_train=True`; per-channel on all 11 bands; clipLimit=2.0, tileGridSize=(8,8)
- Albumentations spatial transforms applied only when `is_train=True` and provide `p=0.5` flips and rotations; masks transformed together with images (`same_transform` semantics)
- Output: augmented image+mask pair of unchanged shape

### 5. Models (`src/models/unet.py`, `src/models/unet_attention.py`)

- Input: batch tensor `(B, 11, 256, 256)` float32
- Output: per-class logits `(B, 11, 256, 256)` float32
- Build via segmentation-models-pytorch: `Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=11, activation=None)`
- Attention variant: adds scSE (concurrent spatial-channel squeeze-and-excitation) gates on decoder feature maps
- Contract: both models share identical encoder and config except the attention component (A/B control)

### 6. Trainer (`src/train/trainer.py`)

- Input: model, train DataLoader, val DataLoader, config dict (epochs, lr, seed)
- Outputs: checkpoints under `artifacts/<run_id>/`, per-epoch CSV of (epoch, train_loss, val_loss)
- Loss: CrossEntropy + Dice (weights from class-frequency inverse, computed on train split only)
- Logging: single log file per run; progress via tqdm

### 7. Metrics (`src/evaluate/metrics.py`)

- Input: predicted logits `(B, 11, H, W)`, ground truth `(B, H, W)` as 0-based train targets in {0..10} (`IGNORE_INDEX` = 11 marks unlabeled, skipped)
- Output: dict `{mean_iou, f1_macro, f1_class_1, accuracy, confusion_matrix}` where class 1 is Marine Debris
- Uses sklearn `confusion_matrix` and custom IoU; mAP relevant only if detection branch is later added (segmentation mIoU is primary)
- Contract: all values computed from tensors, never hardcoded

### 8. Mapping (`src/mapping/heatmap.py`, `src/mapping/georeference.py`)

- Input: predicted Marine Debris mask `(256, 256)` for a patch, original patch `meta` (CRS, transform) from `<...>.read_meta(patch_id)`
- heatmap.py: `debris_concentration()` Gaussian convolution (`scipy.ndimage.gaussian_filter`, sigma ~2 px) -> concentration float map [0, 1]
- georeference.py: `write_concentration_tiff()` writes the concentration map as float32 GeoTiff with the patch's own CRS and transform via rasterio
- Output: `artifacts/maps/<patch_id>_<variant>_debris_concentration.tif` + PNG preview, viewable in QGIS
- Wrapper: `scripts/map_patch.py` (load checkpoint, pick a test patch, write GeoTiff + preview)

## Traffic Scripts

| Script | Purpose |
|---|---|
| `scripts/verify_env.py` | Block 0 gate: environment + dataset sanity checks |
| `scripts/train_one.py` | Train one variant (baseline/attention) on the official split |
| `scripts/evaluate_test.py` | Evaluate one or both variants on the official test split |
| `scripts/loss_curves.py` | Plot train loss + val mIoU per epoch to `artifacts/figs/` |
| `scripts/map_patch.py` | Debris concentration GeoTiff + preview for one patch |
| `scripts/build_dashboard.py` | Regenerate the static `dashboard.html` (zoo summary, curves, confusion matrices, per-class and radar comparisons) from artifacts |
| `scripts/audit_data_leakage.py` | Duplicate / near-duplicate / suspicious-feature leakage audit -> `artifacts/leakage_audit/report.json` |
| `scripts/run_cross_validation.py` | Scene-disjoint 10-fold CV on train+val (test held out) -> `artifacts/cv/<variant>/cv_summary.json` |
| `scripts/train_gan.py` | Train WGAN-GP on the train split; generate synthetic multispectral patches -> `artifacts/gan/` |
| `scripts/detect_overfitting.py` | Classify variants as over/underfitting, write `artifacts/overfitting_report.json`, retrain with `--apply` |

## Class/Label Index Conventions (critical)

- Raw GeoTiff masks hold values 0..15. 0 = unlabeled.
- `encode_15to11` maps to the 11-class set {0..11} (value 0 = unlabeled).
- `to_train_targets` maps raw class c in 1..11 to train target c-1 in {0..10}; unlabeled (0) maps to IGNORE_INDEX (11). The model has 11 logits; channel c corresponds to class c+1 of the aggregated set (channel 0 = Marine Debris).
- `class_weights` returns weights aligned to these 0-based train targets (index 0 = Marine Debris).

## Data Splitting Rules

- Official MARIDA split is scene-disjoint (verified: zero scene overlap across train/val/test)
- 10-fold cross-validation option uses GroupKFold with scene as group; useful for robustness but the official split remains the reporting standard
- CLAHE and augmentation must never touch val/test

## Label Sources

| File | Contains | Used by |
|---|---|---|
| `patches/**/*_cl.tif` | Per-pixel class (1..15) | encoding -> training |
| `patches/**/*_conf.tif` | Per-pixel confidence (1..3) | optional filtering, not in the 10h baseline |
| `labels_mapping.txt` | Per-patch multi-label (15-dim) | leakage checks and class statistics |
| `shapefiles/*.shp` with `.dbf/.prj` | Vector annotations per scene | geospatial verification later |

## Environment and Version Lock

| Dependency | Version constraint | Reason |
|---|---|---|
| Python | 3.12.x | wheel availability for rasterio, segmentation-models-pytorch |
| torch, torchvision | cu126 build | driver caps at CUDA 12.7 |
| rasterio | >= 1.4 | cp312 wheels cache resolved |
| segmentation-models-pytorch | latest | U-Net + scSE prebuilt |
| opencv-python | latest | CLAHE |
| albumentations | latest | augmentation, no custom code |
| scikit-learn | latest | GroupKFold, metrics |