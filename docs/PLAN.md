# Implementation Plan

Project: AI-Based Marine Plastic Pollution Detection
Dataset: MARIDA (MARIne Debris Archive), Zenodo DOI 10.5281/zenodo.5151941
Hardware: RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.7 driver
Runtime: Python 3.12 venv, PyTorch cu126
Deadline: 10 hours from kick-off
A/B scope: plain U-Net (baseline) vs U-Net+Attention (candidate)

## 1. Dataset Summary (verified from extracted MARIDA)

| Metric | Value |
|---|---|
| Patches | 1381 (256x256, 11-band GeoTiff + _cl mask + _conf) |
| Scenes | 63 S2 acquisitions |
| Official split (scene-disjoint) | train 694 / val 328 / test 359 (raw patches) |
| Filtered training set | train 334 (694 raw, 360 dropped by `--filter-empty-dominance 0.90`) / val 328 / test 359 |
| Unique scenes per split | train 36 / val 12 / test 15 (zero overlap) |
| Working class set | 11 aggregated classes (see CONTEXT.md) |
| Target class | Marine Debris (class 1), 373 patches |

## 2. Execution Blocks

### Block 0: Setup (0.5 h)
Goal: reproducible environment, dataset validated
- Create Python 3.12 venv
- Install torch+torchvision from cu126 index
- Install runtime deps (see docs/SKILLS.md)
- Verify: rasterio reads a sample patch, mask values in {1..15}, 1381 .tif files loadable
- Write ADR-0001 (11-class aggregation), ADR-0002 (official split as primary), ADR-0003 (Python 3.12 + cu126)

### Block 1: Ingestion + Encoding (1 h)
Goal: Dataset class returns (image_tensor, mask_tensor) for any patch
- `src/data/ingestion.py`: read GeoTiff via rasterio, normalize bands to [0,1], return float32 tensor (C,H,W)
- `src/data/encoding.py`: apply 15-to-11 class lookup, return int64 mask
- Success: loads all 1381 patches without error, mask values verified as {1..11} (exhaustive check)
- Tests: unit test with 3 known patches, assertion on shape and value range

### Block 2: Splits (0.5 h)
Goal: scene-disjoint data loaders
- `src/data/splits.py`: parse train_X.txt/val_X.txt/test_X.txt, return lists of patch IDs per split
- Scene-grouped 10-fold generator using sklearn GroupKFold (group = scene)
- Official test split sealed: used only for final evaluation, never for model selection
- Success: `splits.py` returns 694 / 328 / 359 raw patch IDs; the train DataLoader built with the dominance filter (`--filter-empty-dominance 0.90`, which drops zero-debris patches with one class above 90% of labeled pixels) yields 334 patches, val 328, test 359 (val and test are never filtered)

### Block 3: Preprocess + Augment (0.5 h)
Goal: CLAHE on train, augmentation pipeline ready
- `src/data/augment.py`: CLAHE via OpenCV (clipLimit=2.0, tileGridSize=8x8), per-channel, train split only
- Albumentations pipeline: horizontal flip, vertical flip, random rotate 90, brightness/contrast shift
- Val/test: no augmentation, no CLAHE (per clean evaluation rule)
- Success: augmented batch renders visually correct (verified via matplotlib sample)

### Block 4: A/B Training (3 h)
Goal: two models trained on identical data and seeds
- `src/models/unet.py`: plain U-Net, encoder from segmentation-models-pytorch (e.g., resnet34 backbone), 11 output classes
- `src/models/unet_attention.py`: same architecture with scSE attention gates added
- `src/train/trainer.py`: training loop with cross-entropy loss + AdamW optimizer, LR = 1e-4
- Subagent A: plain U-Net training (save checkpoints, loss/epoch CSV)
- Subagent B: U-Net+Attention training (same config, same seed)
- Success: both models converge (loss decreasing over 15-20 epochs), checkpoints saved

### Block 5: Evaluation (1.5 h)
Goal: compare A vs B with real metrics on official test split
- `src/evaluate/metrics.py`: pixel-level IoU (per class + mean), mAP, F1, confusion matrix
- Run both models on test split (359 patches)
- Record all metrics in a results dict, log to dev log
- Success: no fabricated numbers, all results from actual model runs, loss curves plotted

### Block 6: Mapping Output (1.5 h)
Goal: pollution concentration heatmap + georeferenced GeoTiff
- `src/mapping/heatmap.py`: convolve predicted debris mask with Gaussian kernel, produce concentration map
- `src/mapping/georeference.py`: write output as GeoTiff with same CRS/transform as input patch (rasterio)
- Success: output GeoTiff renders correctly in QGIS or equivalent viewer, heatmap shows concentration gradient

### Block 7: Documentation + Review (1 h)
Goal: architecture review, final docs, demo
- Run improve-codebase-architecture skill against the codebase, produce HTML review
- Update PIPELINE.md with final data flow
- Record final results in DEVELOPMENT_LOG.md (real numbers only)
- Success: all docs current, demo script runs end-to-end on one test patch

## 2.5 Execution Status

| Block | Status |
|---|---|
| 0 Setup | Done 2026-09-11, env reproducible via `verify_env.py` |
| 1 Ingestion + Encoding | Done, plus Bug 1/Bug 2 label fixes on 2026-09-12 (weights + targets 0-based) |
| 2 Splits | Done, official split scene-disjoint |
| 3 Preprocess + Augment | Done, NaN sanitization added |
| 4 A/B Training | Done 2026-09-12: baseline best val mIoU 0.2431, attention 0.2182 |
| 5 Evaluation | Done 2026-09-12: test mIoU 0.2975 vs 0.2971, debris F1 0.153 vs 0.161 |
| 6 Mapping | Done: `src/mapping/`, `scripts/map_patch.py`, verified GeoTiff |
| 7 Documentation + Review | Done 2026-09-12: manual architecture review recorded (skill not loadable; review run in-session), 65 tests pass |
| 8 Model zoo (extended scope) | Done: cnn_unet (mIoU 0.1405), effnet_transformer (mIoU 0.2831), YOLOv8n detection (mAP50 ~0, sub-resolution blobs), CNN+BiLSTM classifier (collapsed to majority); static `dashboard.html` + Streamlit `dashboard/app.py` |
| 9 Robustness (Block 9) | Done 2026-09-14: leakage audit (30 cross-split near-dup pairs = adjacent-UTM-tile geography, 0 exact dups; test kept), full 10-fold scene-disjoint CV baseline mIoU 0.2754±0.0445, WGAN-GP 200-epoch train + GAN-augmented A/B (both variants; result: GAN augmentation does NOT help — test mIoU drops vs real-data-only) |

Full real numbers and per-block outcomes are in DEVELOPMENT_LOG.md.

## 3. A/B Testing Protocol

| Parameter | Baseline (U-Net) | Candidate (U-Net+Attention) |
|---|---|---|
| Encoder | ResNet-34 pretrained | ResNet-34 pretrained |
| Input channels | 11 | 11 |
| Output classes | 11 | 11 |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-4 | 1e-4 |
| Loss | Cross-entropy + Dice | Cross-entropy + Dice |
| Batch size | 8 | 8 |
| Epochs | 15-20 | 15-20 |
| Seed | 42 | 42 |
| CLAHE | train only | train only |
| Augmentation | albumentations (same) | albumentations (same) |

Model selection: both validated on official val split (328 patches). Final metrics on official test split (359 patches) only, reported once per model.

## 4. Subagent Contracts

### Agent: data-pipeline
- Goal: Dataset + DataLoader module that loads MARIDA patches and returns clean tensors
- Tasks: implement ingestion.py, encoding.py, splits.py, augment.py; write unit tests
- Success metrics: loads 1381 patches without error; masks have values in {1..11}; DataLoader yields batches of shape (8,11,256,256) and (8,256,256); all unit tests pass

### Agent: models-train
- Goal: both U-Net variants defined, trainable, producing checkpoints
- Tasks: implement unet.py, unet_attention.py, trainer.py; train on smoke-set (20 patches) to verify pipeline
- Success metrics: both models train without CUDA OOM; smoke-set loss decreases over 5 epochs; checkpoints saved

### Agent: eval-metrics
- Goal: metric computation and comparison framework
- Tasks: implement metrics.py; run on random predictions first (sanity check), then on real model outputs
- Success metrics: IoU/F1 values in [0,1]; confusion matrix sums to total pixels; no hardcoded values

### Agent: mapping-output
- Goal: heatmap and georeferenced GeoTiff output from predicted masks
- Tasks: implement heatmap.py, georeference.py
- Success metrics: output GeoTiff has same CRS and transform as input; heatmap shows concentration gradient (not uniform)

## 5. File Map

| File | Purpose | Depends on |
|---|---|---|
| `src/data/ingestion.py` | Load GeoTiff patches as tensors | rasterio, numpy |
| `src/data/encoding.py` | 15-to-11 class lookup | numpy |
| `src/data/splits.py` | Parse official split files | pathlib |
| `src/data/augment.py` | CLAHE + albumentations | cv2, albumentations |
| `src/models/unet.py` | Plain U-Net model | segmentation_models_pytorch |
| `src/models/unet_attention.py` | U-Net + scSE attention | segmentation_models_pytorch |
| `src/train/trainer.py` | Training loop, checkpointing | torch |
| `src/evaluate/metrics.py` | IoU, F1, mAP, confusion matrix | torch, sklearn |
| `src/mapping/heatmap.py` | Gaussian concentration map | scipy, numpy |
| `src/mapping/georeference.py` | Write georeferenced GeoTiff | rasterio |

## 6. Constraints and Risks

| Risk | Mitigation |
|---|---|
| 6 GB VRAM insufficient | batch size = 4, gradient accumulation, mixed precision (AMP) |
| Python 3.13 wheel failures | venv pinned to Python 3.12 (verified installed) |
| Class imbalance (Marine Debris 373 patches, Marine Water 870) | Dice + cross-entropy combo handles imbalance; no GAN in 10h scope |
| CUDA OOM on long training | checkpoint every epoch, early stopping on val loss |
| Rasterio CRS issues | verify CRS match before georeferencing; single patch test |
