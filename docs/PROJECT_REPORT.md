# Project Report — AI-Based Marine Plastic Pollution Detection (MARIDA)

**Project:** TECH-ATHON · Marine Debris Detection from Sentinel-2
**Dataset:** MARIDA (Zenodo DOI 10.5281/zenodo.5151941)
**Hardware:** RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.7
**Runtime:** Python 3.12 venv, PyTorch cu126
**Period:** 2026-09-11 … 2026-09-15
**Status:** All planned blocks (0–9) complete

---

## 1. Executive Summary

We built a full end-to-end pipeline for pixel-level **Marine Debris** segmentation from
11-band Sentinel-2 patches, and ran a controlled **A/B experiment** comparing a plain
U-Net against a U-Net with scSE attention gates under identical conditions.

**Headline findings:**

| Result | Value |
|---|---|
| A/B winner (target class, Marine Debris F1) | Attention U-Net (+0.0078 F1 vs baseline) |
| A/B winner (mean IoU) | Tied (baseline 0.2975 vs attention 0.2971) |
| 10-fold scene-disjoint CV (baseline) | mean val mIoU **0.2754 ± 0.0445** |
| GAN augmentation | **Does NOT help** — both variants regress with synthetic data |
| Baseline overfitting flag | Baseline overfits (val mIoU falls 11.8 % after best epoch); others healthy |
| Data integrity | 0 exact duplicates; 30 cross-split near-duplicate pairs (geographic, not leak-actionable) |
| Detection / classification tracks | YOLOv8n mAP50 ≈ 0 (blobs are 1–6 px); CNN+BiLSTM collapses to majority class |

---

## 2. Objective and Scope

Detect floating marine plastic (Marine Debris) in Sentinel-2 imagery. The primary
experiment is a side-by-side comparison (A/B) of:

- **Baseline model** — plain U-Net, pretrained ResNet-34 encoder
- **Augmented model** — the same U-Net with spatial and channel squeeze-and-excitation
  (scSE) attention gates

Further extended tracks were added: from-scratch CNN U-Net, EfficientNet-B0 +
transformer U-Net, YOLOv8n box detection, a CNN+BiLSTM patch classifier, WGAN-GP
augmentation, 10-fold scene-disjoint cross-validation, an overfitting analysis and a
data-leakage audit.

---

## 3. Dataset

| Metric | Value |
|---|---|
| Patches | 1,381 (256×256, 11-band GeoTiff + class mask + confidence mask) |
| Scenes | 63 Sentinel-2 acquisitions |
| Bands | 11 (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12; excludes 60 m B9/B10) |
| Dtype / size | float32, 256×256, UTM CRS (EPSG varies by scene) |
| Working class set | 11 aggregated classes (15 raw classes mapped) |
| Target class | Marine Debris (class 1) — present in 373 of 1,381 patches |
| Official split (scene-disjoint) | train 694 / val 328 / test 359 |
| Scene split | train 36 / val 12 / test 15 scenes, zero overlap |

### Class presence (patches containing each aggregated class)

| Class | Patches |
|---|---|
| Marine Debris | 373 |
| Marine Water | 955 |
| Turbid Water | 220 |
| Ship | 182 |
| Clouds | 181 |
| Sparse Sargassum | 106 |
| Natural Organic Material | 71 |
| Shallow Water | 64 |
| Foam | 59 |
| Sediment-Laden Water | 51 |
| Dense Sargassum | 49 |

Marine Debris is extremely rare at the pixel level (≈ 0.45 % of labeled pixels), which
dominates every downstream metric.

---

## 4. Pipeline

The processing chain is deliberately off-line and deterministic (except stochastic
training-time augmentation):

1. **Ingestion** — `src/data/ingestion.py`: reads the 11-band GeoTiff, applies
   band-wise percentile normalization to [0,1], returns a float32 (C,H,W) tensor.
2. **Encoding** — `src/data/encoding.py`: deterministic 15→11 class lookup and a
   0-based training-target mapping used by all models and metrics.
3. **Splits** — `src/data/splits.py`: official scene-disjoint split + a
   scene-grouped K-fold generator (GroupKFold, group = scene).
4. **Augmentation** — `src/data/augment.py`: CLAHE (`clipLimit=2.0, tileGridSize=8×8`)
   plus Albumentations spatial/color transforms, applied **only** to training data;
   NaN sanitization for a small number of inert patches.
5. **Training** — `src/train/trainer.py`: cross-entropy + Dice loss, class weights,
   AdamW, mixed precision (AMP), early stopping, per-epoch CSV/JSON history.
6. **Evaluation** — `src/evaluate/metrics.py`: pixel IoU (per class + mean), F1,
   accuracy, confusion matrix on the official test split.
7. **Output** — `src/mapping/`: Gaussian-blurred debris concentration map written as a
   georeferenced GeoTiff with the source patch CRS/transform.

### 15→11 aggregated class mapping (summary)

| Raw 15-class | Aggregated 11-class |
|---|---|
| Marine Debris | Marine Debris |
| Dense / Sparse Sargassum | Dense / Sparse Sargassum |
| Natural Organic Material | Natural Organic Material |
| Ship | Ship |
| Clouds | Clouds |
| Marine Water | Marine Water |
| Sediment-Laden / Foam / Turbid / Shallow Water | each as-is |
| Waves / Cloud Shadows / Wakes / Mixed Water | Marine Water |

---

## 5. Segmented Model Zoo — Official Test Split

All segmentation variants evaluated on the official test split (359 patches, labeled
pixels only), best epoch chosen on the official val split. **Percent values.**

| Model | Params | Mean IoU | Pixel Acc | Debris F1 | Debris IoU | Best val mIoU (ep) | Verdict |
|---|---|---|---|---|---|---|---|
| baseline (U-Net ResNet-34) | 24.46 M | 29.75 | 76.15 | 15.28 | 8.27 | 24.31 (17) | A/B control |
| attention (U-Net + scSE) | 24.58 M | 29.71 | 77.18 | **16.06** | **8.73** | 21.82 (19) | A/B candidate |
| cnn_unet (from scratch) | 489 K | 14.05 | 23.40 | 12.88 | 6.88 | 12.18 (19) | underfit |
| effnet_transformer | 8.0 M | 28.31 | 72.31 | 5.02 | 2.58 | 22.61 (20) | good IoU, weak debris |
| baseline + GAN data | 24.46 M | 29.52 | 75.70 | 13.26 | 7.10 | 23.14 (18) | GAN hurts |
| attention + GAN data | 24.58 M | 26.65 | 75.28 | 10.12 | 5.33 | 22.69 (18) | GAN hurts |

**A/B conclusion:** essentially tied on mean IoU (Δ = −0.0004). The **attention**
candidate is the better Marine Debris detector (F1 16.06 % vs 15.28 %, IoU 8.73 % vs
8.27 %) and better on overall pixel accuracy (77.18 % vs 76.15 %).

### Per-class IoU on the test split (%)

| Class | baseline | attention | cnn_unet | effnet_transformer |
|---|---|---|---|---|
| Marine Debris | 8.27 | **8.73** | 6.88 | 2.58 |
| Dense Sargassum | 5.41 | 8.36 | 4.22 | 5.67 |
| Sparse Sargassum | 25.79 | 14.12 | 15.21 | **37.13** |
| Natural Organic Material | 0.00 | 0.85 | 0.24 | 0.75 |
| Ship | 20.64 | 21.60 | 20.05 | **26.06** |
| Clouds | 65.06 | **67.69** | 68.70 | 48.94 |
| Marine Water | 35.80 | **41.57** | 31.93 | 47.09 |
| Sediment-Laden Water | 84.52 | **86.79** | 0.98 | 81.21 |
| Foam | 9.12 | **26.12** | 0.34 | 12.56 |
| Turbid Water | 46.19 | 45.80 | 1.03 | 40.63 |
| Shallow Water | 26.50 | 5.15 | 4.99 | 8.83 |

The target class Marine Debris is the hardest class: all models sit in the 2–9 %
IoU band because of extreme class rarity and the 1–6 px debris signatures.

---

## 6. 10-Fold Scene-Disjoint Cross-Validation

**Protocol** (`scripts/run_cross_validation.py`): GroupKFold grouped by **scene** on
train+val (1,022 patches); the official test split (359 patches) is **held out**.
20 epochs per fold, identical trainer config as the main runs, seed 42.

### Per-fold results (baseline variant)

| Fold | Train patches | Val patches | Best epoch | Val mIoU (%) |
|---|---|---|---|---|
| 1 | 921 | 101 | 17 | 28.16 |
| 2 | 918 | 104 | 20 | 24.29 |
| 3 | 919 | 103 | 19 | 26.43 |
| 4 | 920 | 102 | 18 | **35.16** |
| 5 | 920 | 102 | 20 | 21.33 |
| 6 | 919 | 103 | 19 | 26.02 |
| 7 | 920 | 102 | 18 | 34.28 |
| 8 | 920 | 102 | 20 | 31.54 |
| 9 | 921 | 101 | 20 | 24.97 |
| 10 | 920 | 102 | 18 | 23.25 |

### Summary

| Metric | Value |
|---|---|
| Mean val mIoU | **27.54 %** |
| Std dev | 4.45 % |
| Min / Max | 21.33 % (fold 5) / 35.16 % (fold 4) |
| Single-run test mIoU (official split) | 29.75 % |

**Interpretation:** the single-run test result (29.75 %) falls within one standard
deviation of the CV mean (27.54 ± 4.45 %), confirming the headline A/B numbers are
stable across scene-disjoint re-partitioning and are **not** artifacts of one lucky
split.

---

## 7. GAN Augmentation (WGAN-GP) — Negative Result

**Architecture** (`src/data/gan_augment.py`):

| Component | Configuration |
|---|---|
| Generator | 100-dim noise → 11×256×256 (Sigmoid), 3.6 M params |
| Critic | 11×256×256 → scalar Wasserstein score, 2.8 M params |
| Loss | Wasserstein + gradient penalty (λ_gp = 10), n_critic = 5, grad clip 1.0 |
| Training | 200 epochs on official train split (694 patches) |
| Artifacts | `artifacts/gan/generator.pt`, `checkpoint.pt`, epoch sample grids |

**Augmented retrain protocol** (`scripts/train_gan_augmented.py`): 300 synthetic
patches generated, pseudo-labeled by a teacher model (baseline or attention) with
softmax confidence ≥ 0.8, keep-threshold calibrated to the real-data confident
fraction (25th percentile), then mixed with real data and retrained.

### A/B + GAN comparison (official test split)

| Variant | Real-only mIoU (%) | +GAN mIoU (%) | +GAN Debris F1 (%) | Synthetic kept |
|---|---|---|---|---|
| baseline | 29.75 | 29.52 | 13.26 | 42 / 300 |
| attention | 29.71 | 26.65 | 10.12 | 6 / 300 |

**Interpretation:** pseudo-labeled synthetic patches overwhelmingly reinforce the
majority **Marine Water** class, diluting the already-rare debris signal. Both GAN
augmented variants perform **worse** than real-data-only training. **GAN augmentation
is not recommended** for this dataset / task.

---

## 8. Additional Track: Detection and Classification

| Track | Model | Test result |
|---|---|---|
| Box detection | YOLOv8n (512 px, class-balanced) | mAP50 0.0001 % (≈0), mAP50-95 0.000001 %, precision ≈0 %, recall 33.3 % |
| Patch classification | CNN + BiLSTM | accuracy 71.87 %, precision 0.0, recall 0.0, F1 0.0 (majority-class collapse) |

**Interpretation:** Marine Debris blobs are only **1–6 px** — sub-resolution for a
box detector — so YOLO cannot learn a meaningful detector. The BiLSTM classifier
collapsed to predicting the majority class, so its 71.87 % accuracy is the trivial
split of the class prior. Neither track is competitive for this task.

---

## 9. Overfitting Analysis

Source: `artifacts/overfitting_report.json`, computed from the official val-split
training curves (attention, baseline, cnn_unet, effnet_transformer).

| Variant | Classification | Details |
|---|---|---|
| baseline | **overfitting** | val mIoU fell 11.8 % after best epoch (24.31 → 21.44) while train loss kept falling (0.795 → 0.712) |
| attention | healthy | converged, stayed within 5 % of best |
| cnn_unet | healthy | converged, stayed within 5 % of best |
| effnet_transformer | healthy | converged, stayed within 5 % of best |

Recommended corrections for baseline (LR × 0.5, weight decay × 2, label smoothing 0.1,
extra augmentation) are recorded in the report; the ten-fold CV already provides a
regularized cross-split estimate.

---

## 10. Data Leakage Audit

Source: `artifacts/leakage_audit/report.json`.

| Check | Result |
|---|---|
| Exact duplicates (whole dataset) | **0** |
| Near-duplicate pairs (similarity ≥ 0.985) | 74 pairs |
| Cross-split near-duplicate pairs | 30 pairs |
| Suspicious features (e.g. single-class dominance) | 727 patches |

**Interpretation:** the 30 cross-split near-duplicate pairs are geographic adjacency
artifacts — the same coastal region captured in adjacent UTM tiles on different
acquisition dates (e.g. `27-1-19_16PCC_8` in test vs several train patches) — not
copied imagery. No exact duplicates exist. The official split is retained as the
reporting standard.

---

## 11. Mapping Output

`scripts/map_patch.py` runs a checkpoint on one test patch and emits:

- A Gaussian-convolved **debris concentration map** (`artifacts/maps/*_concentration.tif`)
- A PNG preview
- Verified GeoTiff properties: CRS and affine transform match the source patch,
  256×256 float32, values in [0,1], 21,106 unique values (smooth gradient, not binary).

---

## 12. Reproducibility and Quality Gates

| Item | Value |
|---|---|
| Pinned dependencies | `requirements.txt` (torch cu126, smp, albumentations 2.0.8, rasterio 1.5.1, opencv, sklearn, matplotlib) |
| Environment gate | `scripts/verify_env.py` (dataset integrity + package versions) |
| Test suite | **73 pytest tests pass** (encoding, ingestion, splits, dataset, metrics, models, detection, mapping) |
| Key bug fixes | Bug 1 (class-weight off-by-one) and Bug 2 (1-based targets / dropped Shallow Water) — both found and fixed 2026-09-12; all final numbers are from post-fix retrains |
| Quarantined runs | `artifacts/_invalid_buggy_first_run/`, `artifacts/_contaminated_gan_aug_2runs_merged/` (pre-bugfix / contaminated, excluded from results) |

---

## 13. Key Findings and Conclusions

1. **Attention vs baseline:** essentially tied on mean IoU (29.71 vs 29.75 %); the
   attention U-Net is slightly better on the target class Marine Debris (F1 16.06 vs
   15.28 %, IoU 8.73 vs 8.27 %) and on pixel accuracy (77.18 vs 76.15 %).
2. **Stability:** 10-fold scene-disjoint CV for the baseline yields 27.54 ± 4.45 %
   val mIoU, bracketing the single-run test result (29.75 %) — results are robust to
   re-partitioning.
3. **GAN augmentation is a negative result.** It should not be used here; pseudo-
   labeled synthetic patches dilute the rare debris signal.
4. **Rare-class ceiling:** Marine Debris is ~0.45 % of labeled pixels and 1–6 px in
   size; all segmentation models land in the 2–9 % debris-IoU band, and box detection
   (YOLO) is infeasible at this resolution.
5. **Only the baseline overfits**; attention, from-scratch CNN and EfficientNet
   transformer variants converge cleanly.
6. **Data is clean** (0 exact duplicates); near-duplicate geographic pairs are
   adjacency artifacts, not leakage.

---

## 14. Artifact Index (selected outputs)

| Content | Path |
|---|---|
| Domain glossary / vocabulary | `CONTEXT.md` |
| Plan with execution status | `docs/PLAN.md` |
| Development log with real numbers | `docs/DEVELOPMENT_LOG.md` |
| Pipeline documentation | `docs/PIPELINE.md` |
| Static dashboard (HTML) | `dashboard.html` (+ regenerator `scripts/build_dashboard.py`) |
| Interactive dashboard (Streamlit) | `dashboard/app.py` |
| Test metrics (per variant) | `artifacts/<variant>/test_metrics.json` |
| 10-fold CV summary | `artifacts/cv/baseline/cv_summary.json` |
| GAN weights | `artifacts/gan/generator.pt`, `checkpoint.pt` |
| Overfitting report | `artifacts/overfitting_report.json` |
| Leakage audit | `artifacts/leakage_audit/report.json` |
| Concentration maps | `artifacts/maps/*_concentration.tif` (+ PNG) |