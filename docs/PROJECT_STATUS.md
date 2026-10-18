# Project Status - Marine Plastic Debris Detection

Status date: 15 Sep 2026. Pipeline: `Satellite/Drone Image/Video -> Plastic Detection -> Concentration -> Pollution Map`.

## 1. Dataset

- **MARIDA** (Sentinel-2 marine debris benchmark): 1381 patches, 63 scenes, 256x256, 11 spectral bands (B1-B8, B8A, B11, B12; 60 m bands B9/B10 excluded), float32, UTM CRS.
- 15-class annotations were aggregated to the **11-class working set** (Waves, Cloud Shadows, Wakes, Mixed Water merged into Marine Water). Target class of interest: **Marine Debris (class 1)**, present in 373/1381 patches (~27%).
- Per-pixel confidence masks (1/2/3) available for optional filtering; not used in the final runs.
- **Official MARIDA split** (scene-disjoint, verified): 694 train / 328 val / 359 test, zero scene or patch leakage.
- Data audit (see `artifacts/leakage_audit/report.json`): 0 exact duplicates, 74 near-duplicate pairs (>=0.985 similarity, mostly same-scene neighbors), 30 cross-split near-duplicate pairs flagged, 727 patches flagged for single-class dominance (mostly fully-marine-water or fully-cloud patches, i.e. easy negatives / label imbalance).

## 2. Implemented techniques

All segmentation variants registered in `src/models/registry.py` (`VARIANTS`). Every model consumes `(B, 11, H, W)` and emits `(B, 11, 256, 256)` logits.

| Variant key | Architecture | Status |
|---|---|---|
| `baseline` | U-Net, ResNet-34 encoder (first conv adapted to 11 channels), pretrained | Trained + evaluated |
| `attention` | U-Net + spatial/channel squeeze-and-excitation (scSE) attention gates | Trained + evaluated |
| `cnn_unet` | Compact from-scratch CNN + U-Net (<2 M params) | Trained + evaluated |
| `effnet_transformer` | EfficientNet-B0 encoder (11-ch stem) + Transformer stack bottleneck | Trained + evaluated |
| `yolo_vit_lstm` | YOLOv8n backbone + ViT spatial attention + uni-directional LSTM | Trained (5 epochs) + evaluated |
| `yolo_bilstm` | YOLOv8n backbone + bidirectional LSTM | Trained (5 epochs) + evaluated |

Supporting / side experiments:

- **WGAN-GP augmentation** (`src/data/gan_augment.py`, `scripts/train_gan.py`, `scripts/train_gan_augmented.py`): 11-channel WGAN generator producing synthetic marine patches. Augmented runs exist for baseline and attention.
- **YOLOv8n box detector** (`scripts/train_yolo.py`, `scripts/prepare_yolo.py`, labels under `artifacts/detection/labels/`): bounding-box debris detector, run name `debris_balanced` (balanced class sampling).
- **CNN+LSTM patch-level classifier** (`scripts/train_classifier.py`, `src/models/cnn_lstm.py`): per-patch "debris present / absent" classifier. Failed experiment.
- **Drone/video pipeline** (`src/data/drone_ingestion.py`, `src/detection/video_inference.py`): RGB->11-channel adapter (replicate/zero_pad/rgb_bands/rgb3), frame sampling, per-frame detection, boxes, Gaussian concentration heatmap, annotated output video + JSON report + aggregate PNG.
- **Georeferencing & maps** (`src/mapping/georeference.py`, `src/mapping/heatmap.py`, `dashboard/geo.py`): concentration GeoTIFF export, UTM->WGS84 coordinate extraction via rasterio.warp, and interactive Folium pollution maps.
- Training inputs: CLAHE (train split only), Albumentations spatial/color augmentation, class-weighted CE+Dice loss, 10-fold scene-disjoint splits (`src/data/splits.py`).

## 3. Dashboard (`dashboard/app.py`, Streamlit)

Five tabs mirroring the pipeline stages. Verified end-to-end headlessly (streamlit AppTest) on all four input kinds and the `attention` model.

1. **Source Input** - Sentinel-2 GeoTIFF upload, MARIDA test patch picker (debris filter), drone image (.jpg/.png), or drone/coastal video (.mp4/.avi). Drone upload auto-extracts EXIF GPS (fallback: realistic coastal demo coordinates), and the primary **"Analyze & Generate Pollution Map"** button runs the whole pipeline in one click: detects, computes concentration, writes the pollution heatmap GeoTIFF to `artifacts/maps/`, shows summary metric cards + a download button for the GeoTIFF, then jumps to tab 5.
2. **Model** - pick one of the 5 hybrid techniques; shows checkpoint availability and published test metrics (mIoU, accuracy, Marine Debris F1/IoU) loaded from `artifacts/<variant>/test_metrics.json`.
3. **Plastic Detection** - side-by-side original vs segmentation overlay + debris bounding boxes with confidence (per-pixel mean probability).
4. **Concentration** - density %, debris pixel + physical area (m^2), peak/mean concentration, severity pill (Low <1%, Moderate 1-5%, High 5-15%, Critical >=15%), hot heatmap. Optional user georef (center + GSD) for drone/video.
5. **Pollution Map** - interactive Folium/Leaflet map: patch extent polygon, green->yellow->red concentration cells, togglable heatmap overlay, OpenStreetMap or Esri satellite basemap, real WGS84 bounds extracted from the GeoTiff CRS.

Pollution heatmap output: the one-click analyze writes a single-band float32 GeoTIFF (source CRS + affine transform) named `{label}_{variant}_debris_concentration.tif` under `artifacts/maps/` (e.g. `12-12-20_16PCC_15_attention_debris_concentration.tif`), downloadable from the dashboard and registered 1:1 with the source in any GIS viewer.

Smoke results (attention model): MARIDA patch `12-12-20_16PCC_0` -> 6163 debris px, density 9.4%, High severity, map anchored at real Gulf of Mexico coords (WGS84 bounds -88.335..-88.311 E, 16.002..16.025 N). Uploaded 512x512 drone PNG -> density 6.2%; 12-frame mp4 -> 6 frames sampled, mean density 3.0%.

Deliverable note: all 5 techniques in the dashboard now have trained checkpoints and published metrics; the 2 YOLO hybrids were trained for 5 epochs (smoke-level, see Section 4) so their published numbers are not directly comparable to the 20-epoch runs of the other variants.

Tabs 4 and 5 also expose a one-click **"Download Official Pollution Incident Report (PDF)"** (`dashboard/report_generator.py`, rendered fully in memory). The A4 report contains: navy header banner (incident ref `MPD-YYYY-XXXX`, UTC/local timestamps, model); geospatial & acquisition telemetry; scene content & data provenance (model class coverage across the 11 classes, SHA-256 surface fingerprint, MARIDA split / acquisition date / per-pixel annotator agreement); a four-panel visual evidence strip (original / red overlay + yellow boxes / concentration heatmap / georeferenced static map snapshot); a quantitative analytics matrix (density %, debris m^2, km^2 analyzed, debris m^2 per km^2, largest cluster, plus video sampling stats and the color-coded severity badge); a concentration distribution histogram; model & quality assurance (checkpoint id + test mIoU / accuracy / Marine Debris IoU-F1 from `artifacts/<variant>/test_metrics.json`); the severity-gated operational protocol; repeat-pass surveillance (delta vs the first registered baseline scan, persisted to `artifacts/reports/history.json`); a per-cluster detection inventory; and a machine-readable QR record. Footer carries the legal disclaimer and "Page X of Y".

## 4. Segmentation accuracy - official 359-patch test split

Metrics computed on the untrained test split (11 classes). `mean IoU` = mean of per-class IoU. Best checkpoints from training on the 694-patch train split (val 328 held out during training).

| Model | mean IoU | Pixel accuracy | Debris F1 | Debris IoU | best epoch |
|---|---|---|---|---|---|
| baseline (U-Net ResNet-34) | 0.3681 | 0.7611 | 0.1152 | 0.0611 | 26 |
| attention (U-Net + scSE) | **0.3745** | **0.7788** | **0.1626** | **0.0885** | 23 |
| cnn_unet (from-scratch) | 0.1405 | 0.2340 | 0.1288 | 0.0688 | 19 |
| effnet_transformer | 0.2831 | 0.7231 | 0.0502 | 0.0258 | 20 |
| yolo_vit_lstm | 0.2602 | 0.7646 | 0.1193 | 0.0635 | 5 |
| yolo_bilstm | 0.2523 | 0.7596 | 0.0389 | 0.0198 | 5 |
| baseline + GAN aug | 0.2952 | 0.7570 | 0.1326 | 0.0710 | - |
| attention + GAN aug | 0.2665 | 0.7528 | 0.1012 | 0.0533 | - |
| contaminated run (quarantined) | 0.2225 | 0.6647 | 0.1071 | 0.0566 | - |

Takeaways:
- `baseline` and `attention` are statistically tied on mean IoU; `attention` slightly better on pixel accuracy and Marine Debris F1/IoU.
- Marine Debris segmentation is hard: best debris IoU ~0.087, debris F1 ~0.16. Dominant classes (Marine Water, Sediment-Laden Water, Clouds) score 0.36-0.87 IoU; rare/small classes (Natural Organic Material, debris, shallow water) score near 0.
- GAN augmentation did not improve either variant (reduced mean IoU), so trained checkpoints without GAN data are recommended.

## 5. Object detection + classifier results

- **YOLOv8n box detector** (`debris_balanced`, 100 epochs, imgsz 512): best mAP50 ~0.050 (epoch 92), best recall ~0.304 (epoch 53), best precision ~0.778 (epoch 100). Historically weak - MARIDA debris objects are very small; the segmentation route is the usable one.
- **CNN+LSTM patch classifier**: accuracy 0.719, but debris precision/recall/F1 = 0.0 (collapsed to always-predict-no-debris). Discarded.

## 6. Cross-validation

Run through `scripts/run_cross_validation.py` (scene-grouped, scene-disjoint folds over all 63 scenes) and `scripts/run_all_cv.py` for multi-variant comparisons. Test split is held out entirely.

**Baseline U-Net, 10-fold / 20 epochs per fold** (official protocol):
- Mean val mIoU: **0.27543** (+/- 0.04445 std)
- Min fold: 0.2133, max fold: 0.3516
- Fold sizes: ~918-921 train / ~101-104 val patches
- Fold-level best-epoch mIoU: 0.2816, 0.2429, 0.2643, 0.3516, 0.2133, 0.2602, 0.3428, 0.3154, 0.2497, 0.2325

**attention / cnn_unet / effnet_transformer, 5-fold / 5 epochs per fold** (short protocol, `run_all_cv.py --epochs 5 --folds 5`). Caution: fold metric means are not directly comparable to the baseline's 10-fold/20-epoch numbers.

| Variant | Mean val mIoU | +/- std | val debris F1 | Fold mIoUs |
|---|---|---|---|---|
| attention | 0.1027 | 0.0514 | 0.0869 +/- 0.0366 | 0.0759, 0.0643, 0.0453, 0.1540, 0.1740 |
| cnn_unet | 0.1535 | 0.0275 | 0.1701 +/- 0.0867 | 0.1818, 0.1890, 0.1173, 0.1367, 0.1425 |
| effnet_transformer | 0.1355 | 0.0291 | 0.1108 +/- 0.0338 | 0.1239, 0.1063, 0.1774, 0.1076, 0.1623 |

Comparison table persisted at `artifacts/cv/_all/cv_compare.json`. `run_all_cv.py --smoke` (1 epoch / 2 folds / batch 4) is the regression sanity check for all 6 variants.

## 7. Where everything is stored (artifact map)

Architecture code:
- `src/models/` - `unet.py`, `unet_attention.py`, `unet_cnn.py`, `unet_effnet_transformer.py`, `yolo_vit_lstm.py`, `yolo_bilstm.py`, `registry.py`
- `src/data/` - ingestion, encoding (15->11), dataset, factory, augment (CLAHE/albumentations), splits, drone_ingestion, gan_augment
- `src/evaluate/` - metrics (mIoU, debris F1/precision/recall), evaluate.py
- `src/detection/` - labels, video_inference
- `src/mapping/` - heatmap (Gaussian concentration), georeference (concentration GeoTIFF)

Training / evaluation scripts:
- `scripts/train_one.py` (single variant), `scripts/run_cross_validation.py` (one variant), `scripts/run_all_cv.py` (all variants), `scripts/evaluate_test.py`, `scripts/train_yolo.py`, `scripts/train_gan.py`, `scripts/train_gan_augmented.py`, `scripts/train_classifier.py`, `scripts/audit_data_leakage.py`, `scripts/map_patch.py`, `scripts/build_dashboard.py`

Model checkpoints (PyTorch state dicts, `artifacts/<variant>/best.pth`):
- `artifacts/baseline/best.pth`, `artifacts/attention/best.pth`, `artifacts/cnn_unet/best.pth`, `artifacts/effnet_transformer/best.pth`
- `artifacts/yolo_vit_lstm/best.pth`, `artifacts/yolo_bilstm/best.pth` (5-epoch smoke training, see Sections 3-4)
- `artifacts/baseline_gan_aug_run_final/best.pth`, `artifacts/attention_gan_augmented/best.pth` (GAN-augmented runs)
- `artifacts/gan/generator.pt`, `artifacts/gan/checkpoint.pt` (WGAN), `artifacts/detection/runs/debris_balanced/weights/best.pt` (YOLO box detector), `artifacts/cnn_lstm_classifier/best.pth`
- CV fold checkpoints: `artifacts/cv/baseline/fold_1..10/best.pth` (note: fold_1 and fold_2 were overwritten by the smoke validation run; fold_3..10 intact)

Results / metrics (JSON):
- Test split: `artifacts/<variant>/test_metrics.json` + `confusion_matrix.json` + `history.json/csv`
- CV: `artifacts/cv/baseline/cv_summary.json` (10-fold/20-epoch), `artifacts/cv/{attention,cnn_unet,effnet_transformer}/cv_summary.json` (5-fold/5-epoch), per-fold `best.pth` under `artifacts/cv/<variant>/fold_<n>/`; `artifacts/cv/_all/cv_compare.json` comparison table
- YOLO: `artifacts/detection/runs/debris_balanced/results.csv`, `args.yaml`, curves (PR/F1/confusion matrix PNGs)
- Quarantined (invalid, do not use): `artifacts/_contaminated_gan_aug_2runs_merged/test_metrics_CONTAMINATED_quarantined.json`, `artifacts/_invalid_buggy_first_run/`
- Leakage audit: `artifacts/leakage_audit/report.json`

Figures:
- `artifacts/figs/loss_curves.png`; confusion matrices + sample grid in `artifacts/figs/dashboard/`

Maps / output products:
- Concentration GeoTIFFs: `artifacts/maps/*_debris_concentration.tif`
- Concentration PNGs: `artifacts/maps/*_concentration.png`

Dashboard:
- `dashboard/app.py` (Streamlit UI), `dashboard/pipeline.py` (detection/concentration core), `dashboard/geo.py` (georeferencing + folium). Run: `.\.venv\Scripts\streamlit run dashboard\app.py`
- Static build: `dashboard.html` via `scripts/build_dashboard.py`

Tests: `tests/` (133 tests, all passing). Dataset root is the repo root (patches + `splits/` at project root, loaded via `MaridaDataset(ROOT)`). Docs: `CONTEXT.md` (domain glossary), `DEVELOPMENT_LOG.md` (run history), this file.

## 8. Verified status

- `pytest tests/` -> 133 passed.
- Streamlit app boots with all 5 tabs, 0 exceptions, across GeoTIFF / MARIDA / drone-image / video inputs.
- Environment: Windows, Python 3.12 venv (`.venv`), CUDA-enabled (NVIDIA GeForce RTX 4050, 6 GB). Dependencies pinned in `requirements.txt`.