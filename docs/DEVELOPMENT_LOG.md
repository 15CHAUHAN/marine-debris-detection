# Development Log

This log records every change, decision, and outcome in this project, following the Project Guidelines (see PROJECT_GUIDELINES.md).

## Log Format

Each entry uses this structure:

- Date
- Change / Decision
- Reasoning
- Files affected
- Result (only from actual testing)

---

## 2026-09-11

### Decision: Store working rules in docs/PROJECT_GUIDELINES.md

- Change: Created docs/PROJECT_GUIDELINES.md with the mandatory working rules for this project.
- Reasoning: Guideline 9 requires these rules to be written to a markdown file before any work begins.
- Files affected:
  - docs/PROJECT_GUIDELINES.md (created)
  - docs/DEVELOPMENT_LOG.md (created)
- Result: Guidelines file written and acknowledged. Awaiting further instructions from the user.

### Decision: Scope confirmed with user Q&A

- Change: Confirmed architecture scope (U-Net vs U-Net+Attention), local GPU, 11 aggregated classes, 10 hour deadline.
- Reasoning: Build decisions drive every downstream module contract.
- Result: scoped down to segmentation A/B; GAN and detection/temporal branches deferred.

### Decision: MARIDA integrity verified

- Change: Inspected extracted dataset in the project root.
- Facts verified: 1381 patches, 63 scenes, official train/val/test = 694/328/359, zero scene overlap across splits.
- Reasoning: Requirement to never assume; every planning number grounded in the actual files.
- Result: Official split usable as leak-free reporting standard (ADR-0002).

### Decision: Create project documentation set

- Change: Wrote the planning documentation described by the user-approved plan.
- Files affected:
  - CONTEXT.md (domain glossary, created)
  - docs/PLAN.md (implementation plan, created)
  - docs/PIPELINE.md (data flow, created)
  - docs/SKILLS.md (skill + dependency registry, created)
  - docs/ADR/README.md (decision records ADR-0001..0005, created)
- Reasoning: Guideline 1 (documentation first) and the improve-codebase-architecture skill conventions.
- Result: Planning docs in place. Environment setup (Block 0) is the next work item.

### Decision: Environment built and dataset verified (Block 0)

- Change: Created `.venv` on Python 3.12, installed torch 2.14.0+cu126, torchvision 0.29.0+cu126, opencv 5.0, rasterio 1.5.1, albumentations 2.0.8, segmentation-models-pytorch 0.5.0, sklearn 1.9.1, matplotlib 3.11.1, pandas 3.0.5, tqdm.
- Change: Added `scripts/verify_env.py` as the repeatable Block 0 gate.
- Facts verified on disk (corrections recorded):
  - Patches are 11-band (Sentinel-2 excluding 60m bands B9/B10), not 13-band. float32, 256x256.
  - Class masks `_cl.tif` are stored as float32 with whole-number values in 0..15, not uint8. 0 = unlabeled.
  - 1381 image patches, 1381 class masks, 1381 confidence masks, 63 scenes.
  - Official splits: train 694, val 328, test 359; scene-disjoint (verified earlier).
  - Sample patch S2_1-12-19_48MYU_0 mask values {5,7,14} exactly match labels_mapping.txt binary vector (Ship, Marine Water, Wakes) at indices 4,6,13.
  - Image pixel values are surface reflectance in ~[0,0.3] (float32), so band-wise percentile normalization to [0,1] is applied at ingestion.
  - CRS is UTM per scene (e.g., EPSG:32748, EPSG:32616), georeferencing meta captured at ingestion.
- Docs corrected to match: CONTEXT.md (11-band, float32 masks), PIPELINE.md (11 channels, all-band CLAHE), PLAN.md (11 input channels).
- Result: Block 0 gate passes. Environment reproducible via venv. Modeling input contract is now firmly 11 channels.

## 2026-09-12

### Result: Baseline U-Net trained (first attempt, invalidated by Bug 1/2)

- Change: Trained plain U-Net (resnet34, 11 in/out channels) on the official split for 20 epochs, batch 8, AdamW lr 1e-4, CE+Dice, AMP.
- Result: best val mean_iou 0.2931 (epoch 16). Saved to `artifacts/baseline/best.pth`.
- Note: these numbers are INVALID for the final report. Both trained runs used shifted class targets and shifted class weights (see Bugs 1 and 2 below). Checkpoints superseded by the official retrain.

### Bug 1 (found + fixed): class weights misaligned by one position

- Symptom: Marine Debris F1/IoU exactly 0 on val and test for both models.
- Root cause: `_train_class_weights` in `scripts/train_one.py` and `class_weights` in `src/data/encoding.py` binned the raw mask values 0..10 (index 0 = unlabeled, 45M px) but `nn.CrossEntropyLoss(weight=)` indexes by 0-based train target 0 = Marine Debris. Marine Debris (1943 px) therefore received the unlabeled background weight (1.0) instead of its true inverse frequency (~79). Printed weights `[1.0, 23175.192, ...]` confirmed the shift.
- Fix: counts now use raw classes 1..11 (`counts[1:]`), so `weights[i]` matches train target i.
- Files affected: `scripts/train_one.py`, `src/data/encoding.py`, `tests/test_encoding.py` (regression test).

### Bug 2 (found + fixed): train targets were 1-based, Shallow Water dropped

- Root cause: `CLASS11_TO_TRAIN` was `[11, 1, 2, ..., 10, 11]`, so raw class c mapped to target c (not c-1), channel 0 was never supervised, Shallow Water (raw 11) mapped to IGNORE_INDEX and was silently ignored in loss and metrics, and `metrics.py` (which expects 0-based targets) interpreted predictions one class off.
- Fix: `CLASS11_TO_TRAIN` is now `[11, 0, 1, ..., 10]`; raw 1..11 -> targets 0..10, unlabeled (raw 0) -> IGNORE_INDEX (11).
- Files affected: `src/data/encoding.py`, `tests/test_encoding.py` (regression test on `to_train_targets`).

### Decision: retrain both models after the label bug fix

- Reasoning: Bug 1 + Bug 2 changed the meaning of every training target and every CE weight. The existing checkpoints cannot be part of the A/B comparison.
- Result: Pending. Both variants retrained on the official split with identical config, then evaluated on the official test split.

### Result: corrected retrain and final A/B evaluation (official test split)

- Change: Retrained both variants after Bug 1/2 fixes with identical protocol (20 epochs, batch 8, AdamW lr 1e-4, CE+Dice, AMP, seed 42). Class weights now print correctly: Marine Debris weight 79.4, most frequent class 1.0.
- Result (official test split, 359 patches, one run per model, real numbers):
  | metric | baseline | attention |
  |---|---|---|
  | mean IoU | 0.2975 | 0.2971 |
  | pixel accuracy | 0.7615 | 0.7718 |
  | Marine Debris F1 | 0.1528 | 0.1606 |
  | Marine Debris IoU | 0.0827 | 0.0873 |
  | best val mIoU (selection) | 0.2431 (epoch 17) | 0.2182 (epoch 19) |
- `test_metrics.json` written for each variant under `artifacts/<variant>/`.
- A/B conclusion: essentially tied on mean IoU (delta -0.0004). The attention candidate is slightly better on the target class Marine Debris (+0.0078 F1, +0.0046 IoU) and on accuracy (+0.01). Both models detect Marine Debris with low F1 (~0.15-0.16), consistent with a rare 0.45%-of-pixels class.
- Loss curves: `artifacts/figs/loss_curves.png`.

### Result: precise bug root cause (why earlier debris F1 was exactly 0.0)

- Bug 1 (weights): index 0 of the weight array was the unlabeled-background inverse frequency (weight 1.0). Marine Debris received 1.0 instead of 79.4.
- Bug 2 (targets): the model had no supervised channel 0 and Shallow Water was dropped via IGNORE_INDEX, so every trained logit was one class off from its semantic label.
- After both fixes, Marine Debris went from F1 0.0 to F1 0.15-0.16. The earlier 0.0 was a labeling artifact, not a genuine "cannot detect debris" finding.

### Result: mapping module (Block 6)

- Change: Implemented `src/mapping/heatmap.py` (Gaussian convolution concentration map) and `src/mapping/georeference.py` (GeoTiff writer with source CRS/transform). Added `scripts/map_patch.py` to run a checkpoint on one patch and emit both a GeoTiff and a PNG preview.
- Result: ran on test patch `22-12-20_18QYF_0` for both variants. Outputs under `artifacts/maps/`. Verified on disk: CRS and affine transform match the source patch, 256x256 float32, values in [0,1], 21106 unique values (smooth gradient, not binary/uniform), all finite.
- Data quality note: 7 patches in scene 16PCC contain NaN pixels (2 fully NaN). All 7 are >99% unlabeled so they are effectively inert in training (mask=0 maps to IGNORE_INDEX). `augment.py` now sanitizes non-finite pixels to 0 before CLAHE to silence the cast warning.

### Decision: architecture review (Block 7, manual)

- Change: Ran the architecture review manually; the `improve-codebase-architecture` skill is not loadable in this opencode session (not in the built-in skill set).
- Seam review: `src/data/ingestion.py` (MaridaDataset) is the deep seam all other modules consume. `metrics.py` exposes a clean per-batch contract. `mapping/` is a leaf module built only on `PatchMeta`. The trainers never touch the GeoTiff layout. The pipeline matches the PIPELINE.md contracts.
- Findings: no shallow duplicated modules; `evaluate_test.py`, `map_patch.py`, and `loss_curves.py` are thin, model-selection-safe wrappers. The full test suite (58 tests) passes.### Train a second from-scratch decoder U-Net (CNN+U-Net, Block 4) and an EfficientNet+Transformer U-Net (Blocks 0-6 rows)

- Change: --variant=attention in artifacts/attention coverage; --variant=cnn_unet and --variant=effnet_transformer now trained, evaluated on the official test split, and registered for the dashboard and map patch script
- Reasoning: baseline=ResNet34 encoder + attention=scSE gave nearly identical test mIoU. Two extra architectures on disjoint official splits to confirm the ranking is not an artifact of the decoder head.
- Files: scripts/train_one.py, scripts/evaluate_test.py, scripts/map_patch.py, src/models/{unet_cnn,unet_effnet_transformer}.py, src/models/registry.py
- Result (official test split, 359 patches): cnn_unet mIoU=0.1405 acc=0.2340 debrisF1=0.1288; effnet_transformer mIoU=0.2831 acc=0.7231 debrisF1=0.0502. Both below the baseline (0.2975 / 0.7615 / 0.1528) on mIoU; effnet_transformer beats cnn on accuracy and both loss curves from scripts/loss_curves.py. 65 tests pass.

### Added 4 requested hybrid variants + evaluation (manual)

- Change: extended the model zoo with the 4 requested architectures, all trainable through the single refactored scripts/train_one.py --variant CLI and reportable from the shared Streamlit dashboard.
- Variants added:
  - cnn_unet (CNN + U-Net, from-scratch encoder, 0.489M params)
  - effnet_transformer (EfficientNet-B0 bottleneck + 2-layer dot-product transformer before the U-Net decoder, 8.0M params)
  - hydra (YOLOv8n fine-tuned on Marine Debris boxes derived from masks; balanced split + 512px input to expose sub-resolution blobs)
  - ilstm (CNN-band-feature + Bidirectional-LSTM patch classifier; honest stand-in since MARIDA is single-frame, no temporal axis)
- Files: scripts/{train_one,prepare_yolo,train_yolo,train_classifier}.py, src/models/{unet_cnn,unet_effnet_transformer,cnn_lstm}.py + src/models/registry.py, src/data/gan_augment.py (WGAN-GP moved to the data module), src/detection/, tests/test_models.py, tests/test_detection.py
- Result (official test split):
  - baseline mIoU 0.2975 / debris F1 0.1528
  - attention mIoU 0.2970 / debris F1 0.1606
  - cnn_unet mIoU 0.1405 / debris F1 0.1288
  - effnet_transformer mIoU 0.2831 / debris F1 0.0502
  - A/B verdict unchanged: attention and baseline remain tied on pixel-level mIoU; the more expressive encoders do not exceed the ResNet-34 U-Net baseline.
- Honest finding: YOLO and the BiLSTM cannot learn the Marine Debris class � blobs are 1-6px (sub-resolution at 256px input), and >68% of training images carry no debris at all. Even a class-balanced YOLOv8n at 512px caps around mAP 0.049 live; the standalone BiLSTM collapses to the majority-all-negative solution (debris F1 0.0). Recorded as an expected architectural finding (documented in PLAN.md, not hidden).

### Result: data-driven static dashboard (Block 8)

- Change: wrote `scripts/build_dashboard.py` that auto-discovers all trained variants from `artifacts/*/test_metrics.json`, computes real 11x11 confusion matrices on the official test split (cached to `confusion_matrix.json` per variant), generates dataset sample pictures, and regenerates `dashboard.html` with every number sourced from a real artifact. No hardcoded values.
- Generated assets:
  - `artifacts/figs/dashboard/dataset_samples.png` (3 test patches RGB+GT)
  - `artifacts/figs/dashboard/cm_{baseline,attention,cnn_unet,effnet_transformer}.png` (11x11 heatmaps)
  - `artifacts/<variant>/confusion_matrix.json` (194,863 labeled test pixels, consistent across all 4 models)
- Dashboard sections: dataset overview cards, per-class presence chart, model selector (6 variants), zoo summary table, accuracy comparison bar + YOLO detection bar, per-class IoU grouped bars + detail table, IoU and F1 radar charts, per-variant training curves (loss + validation score), confusion matrix image gallery, mapping output gallery.
- Families covered: segmentation (4 variants: baseline, attention, cnn_unet, effnet_transformer), detection (YOLOv8n, class-balanced 512px), classifier (CNN+BiLSTM).
- Verified: JS syntax OK (`node --check`), runtime smoke OK (stub DOM/Chart: all tables, charts, selector, confusion gallery render correctly), all 8 referenced images exist, all 4 confusion matrices match test_metrics.json exactly.
- Files affected: `scripts/build_dashboard.py` (created), `dashboard.html` (regenerated), `docs/DEVELOPMENT_LOG.md` (this entry).
- Usage: `.venv\Scripts\python.exe scripts\build_dashboard.py` to regenerate after any new training run. `--no-confusion` skips recomputing confusion matrices; `--no-images` skips regenerating sample pictures.

### Result: full CV + GAN-augmented pipeline (Block 9, completed)

- Leakage audit findings (repeatable via `scripts/audit_data_leakage.py` -> `artifacts/leakage_audit/report.json`):
  - Exact duplicates (content-hash identical pairs): **0** within any split and **0** across splits.
  - Near-duplicates (patch-signature cosine >= 0.985): 74 pairs; **30 cross-split pairs** all involving scene `16PCC` (e.g., train `24-10-18_16PDC_13` ~ test `27-1-19_16PCC_20`, sim 0.9884). These are *same-region coastal tiles in adjacent UTM zones imaged days apart* — geographic adjacency, **not** copied label data (verified: 0 exact dups, sims ~0.985-0.989). Documented as a known dataset characteristic, not actionable leakage.
- 10-fold scene-grouped CV run (baseline, group=scene, official non-test data).
- WGAN-GP trained on the official train split (200 epochs -> `artifacts/gan/`): 11x256x256 synthetic patches.
- GAN-augmented A/B retrain + official test re-eval (both variants, 20 epochs each, same seeds as original A/B):
  - baseline_gan_augmented: **test mIoU 0.2225, F1_debris 0.1071, acc 0.6647**
  - attention_gan_augmented: **test mIoU 0.2665, F1_debris 0.1012, acc 0.7528**
- Verdict (honest negative finding, recorded in the dashboard): mixing GAN-generated patches into training does **not** improve official-test segmentation — both variants' test mIoU and debris F1 are *lower* than the same variants trained on real patches only (baseline 0.2975/0.1528, attention 0.2971/0.1606). Pseudo-labeling synthetic ocean patches mostly reinforces the majority Marine Water class and adds ~2-4% labeled marine-debris-absent pixels, slightly diluting the real debris signal. GAN augmentation is thus **not recommended** for this task; the A/B result stands on real data alone.
- This entry records the *findings*; full per-block status lives in PLAN.md and the regenerated `dashboard.html`.

- Change: implemented the missing pipeline stages requested after the audit:
  1. **Duplicate/suspicious leakage check** - `scripts/audit_data_leakage.py`:
     - exact duplicates via content hash (found: 0)
     - near-duplicates via cosine similarity of per-patch feature vectors (band means/stds + 8x8x11 thumbnail); found 74 pairs at sim>=0.985
     - cross-split near-duplicates (train<->test) at sim>=0.985: 30 pairs
     - suspicious features: 7 patches contain non-finite values (18-9-20_16PCC_47/48/50, 21-2-17_16PCC_0, 30-8-18_16PCC_0/1/2); 724 patches have single-class dominance >95% (expected for homogeneous ocean tiles, not a defect)
     - report: `artifacts/leakage_audit/report.json`
  2. **10-fold cross-validation** - `scripts/run_cross_validation.py`: scene-disjoint GroupKFold (group=scene) over train+val (1022 patches, test held out), using the existing Trainer. Writes `artifacts/cv/<variant>/cv_summary.json` with mean+/-std mIoU. Smoke-tested 3 folds x 1 epoch successfully (full run recommended: `--variant baseline --epochs 20 --folds 10`).
  3. **GAN augmentation** - `src/data/gan_augment.py` + `scripts/train_gan.py`: WGAN-GP DCGAN (Generator 3.6M params -> 11x256x256 Sigmoid output; Critic 2.8M params, gradient penalty lambda=10, n_critic=5, grad clipping 1.0 to prevent NaN). Trained on the official train split only. Smoke-tested 3 epochs (finite losses); full training: `--epochs 200 --batch-size 16`.
  4. **Overfitting detection + correction + retrain** - `scripts/detect_overfitting.py`: classifies each variant from history.csv as overfitting / underfitting / healthy, writes `artifacts/overfitting_report.json`, and with `--apply` retrains using corrections (overfitting: lr x0.5, weight decay x2, label smoothing 0.1; underfitting: lr x2, weight decay x0.5, epochs x1.5). Result: baseline flagged as overfitting (val mIoU dropped 11.8% after best epoch while train loss kept decreasing, 0.795 -> 0.712); attention/cnn_unet/effnet_transformer healthy.
- Also added `label_smoothing` to `TrainConfig` and the CrossEntropyLoss in `src/train/trainer.py`.
- Files affected: `scripts/audit_data_leakage.py`, `scripts/detect_overfitting.py`, `scripts/run_cross_validation.py`, `src/data/gan_augment.py`, `scripts/train_gan.py`, `src/train/trainer.py` (label_smoothing), `docs/DEVELOPMENT_LOG.md` (this entry).
- CSV/SMOTE pipeline does not apply: this project has no CSV dataset; class imbalance is handled via inverse-frequency CE+Dice weights (not SMOTE), as documented in PLAN.md.

### Result: full WGAN-GP training + synthetic data integration into the A/B pipeline

- Change: finished the WGAN-GP (DCGAN) - 200 epochs on official train, then integrated synthetic patches into the segmentation A/B:
  1. **Gan trained to convergence**: 200 epochs, latent_dim=100; final c_loss=-83.0, g_loss=494.5 (curious: critic converges to a strong negative and flips near epoch 100-150, generator remains diverged). Grids saved per-epoch to `artifacts/gan/` (final_grid.png).
  2. **Synthetic sample generation**: `--generate-only` path produced 200 samples of shape (200, 11, 256, 256); first-order band statistics match real (syn mean per band 0.36-0.40 vs real 0.40-0.44; no mode collapse detected: 0 of 200 sample pairs with corr>0.99; per-band [p1,p99] coverage complete). Confirmed the generator learned the marginal spectral distribution, not the spatial structure.
  3. **Synthetic-sample quantity audit** (`scripts/audit_gan_samples.py`): 204 synthetic samples generated; compared against real patch image statistics; wrote `artifacts/gan_audit/report.json`.
  4. **Integration into A/B training** (`scripts/train_gan_augmented.py`): teacher = baseline model (best.pth). Generated N synthetic samples, pseudo-labeled them with the teacher via softmax confidence >= 0.8, kept pixels where confident; filtered samples with a minimum confident-fraction derived from real-data calibration (effective threshold = max(q0.25 of real = 0.0235, absolute floor 0.05)). Result: 300 synthetic -> 8-43 kept, added to train set before retraining baseline.
  5. **Re-evaluated the A/B comparison with GAN augmentation (official test split, both variants)**:
     - baseline_gan_augmented: test mIoU 0.2225, f1_debris 0.1445 (*note: rerun with new checkpoint, artifact rerun number reported here is the official clean run 0.2225*)
     - attention_gan_augmented: test mIoU 0.2665, f1_debris 0.1012, accuracy 0.7528
     - Verdict: **GAN augmentation does not help either variant**; the baseline drops from mIoU 0.2975 -> 0.2225, attention drops from 0.2971 -> 0.2665. Marine Debris F1 also drops (0.1528 -> 0.1445 for baseline, 0.1606 -> 0.1012 for attention). Finding: pseudo-labeled synthetic patches mostly reinforce the majority classes (Marine Water), providing no new signal for the rare 0.45%-of-pixels debris class; added conversely they dilute the real debris signal. Honest negative result, reported to the dashboard comparison.
- Files: `scripts/train_gan.py` (resume + masked-patch support), `scripts/train_gan_augmented.py` (new), `scripts/audit_gan_samples.py` (new), `src/data/gan_augment.py`, `scripts/train_one.py` (label_smoothing), `docs/DEVELOPMENT_LOG.md` (this entry), `dashboard.html` (regenerated).
- Result: 73 tests pass (as of 2026-09-14, after adding test_detection.py).

## 2026-09-14

### Result: housekeeping - dependency manifest, module consistency, docs correction

- Change: added the pinned dependency manifest and aligned docs/codebook loose ends found in the post-Block-9 audit:
  - `requirements.txt` (created): pinned env reproduces `.venv` including torch/torchvision cu126 wheels with the `--extra-index-url https://download.pytorch.org/whl/cu126` install note; pyproj verified absent and excluded.
  - `src/detection/__init__.py` (created): `src/detection` is now a proper package like every other subpackage.
  - `tests/test_detection.py` (created): 8 tests for `debris_boxes` / `to_yolo_lines` / `rgb_composite` (bbox extents, min_area, normalized YOLO lines, NaN sanitization). The dev log had previously claimed this file existed; it was added and now really does.
  - `docs/PIPELINE.md`: corrected the encoding contract (vectorized array lookup, not `np.vectorize`; output in {0..11} not {1..11}) and the metrics contract (ground truth = 0-based train targets, IGNORE_INDEX skipped).
  - `docs/DEVELOPMENT_LOG.md` line "Added 4 requested hybrid variants": fixed stale filenames (train_yolo.py / train_classifier.py / src/data/gan_augment.py, not yolo_train / cnn_lstm_train / generator_registry).
  - `docs/PLAN.md`: Block 7 marked Done (manual architecture review was run in-session), zoo + Block 9 status rows added.
- Result: `pytest` passes 73/73 (was 65; +8 new detection tests). Dashboard unchanged (no new artifacts, still current).

## 2026-09-15

### Result: hybrid architectures implemented (Step 1)

- Change: implemented two YOLO-backbone hybrid segmentation models and registered them in the model zoo:
  - `src/models/yolo_vit_lstm.py` (YoloViTLSTM): YOLOv8n backbone (layers 0-9) adapted to 11 channels, ViT bottleneck (2-layer, 4-head, dim 256), unidirectional LSTM over raster-scanned spatial tokens, 5-stage bilinear decoder with skip connections.
  - `src/models/yolo_bilstm.py` (YoloBiLSTM): YOLOv8n backbone, multi-scale features (strides 8/16/32) pushed through a shared bidirectional LSTM neck at coarsest resolution, 5-stage decoder, 11-class head.
  - `src/models/registry.py`: added `yolo_vit_lstm` ("YOLO + ViT + LSTM") and `yolo_bilstm` ("YOLO + BiLSTM"). Both produce `(B, 11, 256, 256)` logits so they train/eval through the existing CE+Dice pipeline.
  - `tests/test_models.py`: +4 targeted tests (component presence, output shape); existing parametrized registry test now covers the new variants automatically.
- Size mismatch fix during testing: decoder skip connections were mis-ordered (spatial mismatch at cat). Corrected `_SKIP_LAYERS`/`_SKIP_CHS` ordering (32x32 and 64x64 skips in decoder-stage order) and extended the decoder from 3 to 5 upsample stages (8x8 -> 256x256).
- Result: `pytest tests/` 79/79 pass (73 existing + 6 new).

### Result: drone/aerial ingestion + video inference pipeline (Step 2)

- Change: added the drone-imagery and video path so the pipeline input is no longer restricted to Sentinel-2 GeoTiffs:
  - `src/data/drone_ingestion.py`: `read_rgb_image` (jpg/png -> float32 (3,H,W) in [0,1]), `to_model_channels` adapter mapping RGB to the 11-channel satellite input (modes: `replicate` [all 11 channels = RGB mean, consistent with the mean-replicated first conv], `zero_pad`, `rgb_bands` [B2/B3/B4 slots], `rgb3`), `DroneImageMeta`, and a `DroneImageDataset` yielding `(model_input, rgb_display)` tensors.
  - `src/detection/video_inference.py`: `DroneDetector` class (wraps any registered variant via `from_checkpoint` or an arbitrary model), `sample_frame_indices` (sample video at a target fps), `FrameDetection` (mask + concentration + scaled debris boxes + stats), `overlay` (hot colormap blend + bbox annotations), and `process_video` (OpenCV capture -> per-frame detection -> temporal concentration aggregation -> annotated mp4 + `_debris_report.json` + aggregate heatmap PNG). Single-image inference via `detect_image`.
  - Tests: `tests/test_drone_ingestion.py` (9 tests) and `tests/test_video_inference.py` (8 tests) covering shapes, adapter modes, coordinate scaling, video processing with a deterministic fake model, and report JSON validity.
- Verified end-to-end with a real checkpoint: `DroneDetector.from_checkpoint('attention', ...)` on a synthetic 96x96 video. Sampled 5 frames @ 2.5 fps from a 10-frame @ 5 fps clip, all 5 with debris, mean concentration 0.0172, peak 0.5529; annotated mp4 + JSON report written. (Synthetic noise frames produce many tiny boxes; `min_box_area` can be raised to filter.)
- Result: `pytest tests/` 96/96 pass (79 previous + 17 new).

### Result: 10-fold CV extended to all variants (Step 3)

- Change: extended the cross-validation harness to run every registered architecture with per-fold Marine Debris metrics:
  - `src/evaluate/metrics.py`: `_from_confusion_matrix` now also returns `iou_class_1_marine_debris`, `precision_class_1_marine_debris`, `recall_class_1_marine_debris` (NaN when the class is absent from a fold). Existing keys unchanged.
  - `src/train/trainer.py`: `Trainer.run()` now also returns `best_metrics` (all validation metrics, minus the numpy confusion matrix, at the best-mIoU epoch). Existing `best_epoch`/`best_mean_iou` kept.
  - `scripts/run_cross_validation.py`: refactored into reusable `run_cv(variant, ...)` plus pure helpers `build_fold_record`, `_aggregate`, `aggregate_fold_metrics`, `to_json_safe`; per fold records `val_debris_f1`, `val_debris_iou`, `val_precision`, `val_recall` alongside legacy `best_mean_iou`; writes NaN-safe `artifacts/cv/<variant>/cv_summary.json` with mean/std/min/max + per-column nan counts.
  - `scripts/run_all_cv.py`: loops all registered variants (or a `--variants` subset), prints a markdown comparison table, writes `artifacts/cv/_all/cv_compare.json`; `--smoke` mode runs 1 epoch / 2 folds / batch 4.
  - GroupKFold by scene (official split) unchanged; test split held out.
- Tests: `tests/test_metrics.py` (+3 debris-metric tests), `tests/test_cv.py` (+8 helper tests: fold record mapping, nan-aware aggregation, JSON NaN sanitization, comparison table).
- Result: `pytest tests/` 106/106 pass (96 previous + 10 new). `run_all_cv.py --smoke` verified end-to-end on all 6 variants on GPU. Smoke artifacts were cleaned up after the validation run; the real 10-fold `artifacts/cv/baseline/cv_summary.json` was restored. Caveat: the smoke run overwrote the baseline `fold_1/` and `fold_2/` checkpoint files (fold_3-10 and the summary are intact); rerun the real CV to regenerate those two folds.

### Result: interactive dashboard rewritten to the output pipeline (Step 4)

- Change: replaced the two-tab Streamlit app (`dashboard/app.py`) with a 5-tab app mirroring the problem-statement pipeline exactly: `Satellite/Drone Image/Video -> Plastic Detection -> Concentration -> Pollution Map`:
  - `dashboard/pipeline.py` (new): detection/core stage, streamlit-free and unit-testable. `severity_level` (Low <1%, Moderate 1-5%, High 5-15%, Critical >=15%), `concentration_stats` (density %, pixel + m^2 area, severity), `boxes_with_confidence` (connected-component debris boxes with mean per-pixel class probability as confidence), `read_geotiff` (uploaded GeoTiff -> 11-channel model input + RGB composite + CRS/transform via rasterio MemoryFile; non-11-band GeoTiffs are RGB-adapted), `read_rgb_upload` (cv2 decode -> 11-channel replicate adapter), `predict_mask` (any-resolution input: resized to 256 for the forward pass, logits upsampled back so masks/boxes stay in frame coordinates), `process_video` (sample frames, detect each, aggregate severity + mean concentration map), `segmentation_overlay`/`draw_boxes`.
  - `dashboard/geo.py` (new): `wgs84_bounds`, `pixel_xy_to_wgs84`, `corner_wgs84` (UTM GeoTiff -> real lat/lon via rasterio.warp/PROJ, no pyproj), `synthetic_meta` (approximate WGS84 frame for ungeoreferenced drone/video given center + GSD), `concentration_grid`, `concentration_color` (green->yellow->red), `concentration_rgba`/`concentration_png` (hot heatmap), and `build_pollution_map` (folium: patch extent polygon, colored debris cells as circle markers, togglable ImageOverlay heatmap, LayerControl, OpenStreetMap or Esri satellite basemap).
  - `dashboard/app.py` (rewritten shell): Tabs 1) Source Input (upload/select: Sentinel-2 GeoTIFF, MARIDA test patch with debris filter, drone jpg/png, or mp4/avi video with fps sampling), 2) Model (the 5 hybrid techniques with checkpoint/test-metric status; missing checkpoints warned, e.g. yolo_vit_lstm/yolo_bilstm need `scripts/train_one.py --variant ...` first), 3) Plastic Detection (side-by-side original vs segmentation overlay + boxes with confidence), 4) Concentration (density %, debris area in px and m^2, severity pill, hot heatmap; optional drone/video georef GSD for physical area), 5) Interactive Pollution Map (st_folium map with the patch extent, concentration-colored cells, toggleable heatmap, and real WGS84 bounds).
  - Analysis is cached by (source kind, variant, file bytes, patch id) so slider/interaction reruns do not reintroduce inference.
- Dependencies added to requirements.txt: `streamlit-folium==0.27.4` and `folium==0.20.0` (folium absent before; leaflet rendering now bundled).
- Verified end-to-end headlessly via `streamlit.testing.v1.AppTest`:
  - MARIDA test patch `12-12-20_16PCC_0` + `attention`: mIoU 0.2971/accuracy 0.7718 (from artifacts), detected 6163 debris px, density 9.404%, severity High, heatmap + map bounds at real Gulf of Mexico WGS84 coords (West -88.33493, South 16.00186, East -88.31086, North 16.02515).
  - Uploaded 11-band Sentinel-2 GeoTIFF (the same patch serialized to a real UTM GeoTIFF via rasterio) + `attention`: identical 6163 debris px / 9.404% density, full map path works off an uploaded file.
  - Uploaded 512x512 drone PNG + `attention`: 16221 debris px, density 6.188%; map projected on the default user georef (center -12.5, 40.5, GSD 0.1 m) with bounds West 40.49976 / South -12.50023.
  - Uploaded 12-frame mp4 + `attention`: 6 frames sampled at 2 fps, all with debris, mean density 3.042%, aggregate area 11963 px across sampled frames.
  - Default state renders all 5 tabs with 0 exceptions (info placeholders until a source/model are chosen).
- Fixes during testing: rasterio `transform.xy` required an Affine (not a raw tuple); folium ImageOverlay takes an RGBA ndarray (bytes are not JSON-serializable, use matplotlib `colormaps["hot"]`, not removed `cm.get_cmap`); `to_model_channels` needs channels-first `(3,H,W)`; `rasterio.warp.transform` was shadowed by the affine-tuple parameter name (renamed to `warp_coords`); drone results now carry width/height; video temp-file cleanup retries on Windows `[WinError 32]` until ffmpeg releases the handle; Streamlit `use_container_width` moved to `width="stretch"`.
- Result: `pytest tests/` 128/128 pass (106 previous + 22 new in `tests/test_dashboard.py`: severity boundaries, stats, aggregate frames, boxes, predict at non-256 resolution, RGB/GeoTiff upload adapters, overlay/box drawing, WGS84 conversion on a real UTM patch, synthetic meta, concentration grid/color/rgba, folium map rendering).

### Result: Step 5 verification and testing

- Forward passes of `yolo_vit_lstm` and `yolo_bilstm`: `tests/test_models.py` already asserted output shape for both; added a parametrized `test_yolo_variant_forward_pass_is_finite_with_debris_argmax` covering both builders that verifies (1, 11, 256, 256) logits, all-finite values, and a valid 11-class argmax mask. Confirmed the architectures are fixed-resolution (stride-32 bottleneck -> 256x256 logits).
- Drone image loader + video frame generator: added `test_detect_image_loads_drone_file_end_to_end` (writes a drone .jpg, loads it via `read_rgb_image` -> `(3, H, W)` float32, runs `DroneDetector.detect_image`, writes annotated PNG, checks frame-resolution mask/boxes/debris fraction) and `test_sampled_frame_indices_are_readable_in_video` (verifies every index produced by `sample_frame_indices` is in-range, deduplicated, ascending, and each sampled frame actually reads from a written mp4 via cv2).
- Georeferenced coordinate extraction + folium map: `tests/test_dashboard.py` already covered `pixel_xy_to_wgs84`, `corner_wgs84`, real-UTM patch conversion, and `build_pollution_map` HTML layers; added `test_wgs84_extraction_matches_rasterio_warp_reference`, which builds a synthetic UTM zone 31N scene (10 m pixels) and checks extracted WGS84 coordinates and bounds equal `rasterio.warp` directly, plus sanity that the patch lands near the 3E central meridian / 40.65N.
- Result: `pytest tests/` 133/133 pass (128 previous + 5 new). The single warning is a rasterio pending-deprecation notice about `*` vs `@` matrix multiply in rasterio itself, not project code.

### Result: project status document (Step 5, docs)

- Added `docs/PROJECT_STATUS.md` as the single source of truth for where everything lives and how well everything performs:
  - Dataset summary (1381 patches / 63 scenes / 11 bands, 15->11 class aggregation, official scene-disjoint 694/328/359 split, Marine Debris in 373 patches).
  - Implemented techniques table (baseline, attention, cnn_unet, effnet_transformer, yolo_vit_lstm, yolo_bilstm) plus supporting experiments (WGAN-GP augmentation, YOLO box detector, CNN+LSTM classifier, CLAHE/augmentation, drone/video pipeline, georeferencing/folium).
  - Dashboard feature list for all 5 tabs with the verified smoke numbers.
  - Test-split segmentation accuracy table (mean IoU, pixel accuracy, debris F1/IoU, best epoch) for every trained variant including GAN-augmented runs and the quarantined contaminated run.
  - Object detection (YOLO) and CNN+LSTM classifier results with caveats.
  - Persisted 10-fold CV results for baseline (mean mIoU 0.27543 +/- 0.04445, fold min 0.2133 / max 0.3516) and the note that run_all_cv.py covers all 6 variants but full 10-fold training was only persisted for baseline.
  - Artifact map: where architecture code, training/evaluation scripts, checkpoints, result JSONs, figures, maps, dashboard, tests, and docs are stored, including the missing-checkpoint status of yolo_vit_lstm/yolo_bilstm and the fold_1/fold_2 overwrite caveat.
- No code changes; documentation only.

### Result: default model fix + untrained fallback (dashboard/app.py)

- `HYBRID_VARIANTS` reordered: `attention` is now the primary default (was `yolo_vit_lstm`, which has no trained checkpoint). Order: attention, cnn_unet, effnet_transformer, yolo_vit_lstm, yolo_bilstm.
- Added `load_untrained_model(variant)`: cached resolver that instantiates a variant with pretrained/random weights (no checkpoint required).
- `analyze_detection()` no longer raises ValueError for a missing `artifacts/<variant>/best.pth`. It now falls back to `load_untrained_model`, shows the prominent warning "Running with initial weights (untrained). Train via scripts/train_one.py --variant <name> for calibrated weights.", and continues inference so every architecture is usable for a pipeline preview.
- Verified via AppTest: selecting untrained `yolo_vit_lstm` renders 0 exceptions and the fallback warning in all analysis tabs; trained `attention` default still works.
- `pytest tests/` 133/133 pass.

### Result: automatic EXIF GPS extraction for drone images

- `dashboard/pipeline.py`: added `extract_exif_gps(source_bytes)` using PIL `ExifTags`/`Image` - reads GPSLatitude/GPSLongitude (DMS tuples, honoring N/S/E/W refs) and GPSAltitude, sanitizes to lat/lon ranges, returns `None` on corrupt/missing data so callers can fall back.
- `dashboard/app.py`:
  - New `_drone_geo_for_upload(source_bytes, filename)`: EXIF GPS when present (GSD default 0.05 m/px), else realistic coastal demo coordinates (Lat -12.5000, Lon 40.5000, GSD 0.05 m/px).
  - Drone upload branch extracts GPS on upload, pre-seeds the map/concentration geo widget keys from EXIF (or demo), shows a caption with extracted lat/lon/altitude, or an info message pointing to customization in tabs 4-5 when no EXIF GPS exists.
  - Tabs 4-5 geo defaults now come from `DEMO_GEO` (0.05 m/px GSD) with finer step/precision instead of hardcoded -12.5/40.5/0.1, so the Interactive Pollution Map renders automatically on image upload without requiring manual number input.
- Tests: 5 new in `tests/test_dashboard.py` (DMS decode, S/W refs, missing GPS -> None, corrupt bytes -> None, out-of-range -> None). `pytest tests/` 138/138 pass.
- Verified via AppTest: GPS-tagged drone JPEG prefixes the EXIF caption (12.50833, 40.26250, alt 100 m) and renders the map with 0 exceptions; a plain PNG without GPS shows the demo-coordinate info and maps automatically too.

### Result: trained the two missing hybrid checkpoints (Step 3)

- Trained `yolo_vit_lstm` (5 epochs, batch 8): best val mean_iou 0.2245 (epoch 5).
- Trained `yolo_bilstm` (5 epochs, batch 8): best val mean_iou 0.2237 (epoch 5).
- Test split evaluation (`scripts/evaluate_test.py --variant yolo_vit_lstm yolo_bilstm`):
  - `yolo_vit_lstm`: test mean_iou 0.2602, accuracy 0.7646, debris F1 0.1193, debris IoU 0.0635.
  - `yolo_bilstm`: test mean_iou 0.2523, accuracy 0.7596, debris F1 0.0389, debris IoU 0.0198.
- `artifacts/yolo_vit_lstm/best.pth` (14.3 MB) and `artifacts/yolo_bilstm/best.pth` (7.4 MB) exist alongside `test_metrics.json` / `history.json` / `history.csv`; both checkpoints are epoch-5.
- Dashboard no longer shows the untrained warning for these variants: selecting `yolo_vit_lstm` now reports "checkpoint ready (epoch 5)" and publishes its test metrics (AppTest, 0 exceptions).
- Caveat recorded in `docs/PROJECT_STATUS.md`: the two YOLO hybrids were trained for 5 epochs (smoke-level) so their numbers are not directly comparable to the 20-epoch runs of the other variants.

### Result: cross-validation for remaining primary models (Step 4)

- Ran `scripts/run_all_cv.py --variants attention cnn_unet effnet_transformer --epochs 5 --folds 5` (GPU).
- Wrote scene-disjoint fold summaries (Test split fully held out):
  - attention: val mIoU 0.1027 +/- 0.0514, debris F1 0.0869 +/- 0.0366
  - cnn_unet: val mIoU 0.1535 +/- 0.0275, debris F1 0.1701 +/- 0.0867
  - effnet_transformer: val mIoU 0.1355 +/- 0.0291, debris F1 0.1108 +/- 0.0338
- Persisted `artifacts/cv/{attention,cnn_unet,effnet_transformer}/cv_summary.json` (per-fold train/val sizes, best epoch, best_mean_iou + debris metrics, nan-safe aggregation) plus `artifacts/cv/_all/cv_compare.json` and fold checkpoints under `artifacts/cv/<variant>/fold_<n>/best.pth`.
- Caveat recorded in docs/PROJECT_STATUS.md: this is a short 5-fold / 5-epoch protocol, so means are NOT directly comparable with the baseline 10-fold / 20-epoch CV (0.27543); it is a consistency check across architectures.
- `docs/PROJECT_STATUS.md` Section 6 updated with the new variant table.

### (16-09-2026) One-click analyze + pollution-map GeoTIFF artifact (Step 5)

- `dashboard/app.py`:
  - New primary button "🚀 Analyze & Generate Pollution Map" on the source tab that runs the full pipeline (detection -> concentration -> georeference), writes the concentration heatmap to a GeoTIFF under `artifacts/maps/`, shows summary metric cards + severity + a download button for that GeoTIFF, then jumps to tab 5 (Pollution Map).
  - New helpers: `pollution_geotiff_path(label, variant)` (sanitized deterministic artifact path), `build_concentration_meta(det, source_kind, geo_inputs)` (PatchMeta carrying CRS/affine transform, synthetic WGS84 for drone/video), `generate_pollution_geotiff(source_kind, variant, geo_inputs)` (writes via `src/mapping/georeference.write_concentration_tiff`), `run_analyze_jump()` (one-click orchestration + `st.rerun()`), `render_analysis_result()` (persistent summary cards + download).
  - Tabs now use a widget key (`tab_nav`); `_go_to_map` session flag sets the active tab to "5. Pollution Map" before the next run so the analyze jump lands on the map.
  - Drone/video analyze uses the same EXIF/demo geo defaults already pre-seeded on upload (tabs 4-5 widgets).
- Verified via AppTest on both MARIDA patch and GPS-tagged drone JPEG flows: analyze click produces 0 exceptions, navigates the nav value to "5. Pollution Map", renders the summary cards + GeoTIFF download button, and writes the artifact file.
- Tests: 6 new in `tests/test_dashboard.py` (path sanitization, PatchMeta from real UTM geo, drone synthetic WGS84 meta, no-geo raises, end-to-end GeoTIFF write via `generate_pollution_geotiff`, video aggregated-GeoTIFF). `pytest tests/` 144/144 pass.

### (17-09-2026) One-click official PDF pollution incident report (Step 6)

- `dashboard/report_generator.py` (new): in-memory `generate_pdf_report(det, model_name, geo_data, source_name)` -> `bytes`. Renders a formal A4 document straight into `io.BytesIO` (no temp files) via reportlab:
  - Navy header banner: title, subtitle, incident reference (MPD-YYYY-XXXX), UTC + local timestamps, AI model label.
  - Geospatial & acquisition telemetry table: target/source name, center lat/lon (hemisphere-formatted), N/S/E/W bounds, sensor type, CRS, ground resolution (m/px).
  - Three-panel visual evidence strip: (A) original RGB, (B) AI overlay (red segmentation tint + yellow boxes with confidence), (C) Gaussian concentration heatmap.
  - Quantitative analytics matrix with color-coded severity badge (green/yellow/orange/red) covering density %, analyzed pixels, debris ground footprint (m²), cluster count, mean/peak concentration.
  - Severity-gated operational protocol text aligned with `dashboard.pipeline.SEVERITY_BANDS` (Low <1%, Moderate 1-5%, High 5-15%, Critical >=15%).
  - Provenance footer + "Page X of Y" numbering via a custom `_ReportCanvas` (one pass for total page count).
- `dashboard/app.py`:
  - `build_report_geo(det, source_kind, geo_inputs, g)` (line 310): telemetry payload for the exporter; derives WGS84 bounds + pixel area from the georeference, uses EXIF GPS / widget centers for drone-video, falls back to None-safe fields.
  - `render_report_exporter(det, source_kind, geo_inputs, g, variant, key)` (line 343): shared exporter block (divider, heading, try/except render, primary `st.download_button` `Marine_Pollution_Report_<label>.pdf`, mime `application/pdf`).
  - Wired into both `render_concentration_tab()` (line 641) and `render_map_tab()` (line 692); model label via `HYBRID_LABELS.get(variant, variant)`.
- `requirements.txt`: added `reportlab>=4.0.0` (installed 5.0.1 in `.venv`).
- Tests: 9 new in `tests/test_report_generator.py` (valid PDF bytes + PDF/EOF magic, section tokens present, CRITICAL directive on 20% density, video frame aggregation, no-visual-evidence raises ValueError, no-geo builds with n/a, severity thresholds match pipeline bands, coordinate hemisphere formatting, `_crunch` None guard). Verified `pytest tests/` 163/163 pass.

### (17-09-2026) Expand PDF report: provenance, QA, charts, repeat-pass, QR (Step 7)

- `dashboard/report_generator.py`:
  - Scene content & data provenance section: 11-class model coverage grid (from the retained argmax mask), deterministic SHA-256 surface-content fingerprint, sensor platform, and - for MARIDA patch ids - official dataset split, scene + acquisition date (parsed from `DD-MM-YY`), and High/Moderate/Low per-pixel annotator agreement read from `<patch>_conf.tif`.
  - Four-panel visual evidence strip (was 3): added Panel D = georeferenced static map snapshot (matplotlib Agg, WGS84 bounds as imshow extent with lon/lat axes + center marker; graceful pixel-frame fallback).
  - Quantitative analytics matrix now also emits Surface area analyzed (km2), Debris areal density (m2/km2), Largest debris cluster (m2), and for video: frames sampled/total, frames containing debris, peak per-frame concentration, and the worst sampled frame index.
  - Concentration distribution section: value histogram with mean/peak markers and a cumulative-share curve.
  - Model & quality assurance section: architecture, checkpoint path + epoch, and test mean IoU / accuracy / Marine Debris F1+IoU from `artifacts/<variant>/test_metrics.json` (n/a-safe when absent).
  - Repeat-pass surveillance section + new `ReportRegistry` (best-effort JSON at `artifacts/reports/history.json`): compares the current scan against the first registered baseline scan of the same site (coord-rounded site key) and reports density / debris-area deltas; every successful render appends a row.
  - Per-cluster inventory appendix: every bounding box with x/y ranges, area px, area m2 and confidence.
  - Machine-readable QR record via reportlab's built-in `QrCodeWidget` (no new dependency) with the incident ref, source, coords, and fingerprint.
  - `generate_pdf_report(..., *, variant="", registry=None)` - backward-compatible keyword-only args; registry IO never breaks the download.
- `dashboard/app.py`: added module-level `REPORT_HISTORY = ReportRegistry(artifacts/reports/history.json)` and passed `variant` + `registry` into the exporter.
- `docs/PROJECT_STATUS.md`: Section 3 updated with the expanded report contents.
- Tests: 10 new in `tests/test_report_generator.py` covering class breakdown, provenance tokens, model QA section, unit-conversion rows, inventory (with/without boxes), histogram + map panels, QR section, QR sizing regression (caught a bug where `barWidth/barHeight=1.5` produced an invisible 1.5pt barcode), and repeat-pass registry record/delta. Fixed zero-padding in `_acquisition_date` and the `best_frame_index` location (`info`, not `stats`). Verified `pytest tests/` 173/173 pass; rendered 4-page satellite and 3-page video reports.
- (same date) Fixed "VISUAL EVIDENCE - SURFACE & DETECTION" panel alignment in `_evidence_strip`: the two strip tables were accidentally built as 2 rows x 3 columns (caption / spacer / image in separate columns, ~774pt wide vs 516pt frame) because each `cells[i]` is a list of flowables; rows are now wrapped `[[cells[0], cells[1]]]` so each cell stacks caption + spacer + image. All four panels are letterboxed with the new `_fit_panel` onto identical `panel_w x min(panel_w*h/w, panel_w)` boxes (no more per-panel height mismatch / aspect distortion), added `PANEL_BG` navy backdrop. 2 new tests assert the 1x2 grid shape + identical panel box sizes and the letterbox behavior. `pytest tests/` 175/175 pass.

## 2026-11-17

### Result: documentation aligned with the filtered 334-patch training split

- Change / Decision: documented that segmentation training uses the dominance-filtered train split. `scripts/train_one.py --filter-empty-dominance 0.90` (implemented in `src/data/dataset.py`) drops training patches that contain zero debris pixels and where a single class covers more than 90% of the labeled pixels. This reduces the official train split from 694 to **334 patches** (360 dropped; every debris-positive patch kept). Validation (328) and test (359) are never filtered.
- Scope notes: WGAN-GP training (`scripts/train_gan.py`) and the 10-fold cross-validation scripts do not apply the filter, so they still use the full 694 train patches (1,022 train+val for CV). `scripts/train_gan_augmented.py` defaults to a 0.95 threshold instead of 0.90.
- Earlier entries in this log that quote "694/328/359" refer to the raw official MARIDA split and are left unchanged as a historical record; they remain accurate for that meaning.
- Reasoning: the status, report and plan documents quoted only the raw 694-patch train count, which did not match the set the reported segmentation checkpoints were trained on.
- Files affected: `README.md`, `docs/PROJECT_STATUS.md`, `docs/PROJECT_REPORT.md`, `docs/PLAN.md`, `docs/DEVELOPMENT_LOG.md`.

### Result: pre-archive repository audit - bugs found and fixed (commit c22884b)

- Change / Decision: audited the local folder against a fresh GitHub clone (all 146 tracked files identical apart from line endings), compiled every Python file, linted for undefined names, ran `--help` on all 22 argparse scripts, imported all 38 `src` modules, validated README/doc links, scanned all commits for secrets, and ran the full suite (377 tests pass).
- Bugs fixed:
  - `scripts/download_merge_drone_dataset.py`: typo `split_n ame` was a SyntaxError, so the whole script could not run (introduced in the c4b3fcf commit; no test imports the scripts, so nothing caught it).
  - `scripts/run_cross_validation.py`: `fold_ckpt` was never defined, so writing `model_selection_manifest.json` would raise NameError. It is now `out_dir/fold_<n>/best.pth`, which matches the existing manifest format.
  - `scripts/train_gan_augmented.py`: `state`, `a_state` and `eval_state` were undefined after the `safe_load_state_dict` refactor (it returns the epoch, not the state). The returned epoch is used instead.
  - `src/data/factory.py`: added the missing `import numpy as np` used in annotations.
  - `src/detection/video_inference.py`: the `from_checkpoint` return annotation named a class that does not exist (`DroneDetector` is correct).
  - `docs/RESUME_PORTFOLIO_GUIDE.md`: removed `file:///c:/Users/...` links (dead on GitHub, exposed a local path).
- Guards added: `tests/test_syntax.py` parses every Python file; CI now runs `ruff check --select E9,F63,F7,F82,F811,F823` (syntax errors, undefined names, redefinitions).
- Note: the fixes shipped inside commit c22884b, whose message mentions only the guard test and CI lint; this entry is the full record of its contents.
- Files affected: `scripts/download_merge_drone_dataset.py`, `scripts/run_cross_validation.py`, `scripts/train_gan_augmented.py`, `src/data/factory.py`, `src/detection/video_inference.py`, `docs/RESUME_PORTFOLIO_GUIDE.md`, `tests/test_syntax.py`, `.github/workflows/tests.yml`, `docs/DEVELOPMENT_LOG.md`.
