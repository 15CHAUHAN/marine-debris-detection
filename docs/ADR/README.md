# Architecture Decision Records (ADR)

Decision records for this project. Each ADR captures a decision, its context, and its consequences. Never re-litigate a recorded decision without a new ADR or explicit user instruction.

## ADR-0001: Use the 11-class aggregated class set

- Status: Accepted
- Date: 2026-09-11
- Context: MARIDA ships 15 thematic classes. The problem statement names a smaller set (marine debris, natural organic material, sea foam, turbid/sediment water, clouds, waves). The authors aggregate Waves, Cloud Shadows, Wakes, and Mixed Water into Marine Water, producing an 11-class set with published baselines.
- Decision: Train and evaluate on the 11 aggregated classes defined in CONTEXT.md.
- Consequences: Smaller output head, reduced class imbalance, comparability to the paper. Cannot report per-14/15-class granularity.

## ADR-0002: Official test split is the reporting standard

- Status: Accepted
- Date: 2026-09-11
- Context: The official splits are scene-disjoint (verified: 36/12/15 scenes, zero overlap), so they are leak-free by construction.
- Decision: Final benchmark numbers are computed on the official test split (359 patches). The official val split (328 patches) is used for model selection. Scene-grouped 10-fold is optional robustness evidence only.
- Consequences: Reproducible comparison to literature. Test set is never used during development.

## ADR-0003: Python 3.12 venv with cu126 PyTorch

- Status: Accepted
- Date: 2026-09-11
- Context: Host has Python 3.13.2 and 3.12. The 3.13 series risks broken wheels for rasterio and segmentation-models-pytorch. The RTX 4050 driver reports CUDA 12.7, which caps the usable PyTorch CUDA build at the 12.6 series (minor-version forward compatibility).
- Decision: Create the project venv on Python 3.12 and install torch/torchvision from the cu126 index. Do not install system-wide packages.
- Consequences: Older but fully supported Python; guarantees a working rasterio/smp install. Pin python to 3.12.x.

## ADR-0004: A/B scope limited to U-Net vs U-Net+Attention

- Status: Accepted
- Date: 2026-09-11
- Context: The problem statement lists five hybrid architectures (YOLO+ViT+LSTM, CNN+U-Net, EfficientNet+Transformer, YOLO+BiLSTM, U-Net+Attention). Deadline is 10 hours. MARIDA is a pixel-level segmentation task, so detection-centric hybrids are secondary.
- Decision: Ship segmentation A/B only: baseline plain U-Net vs candidate U-Net with scSE attention gates. Detector and temporal branches (YOLO, LSTM/BiLSTM) deferred to future work.
- Consequences: Feasible within deadline. The mapping output still works because segmentation masks are the input to concentration heatmaps.

## ADR-0005: GAN-based augmentation is deferred

- Status: Accepted
- Date: 2026-09-11
- Context: The problem statement asks for GAN-based synthetic augmentation for the minority classes. Building and training a GAN is not feasible within a 10 hour window and carries data-leakage risk if synthetic data is poorly controlled.
- Decision: Do not build a GAN in this iteration. Use inverse class-frequency loss weighting (Dice + cross-entropy) and standard spatial augmentation to handle class imbalance.
- Consequences: Simpler pipeline, smaller imbalance risk, no synthetic data in train/val/test. Revisit only if time permits after core pipeline ships.