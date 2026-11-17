# Marine Plastic Debris Detection (TECH-ATHON)

[![Tests](https://github.com/15CHAUHAN/marine-debris-detection/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/15CHAUHAN/marine-debris-detection/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-cu126-orange?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-green)

A dual-scale deep learning pipeline that detects and maps marine plastic pollution at **satellite scale** (Sentinel-2 multispectral, MARIDA benchmark) and **close range** (RGB drone images / video), then turns the detections into a georeferenced **pollution concentration map** with an interactive dashboard.

```
Satellite / Drone Image / Video  ->  Plastic Detection  ->  Concentration  ->  Pollution Map
```

## Architecture

```
Satellite path:  11-band Sentinel-2 patches (256x256) -> U-Net variants (best: scSE attention) -> 11-class segmentation map
Drone path:      RGB images / video frames (resized to 640x640) -> YOLOv8s                      -> 7-class litter bounding boxes
Both paths:      detections -> Gaussian concentration heatmap -> georeferenced GeoTIFF + Folium map + PDF report
```

## Highlights

- **Six segmentation architectures** behind one registry (`src/models/registry.py`): U-Net (ResNet-34), U-Net + scSE attention, a compact from-scratch CNN U-Net, EfficientNet + Transformer U-Net, and two YOLOv8n-backbone + LSTM hybrids.
- **11-channel Sentinel-2 input** (B1-B8, B8A, B11, B12) with the 15 MARIDA classes aggregated to an 11-class working set. Target class: **Marine Debris**.
- **Leakage-safe evaluation**: scene-disjoint official split, scene-grouped cross-validation, and a duplicate / near-duplicate audit (`scripts/audit_data_leakage.py`).
- **Streamlit dashboard** with five tabs (input, model, detection, concentration, map) and a one-click PDF incident report.
- **Georeferenced outputs**: debris concentration exported as a float32 GeoTIFF (source CRS + affine transform), viewable in QGIS, plus interactive Folium maps.
- **Drone / video pipeline**: RGB-to-11-channel adapter, frame sampling, per-frame detection, annotated output video and JSON report.
- **TACO litter detector**: YOLOv8s trained on TACO collapsed into 7 macro classes for RGB photos.
- Pytest suite covering encoding, splits, leakage audit, losses, metrics, models, mapping, dashboard and video inference.

## Results (official MARIDA test split, 359 patches)

| Model | mean IoU | Pixel acc. | Debris F1 | Debris IoU |
|---|---|---|---|---|
| U-Net ResNet-34 (`baseline`) | 0.3681 | 0.7611 | 0.1152 | 0.0611 |
| U-Net + scSE (`attention`) | **0.3745** | **0.7788** | **0.1626** | **0.0885** |
| `cnn_unet` (from scratch) | 0.1405 | 0.2340 | 0.1288 | 0.0688 |
| `effnet_transformer` | 0.2831 | 0.7231 | 0.0502 | 0.0258 |
| `yolo_vit_lstm` (5 epochs) | 0.2602 | 0.7646 | 0.1193 | 0.0635 |
| `yolo_bilstm` (5 epochs) | 0.2523 | 0.7596 | 0.0389 | 0.0198 |

**Read these honestly.** Marine Debris is a rare, small-object class and it is hard: the best debris IoU is about 0.09 and debris F1 about 0.16. `baseline` and `attention` are effectively tied on mean IoU; `attention` is slightly better on debris. The two YOLO hybrids were trained for only 5 epochs, so they are not directly comparable to the 20-epoch runs. WGAN-GP synthetic augmentation did **not** help and is not recommended. Full details, cross-validation numbers and negative results (box detector, CNN+LSTM classifier) are in [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

**Drone / RGB path (YOLOv8s).** Trained on TACO merged with aerial marine-litter data (4,175 images: 2,811 train / 904 val / 460 test, 7 macro classes), it reaches **mAP@50 = 0.4444** on the held-out test split. These figures come from the project notes in [`docs/RESUME_PORTFOLIO_GUIDE.md`](docs/RESUME_PORTFOLIO_GUIDE.md). Training uses `cls=2.0` and `box=10.0` to favour precision.

**Plastic L2A.** An external Sentinel-2 campaign prepared by `scripts/prepare_plastic_l2a.py`. It has no ground-truth labels yet, so it is used only for qualitative checks, not for reported metrics.

## Repository layout

```
src/
  data/         ingestion, 15->11 class encoding, splits, augmentation (CLAHE), drone/RGB adapter, GAN augmentation
  models/       model variants + registry, weight initialisation
  train/        trainer and compound losses (class-weighted CE + Dice / focal)
  evaluate/     mIoU, accuracy, Marine Debris F1/IoU
  evaluation/   GAN quality metrics
  detection/    label handling, video inference
  mapping/      Gaussian concentration heatmap, GeoTIFF georeferencing
  utils/        checkpoints, spectral indices, domain validation
dashboard/      Streamlit app (app.py), detection core (pipeline.py), geo/map (geo.py), PDF report (report_generator.py)
scripts/        training, evaluation, cross-validation, dataset preparation, audits, demo video
splits/         official MARIDA train / val / test patch lists
tests/          pytest suite
docs/           pipeline contracts, plan, status, development log, project report
CONTEXT.md      domain glossary (patch, scene, class sets, leakage, ...)
```

## Setup

Developed on **Windows 11, Python 3.12, NVIDIA RTX 4050 (6 GB), CUDA 12.6**.

```powershell
git clone https://github.com/15CHAUHAN/marine-debris-detection.git
cd marine-debris-detection

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

`torch` / `torchvision` are pinned to the CUDA 12.6 builds. On a machine without a compatible GPU, install the default PyPI builds of those two packages instead and drop the pinned `+cu126` suffix in `requirements.txt`. On Windows use `num_workers=0` in data loaders.

## Data and model weights (not included)

Datasets, checkpoints and run outputs are **not in this repository** because of their size. They are listed in `.gitignore` and you need to obtain them yourself.

| What | Where to get it | Place it at |
|---|---|---|
| **MARIDA** Sentinel-2 patches | [Zenodo, MARIDA](https://doi.org/10.5281/zenodo.5151941) (Kikaki et al., 2022) | `patches/S2_<scene>/...` at the repo root |
| **TACO** litter images | [tacodataset.org](http://tacodataset.org/) | prepared with `scripts/prepare_taco.py` |
| Trained checkpoints | Not published. Train them (below). | `artifacts/<variant>/best.pth` |

MARIDA official split (scene-disjoint, verified):

| Split | Raw patches | Filtered patches | Scenes |
|---|---|---|---|
| Train | 694 | 334 | 36 |
| Val | 328 | 328 (no filtering) | 12 |
| Test | 359 | 359 (no filtering) | 15 |

The filtered train count comes from `--filter-empty-dominance 0.90` in `scripts/train_one.py`: it drops training patches that have no debris pixels and where a single class covers more than 90 % of the labeled pixels. Validation and test are never filtered.

For the drone path, `scripts/download_merge_drone_dataset.py` merges extra aerial litter datasets (for example the Zenodo marine litter set) into TACO's 7-class YOLO format. The 7 classes are Bottle, Can, Plastic bag, Cup, Wrapper/Sachet, Container and Other trash.

The official split files in `splits/` are included. Check your environment and data layout with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_env.py
```

## Usage

Train one variant on the official split:

```powershell
.\.venv\Scripts\python.exe scripts\train_one.py --variant attention --epochs 20 --batch-size 8
```

Quick smoke run, evaluation, and a debris concentration GeoTIFF for one patch:

```powershell
.\.venv\Scripts\python.exe scripts\train_one.py --variant baseline --smoke
.\.venv\Scripts\python.exe scripts\evaluate_test.py
.\.venv\Scripts\python.exe scripts\map_patch.py
```

Scene-disjoint cross-validation, leakage audit, and the TACO YOLO detector:

```powershell
.\.venv\Scripts\python.exe scripts\run_cross_validation.py
.\.venv\Scripts\python.exe scripts\audit_data_leakage.py
.\.venv\Scripts\python.exe scripts\prepare_taco.py
.\.venv\Scripts\python.exe scripts\train_taco.py
```

Each script documents its own flags (`--help`).

## Dashboard

After training at least one variant (the dashboard reads `artifacts/<variant>/best.pth`):

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Inputs: a Sentinel-2 GeoTIFF, a MARIDA test patch, a drone image, or a drone/coastal video. "Analyze & Generate Pollution Map" runs the whole pipeline in one click and writes a concentration GeoTIFF to `artifacts/maps/`. Tabs 4 and 5 also offer an A4 PDF pollution incident report.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## Methodology notes

- **No data leakage.** Training, validation and test patches are never mixed, and synthetic (GAN) patches never enter evaluation sets. The official MARIDA split is scene-disjoint (36 / 12 / 15 scenes).
- **Preprocessing.** CLAHE and Albumentations augmentation are applied to the training split only.
- **Known data issues.** The audit found 30 near-duplicate pairs that cross split boundaries and heavy class imbalance (727 patches dominated by a single class). Treat debris metrics with that in mind.
- Further reading: [`docs/PIPELINE.md`](docs/PIPELINE.md) (module contracts), [`CONTEXT.md`](CONTEXT.md) (terminology), [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).

## Acknowledgements and citations

- **MARIDA**: Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D. E., Karantzalos, K. (2022). *MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.* PLOS ONE.
- **TACO**: Proenca, P. F., Simoes, P. (2020). *TACO: Trash Annotations in Context for Litter Detection.*
- Built with PyTorch, segmentation-models-pytorch, Ultralytics YOLO, rasterio, Streamlit and Folium.

## License

The code in this repository is released under the [MIT License](LICENSE). The datasets (MARIDA, TACO) and any third-party models or weights keep their own licences, which apply separately.
