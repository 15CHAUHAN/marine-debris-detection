# Project Rules & Custom Skills for TECH-ATHON Marine Debris Detection

This project detects marine plastic pollution using Sentinel-2 multispectral satellite imagery (MARIDA / Plastic L2A) and RGB drone/photo litter detection (TACO).

## Active Skills & Rulepacks
The specialized skills for this repository are defined in `.agents/skills/`:
1. `windows-powershell-env`: Windows 11 PowerShell execution rules, `D:\TECH-ATHON\.venv\Scripts\python.exe`, `num_workers=0`, and RTX 4050 GPU optimization.
2. `geospatial-rasterio-processing`: 11-channel Sentinel-2 band definitions, 2%-98% percentile normalization, and Affine transform preservation.
3. `yolo-ultralytics-custom`: YOLOv8s TACO 7-class macro training, high-precision loss balance (`cls=2.0`, `box=10.0`), and artifact path contracts.
4. `pytorch-multiclass-segmentation`: 11-class imbalanced segmentation rules, Focal + Dice compound loss, and Class 1 (Marine Debris) F1 optimization.
5. `streamlit-dashboard-patterns`: Decoupled Streamlit tabs, `@st.cache_resource` caching, and fast Folium map rendering.

## Core Directives for OpenCode
- **Working Directory:** `D:\TECH-ATHON`
- **Interpreter:** Always use `D:\TECH-ATHON\.venv\Scripts\python.exe`
- **No Data Leakage:** Never mix training patches with validation/test patches or synthetic GAN pseudo-label sets.
