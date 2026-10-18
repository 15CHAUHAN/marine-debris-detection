# Skills Registry

Registry of agent skills loaded for this project. Each gives the agent procedural instructions. Loaded via the Skills CLI (`npx skills use`).

| Skill | Source | Purpose in project |
|---|---|---|
| computer-vision-opencv | mindrally/skills | Image preprocessing, CLAHE, color spaces, OpenCV conventions |
| deep-learning-pytorch | mindrally/skills | U-Net construction, training loop, PyTorch best practices |
| matplotlib-best-practices | mindrally/skills | Loss curves, heatmaps, result plots |
| find-skills | vercel-labs/skills | Discovery of further skills on demand |
| improve-codebase-architecture | mattpocock/skills | Architecture review HTML report, module deepening, CONTEXT.md and ADR conventions |
| tao-train-segformer | nvidia/skills | Reference for semantic segmentation training patterns (SegFormer) |
| video-frames | openclaw/openclaw | Frame extraction if drone video sources are added later |

## Runtime dependencies (not skills, installed package level)

Run inside the Python 3.12 venv:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy opencv-python rasterio segmentation-models-pytorch albumentations scikit-learn matplotlib pandas tqdm Pillow
```

## Verification commands after install

```
python -c "import torch, torchvision, cv2, rasterio, albumentations, smp, sklearn, matplotlib; print('ok')"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `ok` and `True NVIDIA GeForce RTX 4050 Laptop GPU`.