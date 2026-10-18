"""Environment and dataset integrity verification (Block 0)."""
import sys
import json
from pathlib import Path

import rasterio
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PATCHES = ROOT / "patches"


def check_imports():
    import torch
    import torchvision
    import cv2
    import albumentations
    import segmentation_models_pytorch as smp
    import sklearn
    import matplotlib
    print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"torchvision {torchvision.__version__}")
    print(f"opencv {cv2.__version__}, rasterio {rasterio.__version__}, albumentations {albumentations.__version__}")
    print(f"smp {smp.__version__}, sklearn {sklearn.__version__}, matplotlib {matplotlib.__version__}")


def check_dataset():
    image_files = sorted(p for p in PATCHES.rglob("*.tif") if not p.name.endswith("_cl.tif") and not p.name.endswith("_conf.tif"))
    mask_files = sorted(PATCHES.rglob("*_cl.tif"))
    conf_files = sorted(PATCHES.rglob("*_conf.tif"))
    print(f"image patches: {len(image_files)}")
    print(f"class masks:   {len(mask_files)}")
    print(f"conf masks:    {len(conf_files)}")
    assert len(image_files) == 1381, f"expected 1381 image patches, got {len(image_files)}"
    assert len(image_files) == len(mask_files) == len(conf_files)

    sample = image_files[0]
    with rasterio.open(sample) as src:
        print(f"sample: {sample.name}")
        print(f"  shape (bands, h, w): {src.count} x {src.height} x {src.width}")
        print(f"  dtype: {src.dtypes[0]}, crs: {src.crs}, transform: {src.transform}")
        img = src.read()
    assert src.count == 11 and src.height == 256 and src.width == 256, "expected 11-band 256x256 patch"

    mask_sample = mask_files[0]
    with rasterio.open(mask_sample) as src:
        mask = src.read(1)
    uniq = np.unique(mask)
    print(f"mask sample {mask_sample.name}: unique values {uniq}, min={uniq.min()}, max={uniq.max()}, dtype={mask.dtype}")
    assert float(uniq.min()) >= 0 and float(uniq.max()) <= 15, "mask values must be in 0..15"
    assert bool(np.all(mask == np.floor(mask))), "mask values must be whole numbers"
    print("  note: masks are stored as float32 with whole-number values (not uint8)")

    with open(ROOT / "labels_mapping.txt", "r") as fh:
        labels = json.load(fh)
    print(f"labels_mapping entries: {len(labels)}")
    assert len(labels) == 1381, "labels_mapping must cover all 1381 patches"
    print("dataset check: ALL PASS")


def check_splits():
    for name in ("train", "val", "test"):
        txt = ROOT / "splits" / f"{name}_X.txt"
        ids = [ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]
        print(f"{name}: {len(ids)} patches")
    print("splits check: PASS")


if __name__ == "__main__":
    check_imports()
    check_splits()
    check_dataset()
    print("ALL BLOCK 0 CHECKS PASSED")