"""Download and merge aerial drone & marine litter datasets into TACO YOLOv8 format.

This script expands the TACO dataset with aerial and marine drone imagery to bridge
the domain gap between ground-level pedestrian photos and top-down drone missions.

Supported Data Sources:
1. --zenodo: Direct download and ingestion of Zenodo Record 10993447
   ('Marine Litter Dataset' - drone captured, 5 classes).
2. --roboflow: Ingestion via the Roboflow Universe API
   (requires --api-key, --workspace, --project, and optional --version).
3. --local-dir: Ingestion of any unzipped local YOLOv8 dataset directory.
4. --zip-path: Extraction and ingestion of a local .zip archive.

All incoming classes are automatically mapped to our 7 canonical TACO macro classes:
    0: Bottle
    1: Can
    2: Plastic bag
    3: Cup
    4: Wrapper_Sachet
    5: Container
    6: Other_Trash

Usage:
    # 1. Download and merge the open Zenodo drone marine litter dataset:
    python scripts/download_merge_drone_dataset.py --zenodo

    # 2. Ingest from an existing local folder or downloaded zip:
    python scripts/download_merge_drone_dataset.py --local-dir /path/to/extracted_yolo
    python scripts/download_merge_drone_dataset.py --zip-path /path/to/dataset.zip

    # 3. Pull a drone dataset from Roboflow Universe:
    python scripts/download_merge_drone_dataset.py --roboflow --api-key YOUR_KEY --workspace WORKSPACE --project PROJECT
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

# Fix Windows cp1252 console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TACO_DATASET_DIR = ROOT / "artifacts" / "taco_dataset"
DATA_YAML_PATH = TACO_DATASET_DIR / "data.yaml"

# Canonical 7 Macro Classes
TARGET_CLASSES = [
    "Bottle",          # 0
    "Can",             # 1
    "Plastic bag",     # 2
    "Cup",             # 3
    "Wrapper_Sachet",  # 4
    "Container",       # 5
    "Other_Trash",     # 6
]

# Intelligent keyword mapping from any source class name to canonical class ID
KEYWORD_MAPPINGS = {
    # 0: Bottle
    r"\b(bottle|bottles|glass_bottle|plastic_bottle)\b": 0,
    # 1: Can
    r"\b(can|cans|drink_can|food_can|tin|tins|beverage_can)\b": 1,
    # 2: Plastic bag
    r"\b(bag|bags|plastic_bag|softplastic|soft_plastic|film|polythene|carrier_bag)\b": 2,
    # 3: Cup
    r"\b(cup|cups|disposable_cup|coffee_cup)\b": 3,
    # 4: Wrapper_Sachet
    r"\b(wrapper|wrappers|sachet|sachets|snack_bag|crisp_packet)\b": 4,
    # 5: Container
    r"\b(container|containers|box|carton|tray|punnet|tupperware|takeaway)\b": 5,
    # 6: Other_Trash (all other debris types: hardplastic, styrofoam, masks, ropes, nets, etc.)
    r"\b(hardplastic|hard_plastic|polystyrene|styrofoam|mask|masks|straw|straws|rope|net|glass|metal|wood|paper|cardboard|trash|waste|debris|litter|other)\b": 6,
}


def map_class_name_to_target(name: str) -> int:
    """Map any source category name to our 7 canonical class IDs (0-6)."""
    norm = name.lower().strip().replace("-", "_")
    for pattern, class_id in KEYWORD_MAPPINGS.items():
        if re.search(pattern, norm):
            return class_id
    # Default fallback for unclassified debris items
    return 6


def download_zenodo_dataset(dest_zip: Path) -> Path:
    """Download Zenodo Record 10993447 ('Marine Litter Dataset')."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/records/10993447/files/Marine%20Litter.zip?download=1"
    print(f"[*] Downloading Zenodo 10993447 (Marine Litter Drone Dataset) -> {dest_zip}...")
    
    # Use curl if available for resilient download with progress bar
    curl_bin = shutil.which("curl") or shutil.which("curl.exe")
    if curl_bin:
        cmd = [curl_bin, "-L", "-o", str(dest_zip), url]
        ret = subprocess.run(cmd, check=False)
        if ret.returncode == 0 and dest_zip.exists() and dest_zip.stat().st_size > 10_000_000:
            print("[OK] Download completed via curl.")
            return dest_zip

    # Fallback to urllib with custom User-Agent
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, dest_zip.open("wb") as out:
        shutil.copyfileobj(resp, out)
    print("[OK] Download completed via urllib.")
    return dest_zip


def parse_source_classes(data_yaml_path: Path | None) -> dict[int, int]:
    """Parse incoming dataset's data.yaml to build source_id -> target_id mapping."""
    mapping = {}
    if data_yaml_path and data_yaml_path.exists():
        try:
            import yaml
            with data_yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            names = data.get("names", [])
            if isinstance(names, list):
                for src_id, name in enumerate(names):
                    tgt_id = map_class_name_to_target(str(name))
                    mapping[src_id] = tgt_id
                    print(f"   Mapping source class {src_id} ('{name}') -> Class {tgt_id} ({TARGET_CLASSES[tgt_id]})")
            elif isinstance(names, dict):
                for src_id, name in names.items():
                    tgt_id = map_class_name_to_target(str(name))
                    mapping[int(src_id)] = tgt_id
                    print(f"   Mapping source class {src_id} ('{name}') -> Class {tgt_id} ({TARGET_CLASSES[tgt_id]})")
        except Exception as e:
            print(f"   Warning reading {data_yaml_path}: {e}")

    # Fallback: Zenodo Marine Litter 5 classes if no yaml found
    if not mapping:
        zenodo_defaults = {
            0: "Bottles",      # -> 0
            1: "Hardplastic",  # -> 6
            2: "Masks",        # -> 6
            3: "Polystyrene",  # -> 6
            4: "Softplastic",  # -> 2
        }
        for src_id, name in zenodo_defaults.items():
            tgt_id = map_class_name_to_target(name)
            mapping[src_id] = tgt_id
            print(f"   Using default Zenodo mapping: {src_id} ('{name}') -> Class {tgt_id} ({TARGET_CLASSES[tgt_id]})")
            
    return mapping


def convert_and_merge_split(
    src_img_dir: Path,
    src_lbl_dir: Path,
    split_name: str,
    class_mapping: dict[int, int],
    prefix: str = "drone_",
) -> tuple[int, int, Counter]:
    """Copy images and re-map labels into the target TACO dataset split."""
    tgt_img_dir = TACO_DATASET_DIR / "images" / split_name
    tgt_lbl_dir = TACO_DATASET_DIR / "labels" / split_name
    tgt_img_dir.mkdir(parents=True, exist_ok=True)
    tgt_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in src_img_dir.iterdir() if p.suffix.lower() in img_extensions]

    num_images_merged = 0
    num_boxes_merged = 0
    class_counts = Counter()

    for img_path in images:
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        dest_img_path = tgt_img_dir / f"{prefix}{img_path.name}"
        dest_lbl_path = tgt_lbl_dir / f"{prefix}{img_path.stem}.txt"

        converted_lines = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        src_cls = int(parts[0])
                        tgt_cls = class_mapping.get(src_cls, 6)
                        coords = [float(x) for x in parts[1:5]]
                        # Validate bounding box coordinates in [0, 1]
                        if all(0.0 <= c <= 1.05 for c in coords):
                            coords = [min(max(c, 0.0), 1.0) for c in coords]
                            converted_lines.append(f"{tgt_cls} {' '.join(f'{c:.6f}' for c in coords)}")
                            class_counts[tgt_cls] += 1
                            num_boxes_merged += 1
                    except ValueError:
                        continue

        # Copy image
        shutil.copy2(img_path, dest_img_path)
        # Write rewritten labels
        dest_lbl_path.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")
        num_images_merged += 1

    return num_images_merged, num_boxes_merged, class_counts


def merge_yolo_dataset(source_root: Path, prefix: str = "drone_") -> None:
    """Locate images/labels within source_root and merge into TACO dataset."""
    print(f"\n[*] Scanning source directory: {source_root}")
    
    # Locate data.yaml if present
    yaml_candidates = list(source_root.glob("*.yaml")) + list(source_root.glob("*/*.yaml"))
    source_yaml = yaml_candidates[0] if yaml_candidates else None
    class_mapping = parse_source_classes(source_yaml)

    # Detect directory layout
    total_imgs = 0
    total_boxes = 0
    total_classes = Counter()

    for split in ["train", "val", "test", "valid"]:
        target_split = "val" if split == "valid" else split
        
        # Check standard YOLO layout variations:
        # 1. source/images/train and source/labels/train
        # 2. source/train/images and source/train/labels
        # 3. source/train (images and txt in same directory)
        candidate_img_dirs = [
            source_root / "images" / split,
            source_root / split / "images",
            source_root / split,
        ]
        candidate_lbl_dirs = [
            source_root / "labels" / split,
            source_root / split / "labels",
            source_root / split,
        ]

        found_img_dir = next((d for d in candidate_img_dirs if d.exists() and any(d.iterdir())), None)
        found_lbl_dir = next((d for d in candidate_lbl_dirs if d.exists()), None)

        if found_img_dir and found_lbl_dir:
            n_img, n_box, counts = convert_and_merge_split(
                found_img_dir, found_lbl_dir, target_split, class_mapping, prefix=prefix
            )
            print(f"   [{target_split.upper()}] Merged {n_img} images and {n_box} boxes from {found_img_dir.name}")
            total_imgs += n_img
            total_boxes += n_box
            total_classes.update(counts)

    # If no splits were detected, treat the entire folder as an 80/10/10 un-split dataset
    if total_imgs == 0:
        all_imgs = [p for p in source_root.rglob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if all_imgs:
            print(f"   No train/val/test splits detected. Performing automatic 80/10/10 split on {len(all_imgs)} images...")
            import random
            random.seed(42)
            random.shuffle(all_imgs)
            n_tot = len(all_imgs)
            n_train = int(n_tot * 0.8)
            n_val = int(n_tot * 0.1)
            
            splits = {
                "train": all_imgs[:n_train],
                "val": all_imgs[n_train:n_train+n_val],
                "test": all_imgs[n_train+n_val:],
            }
            for s_name, img_list in splits.items():
                for img in img_list:
                    lbl = img.with_suffix(".txt")
                    if not lbl.exists():
                        # Check labels/ directory
                        lbl = source_root / "labels" / f"{img.stem}.txt"
                    
                    dest_img = TACO_DATASET_DIR / "images" / s_name / f"{prefix}{img.name}"
                    dest_lbl = TACO_DATASET_DIR / "labels" / s_name / f"{prefix}{img.stem}.txt"
                    dest_img.parent.mkdir(parents=True, exist_ok=True)
                    dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                    
                    converted_lines = []
                    if lbl.exists():
                        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    src_cls = int(parts[0])
                                    tgt_cls = class_mapping.get(src_cls, 6)
                                    coords = [float(x) for x in parts[1:5]]
                                    converted_lines.append(f"{tgt_cls} {' '.join(f'{c:.6f}' for c in coords)}")
                                    total_classes[tgt_cls] += 1
                                    total_boxes += 1
                                except ValueError:
                                    continue
                    shutil.copy2(img, dest_img)
                    dest_lbl.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")
                    total_imgs += 1

    print(f"\n[SUMMARY] Merge Summary for {prefix}:")
    print(f"   Total Images Added: {total_imgs}")
    print(f"   Total Bounding Boxes Added: {total_boxes}")
    for cid, cname in enumerate(TARGET_CLASSES):
        print(f"   - Class {cid} ({cname}): {total_classes[cid]} instances")


def update_data_yaml() -> None:
    """Ensure data.yaml is valid, absolute-path aware, and consistent."""
    data_yaml_content = f"""path: {TACO_DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 7

names:
  0: Bottle
  1: Can
  2: Plastic bag
  3: Cup
  4: Wrapper_Sachet
  5: Container
  6: Other_Trash
"""
    DATA_YAML_PATH.write_text(data_yaml_content, encoding="utf-8")
    print(f"\n[OK] Updated {DATA_YAML_PATH}")


def count_dataset() -> None:
    """Print current dataset statistics."""
    print("\n[INFO] Final Combined Dataset Status:")
    for split in ["train", "val", "test"]:
        p_img = TACO_DATASET_DIR / "images" / split
        p_lbl = TACO_DATASET_DIR / "labels" / split
        n_img = len(list(p_img.glob("*.*"))) if p_img.exists() else 0
        n_lbl = len(list(p_lbl.glob("*.txt"))) if p_lbl.exists() else 0
        print(f"   Split '{split}': {n_img} images, {n_lbl} label files")


def main():
    parser = argparse.ArgumentParser(description="Download & merge drone litter datasets into TACO")
    parser.add_argument("--zenodo", action="store_true", help="Download Zenodo 10993447 (Marine Litter Drone Dataset)")
    parser.add_argument("--local-dir", type=Path, help="Path to local extracted YOLO dataset")
    parser.add_argument("--zip-path", type=Path, help="Path to local .zip dataset")
    parser.add_argument("--roboflow", action="store_true", help="Download from Roboflow Universe")
    parser.add_argument("--api-key", type=str, help="Roboflow API key")
    parser.add_argument("--workspace", type=str, default="marine-debris", help="Roboflow workspace")
    parser.add_argument("--project", type=str, default="marine-debris-detection", help="Roboflow project")
    parser.add_argument("--version", type=int, default=1, help="Roboflow project version")
    parser.add_argument("--prefix", type=str, default="drone_", help="Prefix for merged filenames")
    args = parser.parse_args()

    # Step 1: Handle Zenodo
    if args.zenodo:
        downloads_dir = ROOT / "artifacts" / "downloads"
        zip_path = downloads_dir / "marine_litter.zip"
        if not zip_path.exists() or zip_path.stat().st_size < 10_000_000:
            download_zenodo_dataset(zip_path)
        
        extract_dir = downloads_dir / "marine_litter_extracted"
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            print(f"[*] Extracting {zip_path} -> {extract_dir}...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
        
        merge_yolo_dataset(extract_dir, prefix="zenodo_drone_")

    # Step 2: Handle Local Directory
    elif args.local_dir:
        if not args.local_dir.exists():
            raise FileNotFoundError(f"Local directory {args.local_dir} not found")
        merge_yolo_dataset(args.local_dir, prefix=args.prefix)

    # Step 3: Handle Local Zip File
    elif args.zip_path:
        if not args.zip_path.exists():
            raise FileNotFoundError(f"Zip file {args.zip_path} not found")
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"[*] Extracting {args.zip_path}...")
            with zipfile.ZipFile(args.zip_path, "r") as z:
                z.extractall(tmp_dir)
            merge_yolo_dataset(Path(tmp_dir), prefix=args.prefix)

    # Step 4: Handle Roboflow
    elif args.roboflow:
        if not args.api_key:
            print("[ERROR] Roboflow requires --api-key. Obtain one for free at https://app.roboflow.com")
            sys.exit(1)
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=args.api_key)
            project = rf.workspace(args.workspace).project(args.project)
            dataset = project.version(args.version).download("yolov8")
            merge_yolo_dataset(Path(dataset.location), prefix="rf_drone_")
        except ImportError:
            print("[ERROR] 'roboflow' package not installed. Install via: pip install roboflow")
            sys.exit(1)
    else:
        print("Please specify a source: --zenodo, --local-dir <path>, --zip-path <path>, or --roboflow")
        return

    update_data_yaml()
    count_dataset()


if __name__ == "__main__":
    main()
