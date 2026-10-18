"""Download and convert the TACO dataset to YOLOv8 format.

Maps TACO's 60 fine-grained litter categories into the 7 practical macro
classes used by the dashboard YOLO detector (bottles, cans, bags, cups,
wrappers, containers, other), converts the COCO annotations into normalized
YOLO txt labels, and writes a ``data.yaml`` consumed by ``train_taco.py``.

The raw TACO release keeps images under ``data/batch_<n>/``; their filenames
are recorded in ``annotations.json``. Images are hosted on Flickr, so they
must exist locally before conversion (either downloaded manually via TACO's
own ``download.py``, with ``--download-images`` below, or unzipped from the
Zenodo release).

Usage:
    python scripts/prepare_taco.py [--download] [--download-images]
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
TACO_DIR = ROOT / "artifacts" / "taco_dataset"

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)

# 7 standard macro-classes mapped from TACO's 60 subcategories
TARGET_CLASSES = [
    "Bottle",          # 0: Plastic, glass, metal bottles
    "Can",             # 1: Drink cans, food cans
    "Plastic bag",     # 2: Grocery bags, trash bags, films
    "Cup",             # 3: Plastic cups, paper coffee cups
    "Wrapper_Sachet",  # 4: Crisp packets, sweet wrappers, sachets
    "Container",       # 5: Food trays, takeaway boxes, bidons
    "Other_Trash",     # 6: Bottle caps, straws, broken fragments
]

CATEGORY_MAPPING = {
    # Bottles
    "Plastic bottle": 0, "Clear plastic bottle": 0, "Glass bottle": 0, "Other plastic bottle": 0,
    # Cans
    "Drink can": 1, "Food can": 1, "Aerosol": 1,
    # Bags
    "Plastic bag": 2, "Polypropylene bag": 2, "Garbage bag": 2,
    "Plastic film": 2, "Single-use carrier bag": 2,
    "Paper bag": 2, "Plastified paper bag": 2,
    # Cups
    "Disposable plastic cup": 3, "Paper cup": 3, "Coffee cup": 3,
    "Foam cup": 3, "Glass cup": 3, "Other plastic cup": 3,
    # Wrappers
    "Crisp packet": 4, "Other plastic wrapper": 4, "Food wrapper": 4,
    # Containers
    "Plastic container": 5, "Foam container": 5, "Takeaway container": 5,
    "Disposable food container": 5, "Foam food container": 5, "Other plastic container": 5,
    "Tupperware": 5, "Spread tub": 5, "Meal carton": 5, "Drink carton": 5,
    "Other carton": 5, "Pizza box": 5, "Corrugated carton": 5, "Egg carton": 5,
}


def _relative_search_roots(raw_dir: Path) -> list[Path]:
    """Candidate locations for the annotations file under a raw TACO folder."""
    return [raw_dir / "data", raw_dir, raw_dir / "TACO"]


def find_annotations(raw_dir: Path) -> Path | None:
    """Locate ``annotations.json`` (repo, unzipped, or flat layout).

    TACO's repository keeps it at ``data/annotations.json`` while manually
    placed copies may sit directly in the raw folder.
    """
    for base in _relative_search_roots(raw_dir):
        candidate = base / "annotations.json"
        if candidate.exists():
            return candidate
    return None


def _download(url: str, target: Path, timeout: float = 30.0) -> None:
    """Fetch ``url`` into ``target`` with a per-request timeout."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        target.write_bytes(response.read())


def download_taco_subset(out_dir: Path, download_images: bool = False) -> Path:
    """Download TACO annotations (and optionally images) into ``out_dir``.

    Downloads only ``data/annotations.json`` from the TACO repository unless
    ``download_images`` is set, in which case every image referenced by the
    annotations is fetched from its Flickr ``flickr_640_url`` (VGA-resized).
    Existing files are skipped so re-runs resume where they left off.

    Returns the path to the local ``annotations.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "annotations.json"
    if not json_path.exists():
        print("Downloading TACO annotations...")
        _download(ANNOTATIONS_URL, json_path)
    else:
        print("annotations.json is already present.")

    if download_images:
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        imgs = coco.get("images", [])
        present = downloaded = failed = 0
        for i, img in enumerate(imgs, 1):
            file_name = img["file_name"]
            target = out_dir / file_name
            if target.exists():
                present += 1
            else:
                url = img.get("flickr_640_url") or img.get("flickr_url")
                if not url:
                    failed += 1
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        _download(url, target)
                        downloaded += 1
                    except Exception as exc:  # noqa: BLE001 - keep going on flaky hosts
                        failed += 1
                        if failed <= 20:
                            print(f"  failed {file_name}: {exc}")
            if i % 50 == 0:
                print(
                    f"  progress {i}/{len(imgs)} "
                    f"(present={present}, downloaded={downloaded}, failed={failed})",
                    flush=True,
                )
        print(f"images: {present} present, {downloaded} downloaded, {failed} failed of {len(imgs)}")
    return json_path


def convert_coco_to_yolo(coco_json_path: Path, images_root: Path, output_dir: Path):
    """Convert COCO annotations to YOLO format (images/ + labels/ + data.yaml)."""
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build category ID to target class ID mapping
    cat_id_to_target = {}
    for cat in coco["categories"]:
        name = cat["name"]
        cat_id = cat["id"]
        target = CATEGORY_MAPPING.get(name, 6)  # default to 'Other_Trash'
        cat_id_to_target[cat_id] = target

    # Map image_id to image info
    images_dict = {img["id"]: img for img in coco["images"]}

    # TACO numbers files per batch (batch_1/000006.jpg, batch_2/000006.jpg, ...),
    # so basenames are NOT unique. Prefix with the folder to avoid collisions.
    basename_counts: dict[str, int] = {}
    for img in coco["images"]:
        base = Path(img["file_name"]).name
        basename_counts[base] = basename_counts.get(base, 0) + 1

    # Collect annotations per image
    img_anns = {img_id: [] for img_id in images_dict}
    counted: dict[int, int] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if cat_id not in cat_id_to_target:
            continue
        bbox = ann["bbox"]  # [x_min, y_min, width, height]
        img_info = images_dict[img_id]
        img_w, img_h = img_info["width"], img_info["height"]

        # Normalize to YOLO format [x_center, y_center, w, h]
        x_center = (bbox[0] + bbox[2] / 2.0) / img_w
        y_center = (bbox[1] + bbox[3] / 2.0) / img_h
        w = bbox[2] / img_w
        h = bbox[3] / img_h

        target_cls = cat_id_to_target[cat_id]
        img_anns[img_id].append((target_cls, x_center, y_center, w, h))
        counted[target_cls] = counted.get(target_cls, 0) + 1

    # Split into Train (80%), Val (10%), Test (10%)
    valid_img_ids = [iid for iid, anns in img_anns.items() if len(anns) > 0]
    train_ids, test_val_ids = train_test_split(valid_img_ids, test_size=0.20, random_state=42)
    val_ids, test_ids = train_test_split(test_val_ids, test_size=0.50, random_state=42)

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    missing = 0
    for split_name, ids in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for iid in ids:
            info = images_dict[iid]
            src_file = images_root / info["file_name"]
            if not src_file.exists():
                # Safe flat fallback only when the basename is unambiguous.
                base_name = Path(info["file_name"]).name
                flat = images_root / base_name
                if flat.exists() and basename_counts.get(base_name, 0) == 1:
                    src_file = flat
            if src_file.exists():
                # Unique output name derived from the full relative path.
                out_name = info["file_name"].replace("/", "_").replace("\\", "_")
                shutil.copy2(src_file, img_out / out_name)
                # Write YOLO label file
                lbl_file = lbl_out / f"{Path(out_name).stem}.txt"
                with open(lbl_file, "w") as lf:
                    for cls_id, xc, yc, bw, bh in img_anns[iid]:
                        lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                missing += 1

    # Generate data.yaml
    yaml_content = f"""path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(TARGET_CLASSES)}

names:
"""
    for idx, name in enumerate(TARGET_CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    (output_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    per_class = ", ".join(
        f"{TARGET_CLASSES[i]}={counted.get(i, 0)}" for i in range(len(TARGET_CLASSES))
    )
    print(f"annotations per class: {per_class}")
    print(f"split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    if missing:
        print(f"WARNING: {missing} images not found under {images_root}; "
              f"run with --download-images or fetch the Zenodo release.")
    print(f"Dataset successfully prepared at {output_dir}")
    print(f"data.yaml written to {output_dir / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "taco_raw"), help="Path to raw TACO folder")
    parser.add_argument("--out-dir", default=str(TACO_DIR), help="Path for YOLO formatted output")
    parser.add_argument("--download", action="store_true", default=True,
                        help="fetch annotations.json if missing (default: on)")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="do not fetch annotations.json; fail if it is absent")
    parser.add_argument("--download-images", action="store_true",
                        help="also fetch the VGA-resized images from Flickr")
    args = parser.parse_args()

    raw_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    json_path = find_annotations(raw_dir)
    if json_path is None:
        if not args.download:
            raise SystemExit(f"annotations.json not found in {raw_dir}")
        print(f"annotations.json not found in {raw_dir}; downloading...")
        json_path = download_taco_subset(raw_dir, download_images=args.download_images)
    elif args.download_images:
        json_path = download_taco_subset(raw_dir, download_images=True)

    base = json_path.parent
    images_root = base if (base / "annotations.json").exists() else raw_dir
    if not images_root.exists():
        images_root = raw_dir

    convert_coco_to_yolo(json_path, images_root, out_dir)


if __name__ == "__main__":
    main()