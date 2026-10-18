"""Prepare the YOLO detection dataset from MARIDA masks.

Exports per-patch RGB composites (B4,B3,B2) and YOLO txt labels derived from
debris bounding boxes, respecting the official scene-disjoint splits. Also
builds a class-balanced train split (all positives + matched negatives) so the
detector sees debris every batch.

Usage:
  python scripts/prepare_yolo.py --min-area 6
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ingestion import MaridaDataset
from src.data.splits import load_official_splits
from src.detection.labels import SIZE, debris_boxes, rgb_composite, to_yolo_lines

OUT_DIR = ROOT / "artifacts" / "detection"
CLASS_NAMES = ["marine_debris"]


def _write_patch(marida, pid, img_dir, lbl_dir, min_area: int) -> int:
    image = marida.read_image(pid)
    mask = marida.read_mask(pid)
    rgb = rgb_composite(image)
    name = pid.replace("/", "_")
    (img_dir / f"{name}.png").write_bytes(_png_from_array(rgb))
    boxes = debris_boxes(mask, min_area=min_area)
    (lbl_dir / f"{name}.txt").write_text("\n".join(to_yolo_lines(boxes)))
    return 1 if boxes else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-area", type=int, default=6, help="ignore debris blobs smaller than this many pixels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")

    pos_by_split: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        img_dir = OUT_DIR / "images" / split
        lbl_dir = OUT_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        positives = []
        for pid in splits[split]:
            if _write_patch(marida, pid, img_dir, lbl_dir, args.min_area):
                positives.append(pid)
        pos_by_split[split] = positives
        print(f"{split}: {len(splits[split])} patches, {len(positives)} with debris >= {args.min_area}px")

    # Balanced train: all positives + an equal sample of negatives.
    positives = pos_by_split["train"]
    train_set = set(splits["train"])
    negatives = [p for p in splits["train"] if p not in set(positives)]
    random.shuffle(negatives)
    balanced = positives + negatives[: len(positives)]
    random.shuffle(balanced)
    bal_img, bal_lbl = OUT_DIR / "images" / "train_balanced", OUT_DIR / "labels" / "train_balanced"
    bal_img.mkdir(parents=True, exist_ok=True)
    bal_lbl.mkdir(parents=True, exist_ok=True)
    for pid in balanced:
        _write_patch(marida, pid, bal_img, bal_lbl, args.min_area)
    print(f"train_balanced: {len(balanced)} patches ({len(positives)} debris + {len(balanced) - len(positives)} negatives)")

    def _dump_yaml(name: str, train_subdir: str) -> None:
        data_yaml = {
            "path": str(OUT_DIR),
            "train": train_subdir,
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": CLASS_NAMES,
        }
        (OUT_DIR / name).write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    _dump_yaml("data.yaml", "images/train")
    _dump_yaml("data_balanced.yaml", "images/train_balanced")
    print(f"wrote {OUT_DIR / 'data.yaml'} and {OUT_DIR / 'data_balanced.yaml'}")


def _png_from_array(rgb):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    main()