"""Count exact number of images/patches classified as 1 (Plastic) vs 0 (No Plastic).

Usage:
    python scripts/count_dataset_classes.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    mapping_path = ROOT / "labels_mapping.txt"
    if not mapping_path.exists():
        print(f"Error: {mapping_path} not found.")
        return

    with open(mapping_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # In labels_mapping, index 0 is Class 1 (Marine Debris / Plastic)
    # 1 = Plastic present, 0 = Plastic absent
    total_all = len(labels)
    yes_all = sum(1 for vec in labels.values() if vec[0] == 1)
    no_all = total_all - yes_all

    print("=" * 65)
    print("      DATASET BINARY SEGREGATION: CLASS 1 (PLASTIC / DEBRIS)     ")
    print("=" * 65)
    print(f"Total Patches in Dataset  : {total_all}")
    print(f"  -> Label 1 (Plastic YES): {yes_all} patches ({yes_all / total_all * 100:.2f}%)")
    print(f"  -> Label 0 (Plastic NO) : {no_all} patches ({no_all / total_all * 100:.2f}%)")
    print("-" * 65)
    print("SPLIT BREAKDOWN (Scene-Disjoint Train / Val / Test):")
    print("-" * 65)

    splits_dir = ROOT / "splits"
    for split_name in ("train", "val", "test"):
        split_file = splits_dir / f"{split_name}_X.txt"
        if not split_file.exists():
            continue
        patch_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        total_split = len(patch_ids)
        yes_split = sum(1 for pid in patch_ids if labels.get(f"S2_{pid}.tif", [0])[0] == 1)
        no_split = total_split - yes_split
        pct_yes = (yes_split / total_split * 100) if total_split else 0
        pct_no = (no_split / total_split * 100) if total_split else 0

        print(f"Split: {split_name.upper():<5} | Total: {total_split:<4} | Label 1 (YES): {yes_split:<4} ({pct_yes:5.1f}%) | Label 0 (NO): {no_split:<4} ({pct_no:5.1f}%)")

    print("=" * 65)

if __name__ == "__main__":
    main()
