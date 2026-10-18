"""Plot training loss curves and val mean IoU for trained variants.

Usage:
  python scripts/loss_curves.py baseline attention
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def _load_history(run_dir: Path) -> dict[str, list]:
    rows = []
    with (run_dir / "history.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "train_loss": [float(r["train_loss"]) for r in rows],
        "val_mean_iou": [float(r["val_mean_iou"]) for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="+", choices=["baseline", "attention"])
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for variant in args.variant:
        hist = _load_history(ROOT / "artifacts" / variant)
        axes[0].plot(hist["epoch"], hist["train_loss"], marker="o", ms=3, label=variant)
        axes[1].plot(hist["epoch"], hist["val_mean_iou"], marker="o", ms=3, label=variant)
    axes[0].set_title("train loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].set_title("val mean IoU")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    out_dir = ROOT / "artifacts" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loss_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()