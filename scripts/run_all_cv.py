"""Run 10-fold scene-disjoint cross-validation across all architecture variants.

Loops every registered variant (or a requested subset), reuses run_cv() from
run_cross_validation.py, and writes one comparison summary:
  artifacts/cv/_all/cv_compare.json

A markdown comparison table is printed to stdout.

Usage:
  python scripts/run_all_cv.py                      # all variants, 20 epochs, 10 folds
  python scripts/run_all_cv.py --smoke              # 1 epoch, 2 folds, batch 4 (sanity check)
  python scripts/run_all_cv.py --variants baseline attention --epochs 20
  python scripts/run_all_cv.py --epochs 20 --folds 10 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.registry import VARIANTS
from scripts.run_cross_validation import METRIC_STEMS, run_cv

COMPARE_DIR = ROOT / "artifacts" / "cv" / "_all"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="*", choices=sorted(VARIANTS),
                   help="subset to run; defaults to ALL registered variants")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 2 folds, batch 4 for a quick sanity check")
    return p.parse_args()


def format_cell(summary: dict, stem: str) -> str:
    mean = summary[f"mean_val_{stem}"]
    if mean != mean:  # NaN
        return "N/A"
    std = summary[f"std_val_{stem}"]
    n_nan = summary[f"n_nan_val_{stem}"]
    cell = f"{mean:.4f} ± {std:.4f}" if std == std else f"{mean:.4f}"
    if n_nan:
        cell += f" ({n_nan} nan)"
    return cell


def build_comparison(variant_summaries: dict[str, dict]) -> dict:
    stems = list(METRIC_STEMS.values())
    table = [
        ["variant", *[f"val {stem}" for stem in stems]],
    ]
    for variant, summary in variant_summaries.items():
        table.append([variant, *[format_cell(summary, stem) for stem in stems]])
    return {
        "stems": stems,
        "variants": variant_summaries,
        "table": table,
    }


def print_table(table: list[list[str]]) -> None:
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n{'=' * 90}")
    print("CROSS-VALIDATION COMPARISON (mean val score ± std over folds)")
    print(fmt.format(*table[0]))
    print("-" * 90)
    for row in table[1:]:
        print(fmt.format(*row))
    print("=" * 90)


def main() -> None:
    args = parse_args()
    epochs = 1 if args.smoke else args.epochs
    folds = 2 if args.smoke else args.folds
    batch_size = 4 if args.smoke else args.batch_size

    variants = args.variants if args.variants else sorted(VARIANTS)
    print(
        f"Running {len(variants)} variant(s): {', '.join(variants)}\n"
        f"epochs={epochs}, folds={folds}, batch_size={batch_size}"
    )

    variant_summaries: dict[str, dict] = {}
    for variant in variants:
        summary = run_cv(
            variant,
            epochs=epochs,
            batch_size=batch_size,
            folds=folds,
        )
        variant_summaries[variant] = summary

    compare = build_comparison(variant_summaries)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    compare_path = COMPARE_DIR / "cv_compare.json"
    compare_path.write_text(json.dumps(compare, indent=2))
    print(f"\nComparison saved to {compare_path}")
    print_table(compare["table"])


if __name__ == "__main__":
    main()