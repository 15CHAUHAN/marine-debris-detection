#!/usr/bin/env python
"""Convergence comparison benchmark for weight initialization strategies.

Compares:
- imagenet_replicate: Legacy naive RGB replication across all 11 channels
- imagenet_physics: Physics-informed spectral mapping (default)

Measures:
- Initial epoch loss and validation IoU
- Gradient norm stabilization across non-RGB channels (B08, B11, B12)
- Convergence rate over 3 epochs

Outputs comparison plot to artifacts/reports/weight_init_comparison.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.augment import TrainTransform
from src.data.factory import build_loaders
from src.data.ingestion import MaridaDataset
from src.data.splits import assert_scene_disjoint, load_official_splits
from src.models.registry import resolve_variant
from src.train.trainer import TrainConfig, Trainer


def run_benchmark(
    strategy: str,
    variant: str = "baseline",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: torch.device | None = None,
) -> dict:
    """Run a short training benchmark for a weight initialization strategy."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Benchmarking {strategy} on {variant} (device={device})")
    print(f"{'='*60}")

    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    assert_scene_disjoint(splits, marida)

    # Small subset for fast benchmarking
    train_ids = splits["train"][:32]
    val_ids = splits["val"][:16]

    config = TrainConfig(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        run_name=f"{variant}_{strategy}_bench",
    )

    model = resolve_variant(variant, weight_init_strategy=strategy)
    model.to(device)

    loaders = build_loaders(
        marida,
        {"train": train_ids, "val": val_ids, "test": []},
        batch_size=batch_size,
        train_transform=TrainTransform(),
    )

    # Track gradient norms for non-RGB channels
    grad_norms = {"epoch": [], "B08_NIR": [], "B11_SWIR1": [], "B12_SWIR2": []}

    def grad_hook(name):
        def hook(grad):
            if grad is not None:
                norm = grad.norm().item()
                grad_norms[name].append(norm)
        return hook

    # Register hooks on first conv weights for non-RGB channels
    first_conv = None
    if hasattr(model, "encoder") and hasattr(model.encoder, "conv1"):
        first_conv = model.encoder.conv1
    elif hasattr(model, "unet") and hasattr(model.unet.encoder, "conv1"):
        first_conv = model.unet.encoder.conv1
    elif hasattr(model, "unet") and hasattr(model.unet.encoder, "_conv_stem"):
        first_conv = model.unet.encoder._conv_stem

    if first_conv is not None:
        # Sentinel-2 non-RGB indices: 7=B08, 9=B11, 10=B12
        hooks = [
            first_conv.weight.register_hook(grad_hook("B08_NIR")),
            first_conv.weight.register_hook(lambda g: grad_norms["B11_SWIR1"].append(g.norm().item()) if g is not None else None),
            first_conv.weight.register_hook(lambda g: grad_norms["B12_SWIR2"].append(g.norm().item()) if g is not None else None),
        ]
    else:
        hooks = []

    trainer = Trainer(model, config, device)
    result = trainer.run(loaders["train"], loaders["val"])

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute average gradient norms per epoch for non-RGB channels
    avg_grad_norms = {}
    for name in ("B08_NIR", "B11_SWIR1", "B12_SWIR2"):
        norms = grad_norms[name]
        if norms:
            # Average per epoch (approximately len(train)/batch_size steps per epoch)
            steps_per_epoch = len(loaders["train"])
            epoch_avgs = []
            for i in range(epochs):
                start = i * steps_per_epoch
                end = min((i + 1) * steps_per_epoch, len(norms))
                if start < end:
                    epoch_avgs.append(np.mean(norms[start:end]))
            avg_grad_norms[name] = epoch_avgs
        else:
            avg_grad_norms[name] = []

    # Read history from CSV
    val_miou_history = []
    train_loss_history = []
    history_csv = Path(config.run_dir) / config.run_name / "history.csv"
    if history_csv.exists():
        import csv
        with history_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_loss_history.append(float(row["train_loss"]))
                val_miou_history.append(float(row.get("val_mean_iou", row.get("val_miou", 0.0))))

    return {
        "strategy": strategy,
        "variant": variant,
        "best_val_miou": result["best_mean_iou"],
        "best_epoch": result["best_epoch"],
        "final_train_loss": train_loss_history[-1] if train_loss_history else 0.0,
        "val_miou_history": val_miou_history,
        "train_loss_history": train_loss_history,
        "grad_norms": avg_grad_norms,
    }


def main():
    strategies = ["imagenet_replicate", "imagenet_physics"]
    variant = "baseline"
    epochs = 3
    batch_size = 4
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on {device}")

    results = {}
    for strategy in strategies:
        results[strategy] = run_benchmark(
            strategy=strategy,
            variant=variant,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    for strategy, res in results.items():
        print(f"\n{strategy}:")
        print(f"  Best val mIoU: {res['best_val_miou']:.4f} (epoch {res['best_epoch']})")
        print(f"  Final train loss: {res['final_train_loss']:.4f}")
        print(f"  Val mIoU history: {[round(v, 4) for v in res['val_miou_history']]}")
        print(f"  Train loss history: {[round(v, 4) for v in res['train_loss_history']]}")
        for ch, norms in res['grad_norms'].items():
            if norms:
                print(f"  {ch} avg grad norm/epoch: {[round(n, 6) for n in norms]}")

    # Save JSON results
    out_dir = ROOT / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "weight_init_benchmark.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_json}")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Weight Initialization Comparison: {variant} (3 epochs)", fontsize=14)

    # Val mIoU
    ax = axes[0, 0]
    for strategy, res in results.items():
        epochs_range = range(1, len(res["val_miou_history"]) + 1)
        ax.plot(epochs_range, res["val_miou_history"], marker="o", label=strategy)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mIoU")
    ax.set_title("Validation mIoU Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Train Loss
    ax = axes[0, 1]
    for strategy, res in results.items():
        epochs_range = range(1, len(res["train_loss_history"]) + 1)
        ax.plot(epochs_range, res["train_loss_history"], marker="o", label=strategy)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient Norms - B08 NIR
    ax = axes[1, 0]
    for strategy, res in results.items():
        norms = res["grad_norms"].get("B08_NIR", [])
        if norms:
            ax.plot(range(1, len(norms) + 1), norms, marker="o", label=strategy)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Gradient Norm")
    ax.set_title("B08 (NIR) Channel Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient Norms - B11 SWIR1
    ax = axes[1, 1]
    for strategy, res in results.items():
        norms = res["grad_norms"].get("B11_SWIR1", [])
        if norms:
            ax.plot(range(1, len(norms) + 1), norms, marker="o", label=strategy)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Gradient Norm")
    ax.set_title("B11 (SWIR1) Channel Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = out_dir / "weight_init_comparison.png"
    plt.savefig(out_png, dpi=150)
    print(f"Plot saved to {out_png}")

    # Quick comparison summary
    print("\n" + "="*70)
    print("CONVERGENCE COMPARISON")
    print("="*70)
    rep_miou = results["imagenet_replicate"]["best_val_miou"]
    phys_miou = results["imagenet_physics"]["best_val_miou"]
    print(f"imagenet_replicate best val mIoU: {rep_miou:.4f}")
    print(f"imagenet_physics best val mIoU:   {phys_miou:.4f}")
    if phys_miou > rep_miou:
        print(f"-> Physics-aware wins by +{(phys_miou - rep_miou)*100:.2f}%")
    else:
        print(f"-> Replicate wins by +{(rep_miou - phys_miou)*100:.2f}%")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()