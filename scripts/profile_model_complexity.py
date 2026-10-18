#!/usr/bin/env python
"""Model Complexity Profiler for TECH-ATHON Marine Debris Detection.

Profiles all registered hybrid architectures and computes:
- Total parameter count (M)
- FLOPs / MACs on a dummy input (1, 11, 256, 256)
- Parameter-to-Training-Sample Ratio (params / n_train_samples)
- Peak CUDA memory usage on forward + backward pass

Saves summary to artifacts/reports/model_complexity_report.json
and prints a clean formatted CLI table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.splits import load_official_splits
from src.models.registry import VARIANTS, resolve_variant

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available, FLOPs will be estimated from parameters")


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "m_params": round(total / 1e6, 2)}


def estimate_flops(model: nn.Module, input_shape: tuple = (1, 11, 256, 256)) -> float:
    """Estimate FLOPs using thop if available, else parameter-based heuristic."""
    if THOP_AVAILABLE:
        dummy = torch.randn(*input_shape)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        # FLOPs = 2 * MACs (multiply + add)
        return round(2 * macs / 1e9, 2)  # GFLOPs
    # Heuristic: ~2 FLOPs per parameter per forward pass for typical CNNs
    params = count_parameters(model)["total"]
    return round(2 * params / 1e9, 2)


def measure_peak_memory(model: nn.Module, input_shape: tuple = (1, 11, 256, 256)) -> float:
    """Measure peak CUDA memory for forward + backward pass (in MB)."""
    if not torch.cuda.is_available():
        return 0.0

    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(*input_shape, device=device, requires_grad=True)

    try:
        out = model(dummy)
        loss = out.sum()
        loss.backward()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return round(peak_bytes / (1024 ** 2), 1)  # MB
    except Exception as e:
        print(f"  Memory measurement failed: {e}")
        return 0.0
    finally:
        model.cpu()
        torch.cuda.empty_cache()


def get_training_sample_count() -> int:
    """Get number of training samples from official splits."""
    splits = load_official_splits(ROOT / "splits")
    return len(splits["train"])


def profile_variant(variant_name: str, pretrained: bool = False) -> dict[str, Any]:
    """Profile a single variant."""
    print(f"  Profiling {variant_name}...", end=" ", flush=True)
    try:
        model = resolve_variant(variant_name, pretrained=pretrained)
        model.eval()

        params = count_parameters(model)
        flops = estimate_flops(model)
        n_train = get_training_sample_count()
        params_per_sample = round(params["total"] / n_train, 1)
        peak_mem = measure_peak_memory(model)

        # Overfit risk classification
        if params_per_sample > 5000:
            risk = "HIGH"
        elif params_per_sample > 2000:
            risk = "MEDIUM"
        else:
            risk = "LOW / BALANCED"

        result = {
            "variant": variant_name,
            "params_m": params["m_params"],
            "flops_g": flops,
            "params_per_sample": params_per_sample,
            "peak_memory_mb": peak_mem,
            "overfit_risk": risk,
        }
        print(f"done ({params['m_params']}M params, {params_per_sample:.0f} params/sample, {risk})")
        return result
    except Exception as e:
        print(f"FAILED: {e}")
        return {
            "variant": variant_name,
            "params_m": 0.0,
            "flops_g": 0.0,
            "params_per_sample": 0.0,
            "peak_memory_mb": 0.0,
            "overfit_risk": "ERROR",
            "error": str(e),
        }


def main() -> None:
    print("=" * 80)
    print("MODEL COMPLEXITY PROFILER")
    print("=" * 80)
    print(f"Training samples: {get_training_sample_count()}")
    print()

    results = []
    for variant_name in sorted(VARIANTS.keys()):
        result = profile_variant(variant_name)
        results.append(result)

    # Sort by params_per_sample for the table
    results.sort(key=lambda x: x["params_per_sample"])

    # Print CLI table
    print()
    print("=" * 100)
    print(f"{'Variant':<25} {'Params (M)':>10} {'FLOPs (G)':>10} {'Params/Sample':>14} {'Peak Mem (MB)':>14} {'Overfit Risk':>14}")
    print("-" * 100)
    for r in results:
        print(f"{r['variant']:<25} {r['params_m']:>10.2f} {r['flops_g']:>10.1f} "
              f"{r['params_per_sample']:>14.0f} {r['peak_memory_mb']:>14.1f} {r['overfit_risk']:>14}")
    print("=" * 100)

    # Save JSON report
    out_dir = ROOT / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_complexity_report.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()