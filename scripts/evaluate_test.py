"""Evaluate trained model checkpoints on the official test split.

Usage:
  python scripts/evaluate_test.py --variant baseline attention
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.factory import build_loaders
from src.data.ingestion import MaridaDataset
from src.data.splits import assert_scene_disjoint, load_official_splits
from src.evaluate.metrics import aggregate_metrics, evaluate_predictions
from src.models.registry import VARIANTS, resolve_variant


def _build_model(variant: str, checkpoint: Path, device: torch.device):
    from src.utils.checkpoint import safe_load_state_dict
    model = resolve_variant(variant, pretrained=False)
    safe_load_state_dict(model, checkpoint, map_location="cpu")
    model.to(device).eval()
    return model, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", nargs="+", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    assert_scene_disjoint(splits, marida)
    test_loader = build_loaders(marida, splits, batch_size=args.batch_size)["test"]

    results = {}
    for variant in args.variant:
        run_dir = ROOT / "artifacts" / variant
        checkpoint = run_dir / "best.pth"
        model, ckpt_epoch = _build_model(variant, checkpoint, device)
        metrics_list = []
        with torch.no_grad():
            for image, mask in test_loader:
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=True):
                    logits = model(image)
                metrics_list.append(evaluate_predictions(logits.float(), mask))
        summary = aggregate_metrics(metrics_list)
        summary["checkpoint_epoch"] = ckpt_epoch
        output = {
            "mean_iou": summary["mean_iou"],
            "accuracy": summary["accuracy"],
            "f1_class_1_marine_debris": summary["f1_class_1_marine_debris"],
            "iou_per_class": summary["iou_per_class"],
            "f1_per_class": summary["f1_per_class"],
            "checkpoint_epoch": ckpt_epoch,
        }
        (run_dir / "test_metrics.json").write_text(json.dumps(output, indent=2))
        results[variant] = output
        print(f"{variant}: test mean_iou={output['mean_iou']}, f1_debris={output['f1_class_1_marine_debris']}, accuracy={output['accuracy']}")

    if results:
        best = max(results.items(), key=lambda kv: kv[1]["mean_iou"])
        print(f"\nBest model by test mean_iou: {best[0]} ({best[1]['mean_iou']:.4f})")
        if "baseline" in results:
            delta = results[best[0]]["mean_iou"] - results["baseline"]["mean_iou"]
            print(f"{best[0]} vs baseline: {delta:+.4f} test mean_iou")


if __name__ == "__main__":
    main()