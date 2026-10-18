"""Cross-dataset evaluation: MARIDA-trained checkpoints on Plastic L2A patches.

Runs a trained semantic-segmentation checkpoint (``baseline``, ``attention``,
...) over the 20210626 Plastic L2A dataset prepared by
scripts/prepare_plastic_l2a.py and writes per-patch prediction GeoTIFFs
registered on the source CRS/transform, plus a summary JSON report.

Ground truth for Plastic L2A is still pending, so this script reports the
debris concentration statistics per patch and, if a GT mask exists (re-run
prepare with ``--gt-path``), also full segmentation metrics (mean IoU, class-1
F1) computed with the standard evaluation helpers.

Usage:
  python scripts/evaluate_cross_dataset.py --variant baseline attention
  python scripts/evaluate_cross_dataset.py --variant baseline --debris-threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rasterio

from src.data.encoding import to_train_targets
from src.data.plastic_l2a import PlasticL2aDataset
from src.evaluate.metrics import aggregate_metrics, evaluate_predictions
from src.mapping.georeference import write_concentration_tiff
from src.models.registry import VARIANTS, resolve_variant


def _build_model(variant: str, checkpoint: Path, device: torch.device):
    from src.utils.checkpoint import safe_load_state_dict
    model = resolve_variant(variant, pretrained=False)
    safe_load_state_dict(model, checkpoint, map_location="cpu")
    model.to(device).eval()
    return model, None


def _predict(model, image: np.ndarray, device: torch.device, debris_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (argmax mask, debris probability, full softmax probs)."""
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=True):
        logits = model(tensor)
    prob = torch.softmax(logits.float(), dim=1)
    mask = prob.argmax(dim=1)
    debris_prob = prob[0, 1]
    return mask[0].cpu().numpy().astype(np.int64), debris_prob.cpu().numpy(), prob[0].cpu().numpy()


def _write_mask_tiff(mask: np.ndarray, transform: tuple, crs: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
        count=1, dtype="uint8", crs=crs, transform=Affine(*transform),
    ) as dst:
        dst.write(mask.astype(np.uint8), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", nargs="+", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "artifacts" / "plastic_l2a")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "plastic_l2a_eval")
    parser.add_argument("--debris-threshold", type=float, default=0.65)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset = PlasticL2aDataset(args.data_root, allow_pending=True)
    print(f"Plastic L2A: {len(dataset)} patches, ground truth pending={not dataset.has_ground_truth}")

    results: dict[str, dict] = {}
    for variant in args.variant:
        checkpoint = ROOT / "artifacts" / variant / "best.pth"
        if not checkpoint.exists():
            print(f"SKIP {variant}: no checkpoint at {checkpoint}")
            continue
        model, ckpt_epoch = _build_model(variant, checkpoint, device)

        run_dir = args.out_dir / variant
        run_dir.mkdir(parents=True, exist_ok=True)
        per_patch = []
        metrics_list = []
        for patch_id in dataset.ids:
            image = dataset.read_image(patch_id)
            meta = dataset.read_meta(patch_id)
            mask, debris_prob, probs = _predict(model, image, device, args.debris_threshold)

            _write_mask_tiff(mask, meta.transform, meta.crs, run_dir / f"{patch_id}_pred.tif")
            write_concentration_tiff(debris_prob, meta, run_dir / f"{patch_id}_concentration.tif")

            debris_frac = float(np.count_nonzero(mask == 1) / mask.size)
            entry = {
                "patch_id": patch_id,
                "debris_pixel_fraction": round(debris_frac, 6),
                "predicted_class_counts": {
                    str(c): int(np.count_nonzero(mask == c)) for c in np.unique(mask)
                },
            }
            if dataset.has_ground_truth:
                gt11 = dataset.read_mask(patch_id)
                tgt_np = to_train_targets(gt11)
                n_labeled = int(np.count_nonzero(tgt_np != 11))
                entry["gt_labeled_pixels"] = n_labeled
                if n_labeled > 0:
                    tgt = torch.from_numpy(tgt_np).unsqueeze(0).to(device)
                    pred_logits = torch.from_numpy(
                        np.log(np.clip(probs, 1e-7, 1.0))
                    ).unsqueeze(0).to(device)
                    metrics_list.append(evaluate_predictions(pred_logits, tgt))
            per_patch.append(entry)

        report = {
            "dataset": "plastic_l2a",
            "campaign": "20210626",
            "variant": variant,
            "checkpoint_epoch": ckpt_epoch,
            "debris_threshold": args.debris_threshold,
            "n_patches": len(dataset),
            "ground_truth_available": bool(metrics_list and dataset.has_ground_truth),
            "per_patch": per_patch,
            "mean_debris_pixel_fraction": round(
                float(np.mean([p["debris_pixel_fraction"] for p in per_patch])), 6
            ),
        }
        if metrics_list and dataset.has_ground_truth:
            summary = aggregate_metrics(metrics_list)
            report.update({
                "mean_iou": summary["mean_iou"],
                "f1_class_1_marine_debris": summary["f1_class_1_marine_debris"],
                "accuracy": summary["accuracy"],
                "iou_per_class": summary["iou_per_class"],
            })
        else:
            report["message"] = (
                "Ground truth pending: only prediction statistics are reported. "
                "Re-run scripts/prepare_plastic_l2a.py --gt-path <raster> to score IoU/F1."
            )

        out = run_dir / "cross_dataset_metrics.json"
        out.write_text(json.dumps(report, indent=2))
        results[variant] = report
        print(f"{variant}: {len(dataset)} patches -> {run_dir}")
        print(f"  mean debris fraction: {report['mean_debris_pixel_fraction']}")
        if report["ground_truth_available"]:
            print(f"  mean_iou={report['mean_iou']}, f1_debris={report['f1_class_1_marine_debris']}")

    if results:
        print("\nDone. Per-patch predictions + concentration tiffs in", args.out_dir)


if __name__ == "__main__":
    main()