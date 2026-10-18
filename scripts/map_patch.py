"""Generate a georeferenced Marine Debris concentration GeoTiff for a patch.

Loads a trained model checkpoint (baseline or attention), runs inference on
one patch (by default the first official test-split patch that contains
Marine Debris), builds the debris concentration heatmap, and writes it as a
float32 GeoTiff with the source patch's CRS and transform.

Usage:
  python scripts/map_patch.py --variant attention --patch-id <id>
  python scripts/map_patch.py --variant baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.encoding import to_train_targets
from src.data.ingestion import MaridaDataset
from src.data.splits import load_official_splits
from src.evaluate.metrics import confusion_matrix
from src.mapping.georeference import write_concentration_tiff
from src.mapping.heatmap import debris_concentration
from src.models.registry import resolve_variant

DEBRIS_CHANNEL = 0


def _build_model(variant: str, checkpoint: Path, device: torch.device):
    from src.utils.checkpoint import safe_load_state_dict
    model = resolve_variant(variant, pretrained=False)
    safe_load_state_dict(model, checkpoint, map_location="cpu")
    model.to(device).eval()
    return model, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "attention"], required=True)
    parser.add_argument("--patch-id", default=None, help="visit patch id; defaults to the first test patch labeled Marine Debris")
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")

    patch_id = args.patch_id
    if patch_id is None:
        candidates = [p for p in splits["test"] if (marida.read_mask(p) == 1).sum() > 0]
        patch_id = candidates[0]
        print(f"auto-selected test patch {patch_id!r} (Marine Debris present)")
    patch = marida.get(patch_id)

    model, ckpt_epoch = _build_model(args.variant, ROOT / "artifacts" / args.variant / "best.pth", device)
    print(f"checkpoint epoch={ckpt_epoch}")

    image = marida.read_image(patch_id)
    meta = marida.read_meta(patch_id)
    truth = marida.read_mask(patch_id)

    with torch.no_grad():
        batch = torch.from_numpy(image[None]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=True):
            logits = model(batch)
    pred_label = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)
    debris_mask = (pred_label == DEBRIS_CHANNEL).astype(np.float32)

    cm = confusion_matrix(logits.float(), torch.from_numpy(to_train_targets(truth))[None], num_classes=11)
    debris_tp = int(cm[DEBRIS_CHANNEL, DEBRIS_CHANNEL])
    debris_total = int(cm[DEBRIS_CHANNEL].sum())
    print(f"debris: predicted pixels={int(debris_mask.sum())}, "
          f"TP={debris_tp}, true pixels={debris_total}")

    concentration = debris_concentration(debris_mask, sigma=args.sigma)
    out_dir = ROOT / "artifacts" / "maps"
    out_path = out_dir / f"{patch_id}_{args.variant}_debris_concentration.tif"
    write_concentration_tiff(concentration, meta, out_path)
    print(f"wrote {out_path} (crs={meta.crs})")

    rgb = image[[2, 1, 0]].transpose(1, 2, 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.clip(rgb / max(rgb.max(), 1e-6), 0, 1))
    axes[0].set_title(f"{patch_id} RGB")
    axes[0].axis("off")
    axes[1].imshow(concentration, cmap="hot", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{args.variant} debris concentration (sigma={args.sigma})")
    axes[1].axis("off")
    fig.tight_layout()
    png_path = out_dir / f"{patch_id}_{args.variant}_concentration.png"
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()