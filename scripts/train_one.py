"""Train one model variant (baseline or attention) on the official split.

Usage:
  python scripts/train_one.py --variant baseline --epochs 20 --batch-size 8
  python scripts/train_one.py --variant attention --epochs 20 --batch-size 8
  python scripts/train_one.py --variant baseline --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.augment import TrainTransform
from src.data.encoding import N_CLASSES
from src.data.factory import build_loaders
from src.data.ingestion import MaridaDataset
from src.data.splits import assert_scene_disjoint, load_official_splits
from src.models.registry import VARIANTS, resolve_variant
from src.train.trainer import TrainConfig, Trainer


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (streamed, any size)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_train_manifest(
    variant: str,
    out_dir: Path,
    splits_dir: Path,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    *,
    source: str = "train_one",
) -> dict:
    """Enforce the strict train-only teacher contract and persist the manifest.

    Writes ``artifacts/{variant}/train_manifest.json`` next to ``best.pth``. The
    manifest records:

      * the exact patch IDs used for training (post dominance-filter),
      * SHA-256 hashes of the official split files (train/val/test),
      * val/test patch IDs plus a frozen confirmation that the train IDs are
        disjoint from both.

    Raises RuntimeError instead of writing if train leaks into val/test.
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    train_val_disjoint = train_set.isdisjoint(val_set)
    train_test_disjoint = train_set.isdisjoint(test_set)
    if not (train_val_disjoint and train_test_disjoint):
        raise RuntimeError(
            f"Refusing to write train manifest: teacher train ids overlap val/test "
            f"(train_val_disjoint={train_val_disjoint}, train_test_disjoint={train_test_disjoint})"
        )
    manifest = {
        "variant": variant,
        "checkpoint": "best.pth",
        "source": source,
        "train_patch_ids": sorted(train_set),
        "val_patch_ids": sorted(val_set),
        "test_patch_ids": sorted(test_set),
        "split_hashes_sha256": {
            "train": _sha256_file(splits_dir / "train_X.txt"),
            "val": _sha256_file(splits_dir / "val_X.txt"),
            "test": _sha256_file(splits_dir / "test_X.txt"),
        },
        "train_val_disjoint": train_val_disjoint,
        "train_test_disjoint": train_test_disjoint,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"train manifest (teacher contract) -> {out_dir / 'train_manifest.json'}")
    return manifest


def reaudit_teacher_manifest(variant: str) -> dict:
    """Regenerate train_manifest.json for an existing checkpoint from the official splits.

    Used to bring legacy teacher checkpoints (already trained on the official
    train split via train_one.py) under the leak-free manifest audit without
    retraining. Records the full official train pool as train_patch_ids, which
    is the conservative superset of whatever dominance-filtered subset was
    actually trained on.
    """
    splits = load_official_splits(ROOT / "splits")
    marida = MaridaDataset(ROOT)
    assert_scene_disjoint(splits, marida)
    return write_train_manifest(
        variant,
        ROOT / "artifacts" / variant,
        ROOT / "splits",
        list(splits["train"]),
        list(splits["val"]),
        list(splits["test"]),
        source="reaudit_official_splits",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-power", type=float, default=0.5, help="Square-root inverse weighting (dampens extreme penalty on false positives)")
    parser.add_argument("--min-debris-ratio", type=float, default=0.55, help="Enforce >= this fraction of debris-positive patches in every training batch")
    parser.add_argument("--filter-empty-dominance", type=float, default=0.90, help="Drop real patches with > X% single-class dominance and zero debris pixels (0 disables)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reaudit-manifest", action="store_true", help="Regenerate train_manifest.json for an existing checkpoint from the official splits (no training)")
    parser.add_argument("--encoder-weights", choices=["imagenet_physics", "imagenet_replicate", "seco_s2", "random"], default="imagenet_physics",
                        help="First-convolution weight initialization strategy for pretrained encoders")
    parser.add_argument("--scheduler", choices=["cosine_warmup", "plateau", "cosine", "none"],
                        default="cosine_warmup",
                        help="LR scheduler. Default: cosine_warmup (3-epoch warmup + cosine decay).")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lambda-lovasz", type=float, default=0.0, help="Weight for Lovász-Softmax loss term in compound loss")
    return parser.parse_args()


def _train_class_weights(marida, train_ids, power: float) -> torch.Tensor:
    counts = np.zeros(N_CLASSES + 1, dtype=np.int64)
    for pid in train_ids:
        counts += np.bincount(marida.read_mask(pid).reshape(-1), minlength=N_CLASSES + 1)[: N_CLASSES + 1]
    labeled = counts[1:].sum()
    weights = ((labeled + 1.0) / (counts[1:] + 1.0)) ** power
    weights = weights / weights.min()
    print(f"class weights: {np.round(weights, 3).tolist()}")
    return torch.from_numpy(weights.astype(np.float32))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if args.reaudit_manifest:
        manifest = reaudit_teacher_manifest(args.variant)
        print(json.dumps(manifest, indent=2))
        return

    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    assert_scene_disjoint(splits, marida)

    if args.smoke:
        splits = {
            "train": list(marida.ids[:40]),
            "val": list(marida.ids[40:60]),
            "test": list(marida.ids[60:80]),
        }

    class_weights_tensor = _train_class_weights(marida, splits["train"], args.weight_power)

    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        run_name=f"{args.variant}",
        class_weights=class_weights_tensor,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        lambda_lovasz=args.lambda_lovasz,
    )

    model = resolve_variant(args.variant, weight_init_strategy=args.encoder_weights)
    print(f"model={args.variant}, params={sum(p.numel() for p in model.parameters()):,}")
    print(f"weight_init_strategy={args.encoder_weights}")

    loaders = build_loaders(
        marida,
        splits,
        batch_size=args.batch_size,
        train_transform=TrainTransform(),
        min_debris_ratio=args.min_debris_ratio,
        filter_empty_dominance=args.filter_empty_dominance,
    )
    trainer = Trainer(model, config, device)
    result = trainer.run(loaders["train"], loaders["val"])
    print(f"best epoch={result['best_epoch']}, best val mean_iou={result['best_mean_iou']}")

    train_ids_used = list(loaders["train"].dataset.ids)
    write_train_manifest(
        args.variant,
        ROOT / "artifacts" / args.variant,
        ROOT / "splits",
        train_ids_used,
        splits["val"],
        splits["test"],
    )


if __name__ == "__main__":
    main()