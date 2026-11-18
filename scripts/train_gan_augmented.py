"""Train baseline with GAN-augmented data (semi-supervised pseudo-labeling).

Pipeline:
  1. Load trained generator from artifacts/gan/generator.pt
  2. Generate synthetic multispectral patches
  3. Pseudo-label via the teacher baseline checkpoint (artifacts/baseline/best.pth)
  4. Filter synthetic samples by teacher confidence AND a strict minority-class
     criterion: only patches whose pseudo-label contains >= min_debris_pixels
     Class 1 (Marine Debris) pixels are kept. Sampling repeats until the
     debris-bearing quota (--n-debris-synthetic) is satisfied.
     Pseudo-labels are additionally validated by ENSEMBLE AGREEMENT against a
     second independently-trained model (default: the sibling of the teacher,
     e.g. attention validates baseline) or by MC-dropout consensus, breaking
     the circularity of a single teacher filtering its own optima.
  5. Mix real + filtered synthetic into the train DataLoader
  6. Retrain the baseline variant with the compound Focal+Dice loss and
     evaluate on the official test split

Usage:
  python scripts/train_gan_augmented.py
  python scripts/train_gan_augmented.py --n-synthetic 400 --epochs 20 --min-fraction 0.6
  python scripts/train_gan_augmented.py --variant attention --n-debris-synthetic 200
  python scripts/train_gan_augmented.py --agreement-variant mc --mc-dropout-passes 6
  python scripts/train_gan_augmented.py --agreement-variant attention --agreement-threshold 0.75
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.augment import TrainTransform
from src.data.dataset import DebrisBalancedBatchSampler, MaridaSegmentationDataset, SyntheticSegmentationDataset
from src.data.encoding import IGNORE_INDEX, N_CLASSES
from src.data.ingestion import MaridaDataset
from src.data.splits import assert_scene_disjoint, load_official_splits
from src.data.gan_augment import Generator, generate_samples
from src.evaluate.metrics import aggregate_metrics, evaluate_predictions
from src.models.registry import resolve_variant
from src.train.losses import DEBRIS_INDEX
from src.train.trainer import TrainConfig, Trainer

PATCH_H, PATCH_W = 256, 256
PATCH_PIXELS = PATCH_H * PATCH_W


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="baseline", help="Model variant to retrain with GAN data")
    p.add_argument("--teacher-variant", default=None, help="Variant to pseudo-label (default: same as --variant)")
    p.add_argument("--n-synthetic", type=int, default=300, help="Number of synthetic samples generated per sampling round before filtering")
    p.add_argument("--n-debris-synthetic", type=int, default=200, help="Target number of debris-bearing synthetic patches to keep (Class 1 quota)")
    p.add_argument("--min-debris-pixels", type=int, default=15, help="Minimum Class 1 pixels a pseudo-label must contain to be kept")
    p.add_argument("--debris-pixel-floor-fraction", type=float, default=0.0005, help="Absolute Class 1 floor as a fraction of patch area (0.05%%), ORed with --min-debris-pixels")
    p.add_argument("--max-generation-rounds", type=int, default=60, help="Cap on sampling rounds before giving up on the debris quota")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--confidence", type=float, default=0.8, help="Softmax confidence threshold for counting a confident pixel")
    p.add_argument("--min-fraction", type=float, default=0.05, help="Absolute floor for the confident-pixel fraction (kept if above it)")
    p.add_argument("--threshold-quantile", type=float, default=0.25, help="Quantile of the real-data confident-fraction used as the effective keep threshold")
    p.add_argument("--calibrate-n", type=int, default=60, help="Number of real training patches used to calibrate the keep threshold")
    p.add_argument("--weight-power", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss focusing parameter (compound_focal_dice loss)")
    p.add_argument("--lambda-dice", type=float, default=1.0, help="Weight of the multiclass Dice term in the compound loss")
    p.add_argument("--min-debris-ratio", type=float, default=0.45, help="Enforce >= this fraction of debris-positive patches in every training batch")
    p.add_argument("--filter-empty-dominance", type=float, default=0.95, help="Drop real patches with > X%% single-class dominance and zero debris pixels (0 disables; default 0.95)")
    p.add_argument("--agreement-variant", default=None, help="Second independently-trained model for pseudo-label agreement (breaks circular validation). Default: the sibling of --teacher-variant (baseline<->attention). 'mc' enables Monte Carlo dropout instead.")
    p.add_argument("--agreement-threshold", type=float, default=0.75, help="Minimum ensemble/MC agreement fraction for a pseudo-label to be accepted (default 0.75)")
    p.add_argument("--mc-dropout-passes", type=int, default=4, help="Number of stochastic forward passes when --agreement-variant mc is used")
    p.add_argument("--dropout-p", type=float, default=0.2, help="Dropout probability for the MC-dropout agreement model")
    p.add_argument("--run-name", default=None, help="Subdirectory for artifacts output")
    p.add_argument("--smoke", action="store_true", help="Fast smoke run: shrink splits and cap the debris quota. The teacher leak-audit still runs on the full official splits.")
    return p.parse_args()


@torch.no_grad()
def _forward_softmax(model, batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model in eval mode and return (probs (B,C,H,W), argmax preds (B,H,W))."""
    model.eval()
    with torch.autocast(device_type=device.type, enabled=True):
        logits = model(batch)
    probs = F.softmax(logits.float(), dim=1)
    return probs, probs.argmax(dim=1)


def _enable_mc_dropout(model, p: float) -> int:
    """Switch every Dropout* module to train mode so eval keeps its stochasticity.

    Returns how many dropout modules were enabled (0 means MC-dropout provides
    no signal and agreement is tautological).
    """
    n_enabled = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = float(p)
            module.train()
            n_enabled += 1
    return n_enabled


@torch.no_grad()
def agreement_scores(
    primary,
    primary_batch,
    device: torch.device,
    agreement_variant: str | None,
    mc_dropout_passes: int,
    dropout_p: float,
) -> tuple[torch.Tensor, str, bool]:
    """Return per-sample agreement fraction between pseudo-label sources.

    Two non-circular strategies:

    * ensemble (default): the primary teacher's argmax must agree with a second
      independently-trained model's argmax (baseline<->attention).
    * mc (Monte Carlo dropout): repeat stochastic forward passes of the primary
      teacher and count pixels where every pass predicts the same class.

    The agreement fraction is the pixel-level consensus rate per sample.
    Returns (agreement (B,), mode, degenerate_warning).
    """
    _, preds = _forward_softmax(primary, primary_batch, device)
    if agreement_variant and agreement_variant.startswith("mc"):
        n_dropout = _enable_mc_dropout(primary, dropout_p)
        if n_dropout == 0:
            return torch.ones_like(preds, dtype=torch.float32), "mc", True
        import random as _random
        torch.manual_seed(0)
        _random.seed(0)
        votes = []
        for _ in range(mc_dropout_passes):
            _, p = _forward_softmax(primary, primary_batch, device)
            votes.append(p)
        stacked = torch.stack(votes, dim=0)  # (P, B, H, W)
        consensus = torch.mode(stacked, dim=0).values  # (B, H, W)
        agreement = (stacked == consensus.unsqueeze(0)).float().mean(dim=0).mean(dim=(1, 2))
        return agreement, "mc", False
    if agreement_variant == "self":
        # Fall back to the raw confidence fraction (legacy behaviour).
        assert primary is not None
        probs, _ = _forward_softmax(primary, primary_batch, device)
        conf, _ = probs.max(dim=1)
        return conf.float().mean(dim=(1, 2)), "self", True
    raise ValueError(f"unknown agreement mode: {agreement_variant!r}")


@torch.no_grad()
def pseudo_label(
    teacher,
    agreement_model,
    images: torch.Tensor,
    device: torch.device,
    confidence: float,
    min_fraction: float,
    min_debris_pixels: int,
    agreement_variant: str | None = None,
    agreement_threshold: float = 0.0,
    mc_dropout_passes: int = 4,
    dropout_p: float = 0.2,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pseudo-label images with the teacher, validated by ensemble agreement.

    A sample is kept ONLY when ALL hold:
      * confident-pixel fraction >= min_fraction (softmax confidence), and/or
        - if agreement_variant == 'self', the confidence fraction IS the
          agreement score and must pass agreement_threshold,
      * the pseudo-label contains >= min_debris_pixels Class 1 (debris) pixels,
      * the agreement score (ensemble argmax consensus or MC-dropout consensus)
        >= agreement_threshold.
    """
    teacher.eval()
    if agreement_model is not None:
        agreement_model.eval()
    kept_images, kept_targets, confidences, debris_counts, kept_debris_counts = [], [], [], [], []
    kept_agreements: list[float] = []
    N = images.shape[0]
    for i in range(0, N, batch_size):
        batch = images[i:i + batch_size].to(device, non_blocking=True)
        probs, preds = _forward_softmax(teacher, batch, device)
        confs = probs.max(dim=1).values
        if agreement_model is not None:
            _, agree_preds = _forward_softmax(agreement_model, batch, device)
            agreement = (preds == agree_preds).float().mean(dim=(1, 2))
        else:
            agreement, _, _ = agreement_scores(
                teacher, batch, device, agreement_variant, mc_dropout_passes, dropout_p
            )
        for j in range(preds.shape[0]):
            keep_frac = float((confs[j] >= confidence).float().mean())
            confidences.append(keep_frac)
            n_debris = int((preds[j] == DEBRIS_INDEX).sum().item())
            debris_counts.append(n_debris)
            agree_j = float(agreement[j].item())
            agree_ok = agree_j >= agreement_threshold
            if keep_frac >= min_fraction and n_debris >= min_debris_pixels and agree_ok:
                kept_images.append(images[i + j].numpy().astype(np.float32))
                kept_targets.append(preds[j].cpu().numpy().astype(np.int64))
                kept_debris_counts.append(n_debris)
                kept_agreements.append(agree_j)

    kept_targets_arr = np.stack(kept_targets, axis=0) if kept_targets else np.zeros((0, PATCH_H, PATCH_W), np.int64)
    kept_labeled = kept_targets_arr[kept_targets_arr != IGNORE_INDEX]
    kept_class1_ratio = float((kept_labeled == DEBRIS_INDEX).mean()) if kept_labeled.size else None
    stats = {
        "n_generated": N,
        "n_kept": len(kept_images),
        "n_debris_kept": len(kept_images),
        "min_debris_pixels": int(min_debris_pixels),
        "confidences": np.array(confidences, dtype=np.float32),
        "mean_confident_fraction": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "debris_counts": np.array(debris_counts, dtype=np.int64),
        "kept_debris_counts": np.array(kept_debris_counts, dtype=np.int64),
        "kept_agreements": np.array(kept_agreements, dtype=np.float32),
        "mean_kept_agreement": round(float(np.mean(kept_agreements)), 4) if kept_agreements else None,
        "kept_class1_ratio": round(kept_class1_ratio, 6) if kept_class1_ratio is not None else None,
    }
    imgs = np.stack(kept_images, axis=0) if kept_images else np.zeros((0, 11, PATCH_H, PATCH_W), np.float32)
    return imgs, kept_targets_arr, stats


def load_teacher_manifest(teacher_variant: str, artifacts_dir: Path = ROOT / "artifacts") -> dict:
    """Load the teacher's train_manifest.json, raising if it is absent."""
    manifest_path = artifacts_dir / teacher_variant / "train_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Teacher contract manifest missing: {manifest_path}. "
            f"Re-train the teacher with scripts/train_one.py --variant {teacher_variant} "
            f"(or --reaudit-manifest) to emit train_manifest.json before pseudo-labeling."
        )
    return json.loads(manifest_path.read_text())


def verify_teacher_no_leakage(
    teacher_variant: str,
    splits: dict[str, list[str]],
    artifacts_dir: Path = ROOT / "artifacts",
) -> dict:
    """Pre-flight teacher contract check.

    Returns a dict with ``teacher_leakage_verified`` and disjointness audit
    fields. Raises ``RuntimeError("Potential data leakage detected in teacher
    model!")`` if any official val/test patch id appears in the teacher's
    recorded training set, or if the manifest is missing/unverifiable.
    """
    manifest = load_teacher_manifest(teacher_variant, artifacts_dir)
    teacher_train = set(manifest.get("train_patch_ids") or [])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])
    val_overlap = sorted(teacher_train & val_ids)
    test_overlap = sorted(teacher_train & test_ids)
    if val_overlap or test_overlap:
        raise RuntimeError(
            f"Potential data leakage detected in teacher model! "
            f"teacher={teacher_variant} train_manifest.json overlaps "
            f"val={val_overlap[:5]} test={test_overlap[:5]}"
        )
    print(
        f"teacher leak pre-flight OK: {teacher_variant} train_patch_ids "
        f"({len(teacher_train)}) disjoint from val ({len(val_ids)}) and test ({len(test_ids)})"
    )
    return {
        "teacher_variant": teacher_variant,
        "teacher_leakage_verified": len(val_overlap) == 0 and len(test_overlap) == 0,
        "n_teacher_train": len(teacher_train),
        "n_val_overlap": len(val_overlap),
        "n_test_overlap": len(test_overlap),
    }


def _train_class_weights(marida: MaridaDataset, train_ids: list[str], power: float) -> tuple[torch.Tensor, float]:
    """Compute CE class weights from real training patches only (original convention).

    Returns (weights, class1_ratio) where class1_ratio is the fraction of
    labeled pixels in the real training pool that belong to Class 1
    (Marine Debris, 11-class value 1).
    """
    counts = np.zeros(N_CLASSES + 1, dtype=np.int64)
    for pid in train_ids:
        counts += np.bincount(marida.read_mask(pid).reshape(-1), minlength=N_CLASSES + 1)[: N_CLASSES + 1]
    labeled = counts[1:].sum()
    if labeled == 0:
        return torch.ones(N_CLASSES, dtype=torch.float32), 0.0
    weights = ((labeled + 1.0) / (counts[1:] + 1.0)) ** power
    weights = weights / weights.min()
    class1_ratio = float(counts[1]) / float(labeled)
    print(f"real-only class weights: {np.round(weights, 3).tolist()}")
    print(f"real Class 1 (Marine Debris) representation: {class1_ratio * 100:.4f}% of labeled pixels")
    return torch.from_numpy(weights.astype(np.float32)), class1_ratio


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    marida = MaridaDataset(ROOT)
    official_splits = load_official_splits(ROOT / "splits")
    assert_scene_disjoint(official_splits, marida)
    splits = dict(official_splits)

    if args.smoke:
        print("[smoke] shrinking splits for a fast run (teacher leak-audit still uses official splits)")
        splits = {
            "train": list(official_splits["train"][:40]),
            "val": list(official_splits["val"][:8]),
            "test": list(official_splits["test"][:8]),
        }
        args.calibrate_n = min(args.calibrate_n, 8)
        args.n_debris_synthetic = min(args.n_debris_synthetic, 8)
        args.max_generation_rounds = min(args.max_generation_rounds, 2)
        args.epochs = min(args.epochs, 1)
        args.min_debris_pixels = min(args.min_debris_pixels, 1)  # exercise synthetic retention
        args.min_fraction = 0.0  # relax confident-fraction floor for smoke retention
        args.agreement_threshold = 0.0  # keep smoke deterministic (no consensus cutoff)

    # ---- 1. Load teacher and pseudo-label synthetic samples ----------------
    teacher_var = args.teacher_variant or args.variant
    teacher_ckpt = ROOT / "artifacts" / teacher_var / "best.pth"
    if not teacher_ckpt.exists():
        raise SystemExit(f"teacher checkpoint not found: {teacher_ckpt}")
    teacher = resolve_variant(teacher_var, pretrained=False)
    from src.utils.checkpoint import safe_load_state_dict
    teacher_epoch = safe_load_state_dict(teacher, teacher_ckpt, map_location="cpu")
    teacher.to(device).eval()
    print(f"teacher={teacher_var} (epoch={teacher_epoch if teacher_epoch is not None else '?'})")

    # PRE-FLIGHT LEAKAGE VERIFICATION: the teacher must have been trained on a
    # train-only patch set (train_manifest.json), completely disjoint from the
    # OFFICIAL val/test splits before any pseudo-labeling happens. This runs on
    # the official split pool regardless of smoke-mode shrinking.
    teacher_audit = verify_teacher_no_leakage(teacher_var, official_splits, ROOT / "artifacts")
    assert teacher_audit["teacher_leakage_verified"], "teacher leak check must pass before pseudo-labeling"

    # Breaks circular validation: agreement is checked against a SECOND model
    # that was not the one optimized to produce these pseudo-labels. Default
    # picks the sibling variant between baseline and attention.
    agreement_var = args.agreement_variant
    agreement_model = None
    if agreement_var is None:
        sibling = "attention" if teacher_var == "baseline" else "baseline"
        if teacher_var in ("baseline", "attention") and (ROOT / "artifacts" / sibling / "best.pth").exists():
            agreement_var = sibling
    if agreement_var and not agreement_var.startswith("mc") and agreement_var != "self":
        agree_ckpt = ROOT / "artifacts" / agreement_var / "best.pth"
        if not agree_ckpt.exists():
            raise SystemExit(f"agreement checkpoint not found for --agreement-variant {agreement_var}: {agree_ckpt}")
        agreement_model = resolve_variant(agreement_var, pretrained=False)
        from src.utils.checkpoint import safe_load_state_dict
        agree_epoch = safe_load_state_dict(agreement_model, agree_ckpt, map_location="cpu")
        agreement_model.to(device).eval()
        print(f"agreement model={agreement_var} (epoch={agree_epoch if agree_epoch is not None else '?'}) [ensemble consensus validation]")
    else:
        mode = agreement_var if agreement_var else "self"
        if mode == "self":
            print("  WARNING: no second model and no MC-dropout agreement configured; "
                  "pseudo-labels are filtered by the single teacher only (circular validation). "
                  "Pass --agreement-variant (e.g. the sibling variant, or 'mc') to break circularity.")
        else:
            print(f"Pseudo-label validation uses MC-dropout agreement ({args.mc_dropout_passes} passes, p={args.dropout_p})")

    gen = Generator(latent_dim=args.latent_dim).to(device)
    gen_ckpt = ROOT / "artifacts" / "gan" / "generator.pt"
    gen.load_state_dict(torch.load(gen_ckpt, map_location=device, weights_only=True))
    print(f"loading trained generator from {gen_ckpt}")

    print("pseudo-labeling via teacher and filtering (debris-targeted)...")

    # Calibrate the keep threshold against real data - TRAIN SPLIT ONLY.
    sample_ids = splits["train"][: args.calibrate_n]
    # Explicit leakage assertions: calibration must sample ONLY from the train split.
    assert set(sample_ids).issubset(set(splits["train"])), "Calibration set leaked into non-train splits!"
    assert set(sample_ids).isdisjoint(set(splits["val"])), "Validation leak in calibration!"
    assert set(sample_ids).isdisjoint(set(splits["test"])), "Test leak in calibration!"
    print(f"  calibration purity OK: {len(sample_ids)} samples drawn from splits['train'] only "
          f"(disjoint from val/test)")
    real_fracs = []
    with torch.no_grad():
        for pid in sample_ids:
            img = torch.from_numpy(marida.read_image(pid)).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=True):
                logits = teacher(img)
            probs = F.softmax(logits.float(), dim=1)
            peak = probs[0].max(dim=0)
            real_fracs.append(float((peak.values >= args.confidence).float().mean()))
    calibration_source = "train"
    q = np.quantile(real_fracs, args.threshold_quantile)
    effective_thr = max(q, args.min_fraction)
    print(f"  real confident-fraction: mean={np.mean(real_fracs):.4f}  q{args.threshold_quantile:.2f}={q:.4f}  effective_thr={effective_thr:.4f}")

    # Debris quota sampling: keep generating until we have enough debris-bearing
    # synthetic patches or the round cap is hit.
    # Resolve the retention floor per the spec: a patch qualifies at >= 15 Class 1
    # pixels OR >= 0.05% of patch area, i.e. the effective threshold is the lower
    # of the two (default min(15, 33) = 15).
    floor_debris_px = int(round(args.debris_pixel_floor_fraction * PATCH_PIXELS))
    min_debris_eff = min(args.min_debris_pixels, floor_debris_px)
    print(
        f"  debris retention criterion: >= {min_debris_eff} Class 1 pixels "
        f"(min( --min-debris-pixels={args.min_debris_pixels}, 0.05% floor={floor_debris_px} px) "
        f"=> {min_debris_eff / PATCH_PIXELS * 100:.4f}% of {PATCH_PIXELS} px)"
    )
    print(f"  debris quota: {args.n_debris_synthetic} synthetic patches (max {args.max_generation_rounds} sampling rounds of {args.n_synthetic})")

    kept_images, kept_targets = [], []
    n_generated_total = 0
    n_debris_kept_total = 0
    all_conf_fracs: list[float] = []
    kept_debris_counts: list[int] = []
    for round_idx in range(1, args.max_generation_rounds + 1):
        remaining_quota = args.n_debris_synthetic - n_debris_kept_total
        if remaining_quota <= 0:
            break
        raw_batch = generate_samples(gen, device, args.n_synthetic, args.latent_dim)
        imgs, tgts, batch_stats = pseudo_label(
            teacher, agreement_model, raw_batch, device,
            confidence=args.confidence,
            min_fraction=effective_thr,
            min_debris_pixels=min_debris_eff,
            agreement_variant=agreement_var,
            agreement_threshold=args.agreement_threshold,
            mc_dropout_passes=args.mc_dropout_passes,
            dropout_p=args.dropout_p,
        )
        n_generated_total += int(batch_stats["n_generated"])
        n_debris_kept_total += int(batch_stats["n_debris_kept"])
        all_conf_fracs.extend(batch_stats["confidences"].tolist())
        if imgs.shape[0] > 0:
            kept_images.append(imgs)
            kept_targets.append(tgts)
            kept_debris_counts.extend(batch_stats["kept_debris_counts"].tolist())
            print(
                f"  [round {round_idx}/{args.max_generation_rounds}] generated={batch_stats['n_generated']} "
                f"kept={batch_stats['n_kept']} debris-kept={batch_stats['n_debris_kept']} "
                f"(quota {n_debris_kept_total}/{args.n_debris_synthetic})"
            )
        else:
            print(
                f"  [round {round_idx}/{args.max_generation_rounds}] generated={batch_stats['n_generated']} "
                f"kept={batch_stats['n_kept']} -> ZERO debris-bearing patches; resampling to satisfy quota"
            )

    if n_debris_kept_total < args.n_debris_synthetic:
        print(
            f"  WARNING: debris quota not fully reached ({n_debris_kept_total}/{args.n_debris_synthetic}); "
            f"proceeding with {n_debris_kept_total} debris-positive synthetic patches"
        )

    if kept_images:
        synth_images = np.concatenate(kept_images, axis=0)
        synth_targets = np.concatenate(kept_targets, axis=0)
    else:
        synth_images = np.zeros((0, 11, PATCH_H, PATCH_W), np.float32)
        synth_targets = np.zeros((0, PATCH_H, PATCH_W), np.int64)

    # ---- 2. Build filtered real dataset -----------------------------------
    filter_thr = args.filter_empty_dominance
    real_ds = MaridaSegmentationDataset(
        marida,
        splits["train"],
        transform_fn=TrainTransform(),
        filter_empty_dominance=None if filter_thr is None or filter_thr <= 0 else filter_thr,
    )
    print(
        f"  dominance filter (> {filter_thr} single-class & zero-debris, 0 disables): "
        f"dropped {real_ds.n_dropped}/{len(real_ds) + real_ds.n_dropped} real train patches, "
        f"kept {len(real_ds)} (debris-positive: {real_ds.n_debris_positive}, "
        f"negative: {len(real_ds) - real_ds.n_debris_positive})"
    )

    # ---- 3. Class weights + Class 1 representation (real pool = filtered) ----
    class_weights_tensor, real_class1_ratio = _train_class_weights(marida, real_ds.ids, args.weight_power)
    if synth_targets.shape[0] > 0:
        synth_labeled = synth_targets[synth_targets != IGNORE_INDEX]
        synth_class1_ratio = float((synth_labeled == DEBRIS_INDEX).mean()) if synth_labeled.size else 0.0
        kept_debris = np.array(kept_debris_counts, dtype=np.int64)
        print(
            f"  generated total={n_generated_total}  kept (debris-positive)={synth_targets.shape[0]}  "
            f"quota target={args.n_debris_synthetic}"
        )
        print(f"  kept patch Class 1 pixel counts: min={kept_debris.min() if kept_debris.size else 0} "
              f"max={kept_debris.max() if kept_debris.size else 0} "
              f"mean={kept_debris.mean() if kept_debris.size else 0:.2f}")
        print(
            f"  Class 1 representation: synthetic pool={synth_class1_ratio * 100:.4f}%  "
            f"real pool={real_class1_ratio * 100:.4f}% of labeled pixels "
            f"(synthetic/real ratio={synth_class1_ratio / real_class1_ratio:.2f}x)"
        )
        print(f"  target value distribution (kept): {dict(zip(*np.unique(synth_targets, return_counts=True)))}")
    else:
        synth_class1_ratio = 0.0
        print(f"  WARNING: synthetic pool empty after debris-targeted filtering; training on real data only")
    if all_conf_fracs:
        print(f"  overall mean confident-fraction across generated samples: {np.mean(all_conf_fracs):.4f}")

    # ---- 4. Build combined dataset + balanced batches ---------------------
    synth_ds = (
        SyntheticSegmentationDataset(synth_images, synth_targets, transform_fn=TrainTransform())
        if synth_images.shape[0] > 0
        else None
    )
    datasets = [real_ds] if synth_ds is None else [real_ds, synth_ds]
    train_ds = ConcatDataset(datasets) if len(datasets) > 1 else real_ds
    print(f"train set: {len(real_ds)} real + {len(synth_ds) if synth_ds else 0} synthetic = {len(train_ds)} total")

    # Debris-balanced batches: every batch must contain >= min_debris_ratio
    # debris-positive patches. Real positives come from real_ds.debris_indices;
    # all kept synthetic patches are debris-positive by construction.
    if args.min_debris_ratio is not None and args.min_debris_ratio > 0:
        real_pos_indices = real_ds.debris_indices
        synth_pos_start = len(real_ds)
        synth_count = len(synth_ds) if synth_ds else 0
        positive_indices = real_pos_indices + list(range(synth_pos_start, synth_pos_start + synth_count))
        train_sampler = DebrisBalancedBatchSampler(
            data_len=len(train_ds),
            positive_indices=positive_indices,
            batch_size=args.batch_size,
            min_debris_ratio=args.min_debris_ratio,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
        print(f"  debris-balanced batches: >= {args.min_debris_ratio:.0%} debris-positive per batch "
              f"({len(positive_indices)} positive / {len(train_ds)} total)")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    run_name = args.run_name or (f"{args.variant}_gan_augmented_smoke" if args.smoke else f"{args.variant}_gan_augmented")
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        run_name=run_name,
        class_weights=class_weights_tensor,
        loss_type="compound_focal_dice",
        focal_gamma=args.focal_gamma,
        lambda_dice=args.lambda_dice,
    )
    from src.data.augment import identity
    val_ds = MaridaSegmentationDataset(marida, splits["val"], transform_fn=identity)
    test_ds = MaridaSegmentationDataset(marida, splits["test"], transform_fn=identity)
    loaders = {
        "train": train_loader,
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
    }

    # ---- 5. Train ---------------------------------------------------------
    model = resolve_variant(args.variant)
    print(f"model={args.variant}, params={sum(p.numel() for p in model.parameters()):,}")
    trainer = Trainer(model, config, device)
    result = trainer.run(loaders["train"], loaders["val"])
    print(f"best epoch={result['best_epoch']}, best val mIoU={result['best_mean_iou']:.4f}")

    # ---- 6. Evaluate on official test split --------------------------------
    run_dir = Path(config.run_dir) / config.run_name
    from src.utils.checkpoint import safe_load_state_dict
    eval_epoch = safe_load_state_dict(model, run_dir / "best.pth", map_location="cpu")
    model.to(device).eval()
    metrics_list = []
    with torch.no_grad():
        for image, mask in tqdm(loaders["test"], desc="test"):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=True):
                logits = model(image)
            metrics_list.append(evaluate_predictions(logits.float(), mask))
    summary = aggregate_metrics(metrics_list)
    output = {
        "mean_iou": summary["mean_iou"],
        "accuracy": summary["accuracy"],
        "f1_class_1_marine_debris": summary["f1_class_1_marine_debris"],
        "iou_class_1_marine_debris": summary["iou_class_1_marine_debris"],
        "iou_per_class": summary["iou_per_class"],
        "f1_per_class": summary["f1_per_class"],
        "checkpoint_epoch": eval_epoch if eval_epoch is not None else result["best_epoch"],
        "n_real": len(real_ds),
        "n_real_dropped_by_dominance": int(real_ds.n_dropped),
        "n_synthetic_kept": int(synth_images.shape[0]),
        "n_synthetic_generated_total": int(n_generated_total),
        "n_debris_synthetic_target": int(args.n_debris_synthetic),
        "min_debris_pixels_effective": int(min_debris_eff),
        "min_debris_ratio": args.min_debris_ratio,
        "filter_empty_dominance": None if filter_thr is None or filter_thr <= 0 else filter_thr,
        "max_generation_rounds": args.max_generation_rounds,
        "synthetic_class1_ratio": round(synth_class1_ratio, 6),
        "real_class1_ratio": round(real_class1_ratio, 6),
        "teacher": teacher_var,
        "confidence_threshold": args.confidence,
        "keep_threshold_effective": round(effective_thr, 4),
        "agreement_variant": agreement_var,
        "agreement_threshold": args.agreement_threshold,
        "mc_dropout_passes": args.mc_dropout_passes,
        "dropout_p": args.dropout_p,
        "focal_gamma": args.focal_gamma,
        "lambda_dice": args.lambda_dice,
        "loss_type": "compound_focal_dice",
    }
    out_dir = ROOT / "artifacts" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(output, indent=2))
    print(f"\nofficial test metrics -> {out_dir / 'test_metrics.json'}")
    print(f"  test mean_iou={output['mean_iou']}, f1_debris={output['f1_class_1_marine_debris']}, accuracy={output['accuracy']}")

    # ---- 7. Leakage compliance audit report --------------------------------
    # Synthetic samples are strictly isolated to the training loader: val_ds /
    # test_ds are built directly from the official splits with the identity
    # transform, and the synthetic dataset is only ever part of the train set.
    val_leak_ids = sorted(set(real_ds.ids) & set(splits["val"]))
    test_leak_ids = sorted(set(real_ds.ids) & set(splits["test"]))
    assert not val_leak_ids, f"real train pixels leaked into val: {val_leak_ids[:5]}"
    assert not test_leak_ids, f"real train pixels leaked into test: {test_leak_ids[:5]}"
    for name, ds in (("val", loaders["val"].dataset), ("test", loaders["test"].dataset)):
        assert not isinstance(ds, ConcatDataset), f"synthetic ConcatDataset leaked into {name} loader"
        assert not isinstance(ds, SyntheticSegmentationDataset), f"synthetic samples leaked into {name} loader"
        assert ds.transform_fn is identity, f"{name} loader must use the identity transform (unaugmented)"

    leakage_report = {
        "teacher_variant": teacher_var,
        "teacher_checkpoint": str(teacher_ckpt),
        "teacher_leakage_verified": bool(teacher_audit["teacher_leakage_verified"]),
        "teacher_manifest": str(ROOT / "artifacts" / teacher_var / "train_manifest.json"),
        "num_real_train": int(len(real_ds)),
        "num_synthetic_generated": int(n_generated_total),
        "num_synthetic_retained": int(synth_images.shape[0]),
        "val_leak_check": not val_leak_ids,
        "test_leak_check": not test_leak_ids,
        "calibration_source": calibration_source,
        "calibration_n": int(len(sample_ids)),
        "synthetic_isolation": "synthetic appended to train only; val/test built from official splits with identity transform",
        "test_evaluation_purity": "100% unaugmented official test split",
        "num_val_split_used": int(len(splits["val"])),
        "num_test_split_used": int(len(splits["test"])),
        "smoke": bool(args.smoke),
    }
    (out_dir / "leakage_compliance_report.json").write_text(json.dumps(leakage_report, indent=2))
    print(f"leakage compliance report -> {out_dir / 'leakage_compliance_report.json'}")
    print(f"  teacher_leakage_verified={leakage_report['teacher_leakage_verified']}, "
          f"val_leak_check={leakage_report['val_leak_check']}, test_leak_check={leakage_report['test_leak_check']}")


if __name__ == "__main__":
    main()