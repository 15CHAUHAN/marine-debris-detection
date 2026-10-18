"""Verify WGAN-GP synthetic data quality against real MARIDA training patches.

Standalone evaluation of a saved generator checkpoint:

  1. Load real official-train patches via MaridaDataset.
  2. Generate N synthetic 11-channel patches with the generator.
  3. Compute RGB-FID, Spectral Angle Mapper (SAM), and Band Correlation MSE.
  4. Write artifacts/gan_verification/gan_quality_metrics.json.
  5. Build side-by-side visual gallery PNG (real RGB / synthetic RGB /
     synthetic false-color IR) and an interactive HTML inspection sheet.
  6. Print a PASS/FAIL report (targets: RGB-FID < 50, SAM < 10 deg).

Usage:
  python scripts/verify_gan_quality.py --n-samples 200
  python scripts/verify_gan_quality.py --generator-path artifacts/gan/generator.pt --n-samples 20
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.gan_augment import Generator, generate_samples
from src.data.ingestion import MaridaDataset
from src.data.splits import load_official_splits
from src.evaluation.gan_metrics import (
    NIR_FALSE_COLOR_BANDS,
    TRUE_COLOR_BANDS,
    compute_all_metrics,
    to_rgb_composite,
)

FID_PASS_THRESHOLD = 50.0
SAM_DEG_PASS_THRESHOLD = 10.0
GALLERY_ROWS = 6


def load_real_patches(patch_ids: list[str], n: int, seed: int = 42) -> torch.Tensor:
    """Load up to n real training patches as (n, 11, 256, 256) float32."""
    rng = np.random.default_rng(seed)
    chosen = list(rng.choice(patch_ids, size=min(n, len(patch_ids)), replace=False))
    marida = MaridaDataset(ROOT)
    images = [marida.read_image(pid) for pid in chosen]
    return torch.from_numpy(np.stack(images))


def _figure_grid(images_3ch: list[np.ndarray], titles: list[str], ncols: int, out_path: Path) -> None:
    """Render a row of 3-channel [0,1] HWC images into a matplotlib figure."""
    nrows = (len(images_3ch) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    axes = np.atleast_2d(axes)
    for i, (img, title) in enumerate(zip(images_3ch, titles)):
        ax = axes[i // ncols, i % ncols]
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for i in range(len(images_3ch), nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle(out_path.stem.replace("_", " ").title(), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_gallery(
    real_patches: torch.Tensor,
    fake_patches: torch.Tensor,
    out_path: Path,
) -> None:
    """True-color real, true-color synthetic, and false-color IR synthetic rows.

    Top row:   6 real patches (RGB [B4,B3,B2])
    Middle row: 6 synthetic patches (RGB [B4,B3,B2])
    Bottom row: 6 synthetic patches (false-color [B8,B4,B3])
    """
    k = min(GALLERY_ROWS, real_patches.shape[0], fake_patches.shape[0])
    real_rgb = to_rgb_composite(real_patches[:k], TRUE_COLOR_BANDS)
    fake_rgb = to_rgb_composite(fake_patches[:k], TRUE_COLOR_BANDS)
    fake_nir = to_rgb_composite(fake_patches[:k], NIR_FALSE_COLOR_BANDS)

    def hw(arr: np.ndarray) -> list[np.ndarray]:
        return [np.transpose(arr[i], (1, 2, 0)) for i in range(arr.shape[0])]

    fig, axes = plt.subplots(3, k, figsize=(k * 2.6, 3 * 2.6))
    for col in range(k):
        axes[0, col].imshow(hw(real_rgb)[col]); axes[0, col].axis("off")
        axes[1, col].imshow(hw(fake_rgb)[col]); axes[1, col].axis("off")
        axes[2, col].imshow(hw(fake_nir)[col]); axes[2, col].axis("off")
    axes[0, 0].set_ylabel("REAL  (RGB)", fontsize=9, rotation=0, labelpad=30, va="bottom")
    axes[1, 0].set_ylabel("SYNTHETIC  (RGB)", fontsize=9, rotation=0, labelpad=30, va="bottom")
    axes[2, 0].set_ylabel("SYNTHETIC  (IR)", fontsize=9, rotation=0, labelpad=30, va="bottom")
    fig.suptitle("Real vs Synthetic WGAN-GP Sentinel-2 Patches", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  gallery -> {out_path}")


def _matrix_heatmap_png(matrix: np.ndarray, labels: list[str], title: str) -> str:
    """Render an 11x11 correlation heatmap and return its base64 PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, fontsize=7, rotation=45)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_html_report(
    metrics: dict,
    gallery_png: Path,
    out_path: Path,
) -> None:
    """Interactive inspection sheet with metric cards, gallery, and heatmaps."""
    gallery_b64 = _image_base64(gallery_png)
    real_corr = np.asarray(metrics.get("real_correlation"))
    fake_corr = np.asarray(metrics.get("fake_correlation"))
    labels = metrics.get("band_labels", list(map(lambda i: f"B{i}", range(1, 12))))
    real_heat = _matrix_heatmap_png(real_corr, labels, "Real inter-band correlation")
    fake_heat = _matrix_heatmap_png(fake_corr, labels, "Synthetic inter-band correlation")
    diff = (real_corr - fake_corr) ** 2 if real_corr.size else None
    diff_heat = _matrix_heatmap_png(diff, labels, "Squared band-correlation diff") if diff is not None else ""

    def fmt(v) -> str:
        return "n/a" if v is None else f"{float(v):.4f}"

    fid = fmt(metrics.get("rgb_fid"))
    sam = fmt(metrics.get("spectral_angle_deg"))
    sam_mean = fmt(metrics.get("spectral_angle_mean_deg"))
    corr = fmt(metrics.get("band_correlation_mse"))
    samples = metrics.get("samples_evaluated", 0)

    def verdict(metric_key: str, threshold: float) -> str:
        val = metrics.get(metric_key)
        if val is None:
            return '<span class="badge warn">n/a</span>'
        ok = float(val) < threshold
        return f'<span class="badge {"pass" if ok else "fail"}>{("PASS" if ok else "FAIL")}</span>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>WGAN-GP Synthetic Data Quality Inspection</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; background: #f7f8fa; color: #1b2430; }}
h1 {{ font-size: 22px; }}
h2 {{ font-size: 16px; margin-top: 28px; border-bottom: 1px solid #d8dee8; padding-bottom: 4px; }}
.cards {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.card {{ background: #fff; border: 1px solid #d8dee8; border-radius: 10px; padding: 14px 20px; min-width: 170px; box-shadow: 0 1px 3px rgba(16,24,40,.06); }}
.card .value {{ font-size: 26px; font-weight: 700; }}
.card .label {{ font-size: 12px; color: #5b6675; text-transform: uppercase; letter-spacing: .04em; }}
.badge {{ padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }}
.badge.pass {{ background: #e3f5e9; color: #16704a; }}
.badge.fail {{ background: #fde8e8; color: #b3261e; }}
.badge.warn {{ background: #fef3d9; color: #8a5a00; }}
img.gallery {{ width: 100%; border-radius: 8px; border: 1px solid #d8dee8; }}
.heat {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.heat img {{ width: 340px; border-radius: 8px; border: 1px solid #d8dee8; }}
.note {{ color: #5b6675; font-size: 13px; }}
</style></head><body>
<h1>WGAN-GP Synthetic Data Quality — Visual Inspection Sheet</h1>
<p class="note">Sentinel-2 true-color uses bands [B4, B3, B2]; false-color IR uses [B8, B4, B3].
Percentile-normalized per band (2%–98%).</p>
<div class="cards">
  <div class="card"><div class="value">{fid}</div><div class="label">RGB-FID</div>{verdict("rgb_fid", FID_PASS_THRESHOLD)}</div>
  <div class="card"><div class="value">{sam}</div><div class="label">SAM (deg, paired)</div>{verdict("spectral_angle_deg", SAM_DEG_PASS_THRESHOLD)}</div>
  <div class="card"><div class="value">{sam_mean}</div><div class="label">SAM (deg, mean spectrum)</div></div>
  <div class="card"><div class="value">{corr}</div><div class="label">Band-corr MSE</div></div>
  <div class="card"><div class="value">{samples}</div><div class="label">Patches evaluated</div></div>
</div>
<h2>Real vs Synthetic Gallery</h2>
<img class="gallery" src="data:image/png;base64,{gallery_b64}" alt="gallery">
<h2>Inter-Band Correlation Heatmaps</h2>
<div class="heat">
  <img src="data:image/png;base64,{real_heat}" alt="real correlation">
  <img src="data:image/png;base64,{fake_heat}" alt="fake correlation">
  {f'<img src="data:image/png;base64,{diff_heat}" alt="diff">' if diff_heat else ''}
</div>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"  inspection report -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-path", default=str(ROOT / "artifacts" / "gan" / "generator.pt"))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "gan_verification"))
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")

    gen_path = Path(args.generator_path)
    if not gen_path.exists():
        raise SystemExit(f"generator checkpoint not found: {gen_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Real patches (official train split) ----------------------------
    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    real_patches = load_real_patches(splits["train"], args.n_samples, seed=args.seed)
    print(f"real patches: {real_patches.shape}")

    # ---- 2. Synthetic patches ----------------------------------------------
    gen = Generator(latent_dim=args.latent_dim)
    if str(gen_path).endswith((".pt", ".pth")):
        gen.load_state_dict(torch.load(gen_path, map_location="cpu", weights_only=True))
    gen.to(device).eval()
    fake_patches = generate_samples(gen, device, args.n_samples, args.latent_dim)
    fake_patches = fake_patches.clamp(0.0, 1.0)
    print(f"synthetic patches: {fake_patches.shape}")

    # ---- 3. Metrics --------------------------------------------------------
    metrics = compute_all_metrics(real_patches, fake_patches, device=device)
    metrics["generator_checkpoint"] = str(args.generator_path)

    json_out = out_dir / "gan_quality_metrics.json"
    json_out.write_text(json.dumps(metrics, indent=2))
    print(f"metrics -> {json_out}")

    rgb_fid = metrics.get("rgb_fid")
    sam_deg = metrics.get("spectral_angle_deg")
    corr_mse = metrics.get("band_correlation_mse")
    fid_ok = rgb_fid is not None and rgb_fid < FID_PASS_THRESHOLD
    sam_ok = sam_deg is not None and sam_deg < SAM_DEG_PASS_THRESHOLD
    print("\n===== GAN QUALITY REPORT =====")
    print(f"  RGB-FID                 : {rgb_fid:.2f}  (PASS < {FID_PASS_THRESHOLD})" if rgb_fid is not None else "  RGB-FID: n/a")
    print(f"  SAM (deg)               : {sam_deg:.2f}  (PASS < {SAM_DEG_PASS_THRESHOLD})" if sam_deg is not None else "  SAM: n/a")
    print(f"  Band correlation MSE    : {corr_mse:.4f}")
    print(f"  Samples evaluated       : {metrics['samples_evaluated']}")
    print(f"  Generators checkpoint   : {args.generator_path}")
    print(f"  Feature extractor       : {metrics.get('fid_feature_extractor')}")
    overall = fid_ok and sam_ok
    print(f"  OVERALL                 : {'PASS' if overall else 'FAIL'}")
    print("=============================")

    # ---- 4. Gallery + HTML ----------------------------------------------
    gallery_png = out_dir / "real_vs_synthetic_gallery.png"
    build_gallery(real_patches, fake_patches, gallery_png)
    html_path = out_dir / "inspection_report.html"
    build_html_report(metrics, gallery_png, html_path)

    print("\nDone. Outputs:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()