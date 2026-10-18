"""Train WGAN-GP on MARIDA training patches and generate synthetic samples.

Training data: official train split only (scene-disjoint).
After training, the generator is saved and a small set of samples is written
to artifacts/gan_augmented/ for inspection.

Usage:
  python scripts/train_gan.py --epochs 50 --batch-size 8
  python scripts/train_gan.py --epochs 200 --batch-size 16 --lr 0.0001
  python scripts/train_gan.py --generate-only  # generate from saved model

Preview RGB uses Sentinel-2 true-color bands [B4, B3, B2] (indices [3, 2, 1])
with per-band 2%-98% percentile normalization via to_rgb_composite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ingestion import MaridaDataset
from src.data.splits import load_official_splits
from src.data.gan_augment import Generator, Critic, train_gan_epoch, generate_samples
from src.evaluation.gan_metrics import TRUE_COLOR_BANDS, to_rgb_composite


def load_train_images(marida: MaridaDataset, patch_ids: list[str]) -> torch.Tensor:
    """Load training patches as (N, 11, 256, 256) float32 in [0,1]."""
    images = []
    for pid in patch_ids:
        images.append(marida.read_image(pid))
    return torch.from_numpy(np.stack(images))


def save_samples_as_png(tensors: torch.Tensor, out_dir: Path, prefix: str = "sample") -> None:
    """Save the first 8 generated samples as a grid PNG.

    Uses the Sentinel-2 true-color composite [B4, B3, B2] (indices [3, 2, 1]),
    each band percentile-normalized to [0,1] by to_rgb_composite.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(8, tensors.shape[0])
    rgb = to_rgb_composite(tensors[:n], TRUE_COLOR_BANDS)  # (n, 3, H, W) in [0,1]
    rgb_t = torch.from_numpy(rgb).contiguous()
    grid = vutils.make_grid(rgb_t, nrow=4, padding=2, normalize=False)
    from PIL import Image as PILImage
    arr = np.nan_to_num(grid.permute(1, 2, 0).numpy(), nan=0.0, posinf=1.0, neginf=0.0)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    PILImage.fromarray(arr).save(out_dir / f"{prefix}_grid.png")
    print(f"  saved {out_dir / f'{prefix}_grid.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--generate-n", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="resume from artifacts/gan/checkpoint.pt (skips epoch recompute, continues from saved epoch)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    train_ids = splits["train"]

    gen = Generator(latent_dim=args.latent_dim).to(device)
    crit = Critic().to(device)
    print(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Critic params:    {sum(p.numel() for p in crit.parameters()):,}")

    out_dir = ROOT / "artifacts" / "gan"
    gen_ckpt = out_dir / "generator.pt"
    full_ckpt = out_dir / "checkpoint.pt"

    if args.generate_only:
        if not gen_ckpt.exists():
            print(f"No checkpoint at {gen_ckpt}; cannot generate.")
            return
        gen.load_state_dict(torch.load(gen_ckpt, map_location=device, weights_only=True))
        samples = generate_samples(gen, device, args.generate_n, args.latent_dim)
        save_samples_as_png(samples, out_dir, prefix="generated")
        print(f"Generated {args.generate_n} samples -> {out_dir}")
        return

    print(f"Loading {len(train_ids)} training patches...")
    train_images = load_train_images(marida, train_ids)
    print(f"  train images shape: {train_images.shape}")

    dataset = TensorDataset(train_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(crit.parameters(), lr=args.lr, betas=(0.0, 0.9))

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    start_epoch = 1
    if args.resume and full_ckpt.exists():
        from src.utils.checkpoint import safe_load_checkpoint
        state = safe_load_checkpoint(full_ckpt, map_location=device)
        gen.load_state_dict(state["generator"])
        crit.load_state_dict(state["critic"])
        opt_g.load_state_dict(state["optimizer_g"])
        opt_c.load_state_dict(state["optimizer_c"])
        start_epoch = state["epoch"] + 1
        print(f"Resumed from epoch {state['epoch']}; continuing to {args.epochs}.")
    else:
        log_path.write_text("")

    print(f"Training via epochs {start_epoch}..{args.epochs} (n_critic=5, lambda_gp=10)...")
    for epoch in range(start_epoch, args.epochs + 1):
        metrics = train_gan_epoch(gen, crit, loader, opt_g, opt_c, device,
                                  n_critic=5, lambda_gp=10, latent_dim=args.latent_dim)
        log_line = f'{{"epoch":{epoch},"c_loss":{metrics["c_loss"]:.4f},"g_loss":{metrics["g_loss"]:.4f}}}\n'
        with log_path.open("a") as fh:
            fh.write(log_line)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:>3d}: c_loss={metrics['c_loss']:.4f} g_loss={metrics['g_loss']:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            samples = generate_samples(gen, device, 8, args.latent_dim)
            save_samples_as_png(samples, out_dir, prefix=f"epoch{epoch:03d}")
            torch.save(gen.state_dict(), gen_ckpt)
            torch.save(
                {
                    "epoch": epoch,
                    "generator": gen.state_dict(),
                    "critic": crit.state_dict(),
                    "optimizer_g": opt_g.state_dict(),
                    "optimizer_c": opt_c.state_dict(),
                },
                full_ckpt,
            )

    print(f"\nTraining complete. Generator saved to {gen_ckpt}")
    final_samples = generate_samples(gen, device, 16, args.latent_dim)
    save_samples_as_png(final_samples, out_dir, prefix="final")


if __name__ == "__main__":
    main()