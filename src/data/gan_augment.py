"""WGAN-GP DCGAN for generating synthetic multispectral MARIDA patches.

Architecture:
  Generator : 100-dim noise -> 11 x 256 x 256 (Sigmoid, [0,1])
  Critic    : 11 x 256 x 256 -> scalar (no sigmoid; Wasserstein score)

Loss: Wasserstein distance + gradient penalty (lambda_gp=10).
The critic is trained 5 times per generator step (n_critic=5).

All convolutions use (kernel=4, stride=2, padding=1) for factor-2 spatial scaling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Noise vector -> (B, 11, 256, 256) float32 in [0, 1]."""

    def __init__(self, latent_dim: int = 100, out_channels: int = 11) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # latent_dim -> 512 * 4 * 4
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 512, 4, 4)
        return self.deconv(x)


class Critic(nn.Module):
    """(B, 11, 256, 256) -> (B,) Wasserstein score (no activation)."""

    def __init__(self, in_channels: int = 11) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.fc(h.view(h.size(0), -1))


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor,
                     device: torch.device, lambda_gp: float = 10.0) -> torch.Tensor:
    """Compute the WGAN-GP gradient penalty."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_gan_epoch(
    generator: Generator,
    critic: Critic,
    dataloader,
    optimizer_g: torch.optim.Adam,
    optimizer_c: torch.optim.Adam,
    device: torch.device,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    latent_dim: int = 100,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run one full WGAN-GP training epoch. Returns {"g_loss", "c_loss"}."""
    generator.train()
    critic.train()
    total_c, total_g = 0.0, 0.0
    steps_c, steps_g = 0, 0
    for batch in dataloader:
        real = batch[0].to(device)
        bs = real.size(0)
        for _ in range(n_critic):
            z = torch.randn(bs, latent_dim, device=device)
            fake = generator(z).detach()
            gp = gradient_penalty(critic, real, fake, device, lambda_gp)
            c_loss = critic(fake).mean() - critic(real).mean() + gp
            if not torch.isfinite(c_loss):
                optimizer_c.zero_grad()
                continue
            optimizer_c.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            optimizer_c.step()
            total_c += c_loss.item()
            steps_c += 1
        z = torch.randn(bs, latent_dim, device=device)
        fake = generator(z)
        g_loss = -critic(fake).mean()
        if not torch.isfinite(g_loss):
            optimizer_g.zero_grad()
            continue
        optimizer_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
        optimizer_g.step()
        total_g += g_loss.item()
        steps_g += 1
    return {"c_loss": total_c / max(steps_c, 1), "g_loss": total_g / max(steps_g, 1)}


def generate_samples(
    generator: Generator,
    device: torch.device,
    n_samples: int,
    latent_dim: int = 100,
) -> torch.Tensor:
    """Generate n_samples synthetic (11, 256, 256) patches in [0, 1]."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        fake = generator(z).clamp(0.0, 1.0)
    return fake.cpu()