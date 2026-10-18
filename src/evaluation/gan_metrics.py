"""Multi-band quality metrics for WGAN-GP synthetic MARIDA patches.

This module quantifies how closely synthetic 11-channel Sentinel-2 patches
match real training patches:

  * `compute_multispectral_fid`: classic FID on the true-color RGB composite
    [B4, B3, B2] extracted from an Inception-v3 feature space.
  * `compute_spectral_angle_mapper`: mean pairwise Spectral Angle Mapper
    across all 11 bands between real and synthetic spatial means.
  * `compute_bandwise_correlation`: inter-band correlation matrices for real
    and synthetic data plus the mean squared error between them. Sentinel-2
    bands B05, B06, B07, B8A, B11 must keep a realistic inter-band covariance.

All functions accept (N, 11, H, W) float32 arrays in [0, 1] and return a flat
JSON-safe dict (matrices embedded as nested lists).
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm

# 0-indexed band positions in the MARIDA 11-band stack (CONTEXT.md):
# B1=0, B2=1, B3=2, B4=3, B5=4, B6=5, B7=6, B8=7, B8A=8, B11=9, B12=10.
TRUE_COLOR_BANDS = (3, 2, 1)      # [B4, B3, B2]
NIR_FALSE_COLOR_BANDS = (7, 3, 2)  # [B8, B4, B3]
INCEPTION_MEAN = (0.485, 0.456, 0.406)
INCEPTION_STD = (0.229, 0.224, 0.225)
_FID_EPS = 1e-6


def _as_array(patches: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    arr = patches.numpy() if isinstance(patches, torch.Tensor) else np.asarray(patches)
    if arr.ndim != 4 or arr.shape[1] != 11:
        raise ValueError(f"{name} must be (N, 11, H, W), got {arr.shape}")
    return arr.astype(np.float32)


def to_rgb_composite(
    patches: np.ndarray | torch.Tensor,
    bands: tuple[int, int, int] = TRUE_COLOR_BANDS,
    pmin: float = 2.0,
    pmax: float = 98.0,
) -> np.ndarray:
    """Extract a 3-band composite and percentile-normalize each band to [0,1].

    Returns (N, 3, H, W) float32 in [0, 1] suitable for matplotlib/PIL.
    """
    arr = _as_array(patches, "patches")
    rgb = arr[:, list(bands), :, :]  # (N, 3, H, W)
    out = np.zeros_like(rgb, dtype=np.float32)
    for n in range(rgb.shape[0]):
        for b in range(3):
            band = rgb[n, b]
            low, high = np.percentile(band, [pmin, pmax])
            if high - low < 1e-6:
                low, high = 0.0, float(band.max()) if band.max() > 0 else 1.0
            out[n, b] = np.clip((band - low) / (high - low), 0.0, 1.0)
    return out


def _rgb_batch(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb.astype(np.float32)).permute(0, 2, 3, 1).permute(0, 3, 1, 2).contiguous()


def _inception_feature_extractor(device: torch.device) -> tuple[nn.Module, str]:
    """Load Inception-v3 for FID. Falls back to untrained weights if download fails."""
    from torchvision.models import inception_v3

    source = "inception_v3_imagenet"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = inception_v3(weights="DEFAULT")
    except Exception:  # pragma: no cover - offline environment fallback
        model = inception_v3(weights=None)
        source = "inception_v3_untrained"
        print(f"  WARNING: could not load ImageNet Inception weights; using untrained extractor ({source})")
    model.fc = nn.Identity()  # keep 2048-dim avgpool features
    model.eval().to(device)
    return model, source


@torch.no_grad()
def _extract_features(rgb: np.ndarray, model: nn.Module, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """RGB (N,3,H,W) in [0,1] -> Inception 2048-dim features (N, 2048)."""
    features: list[np.ndarray] = []
    tensor = _rgb_batch(rgb)
    for i in range(0, tensor.shape[0], batch_size):
        batch = tensor[i:i + batch_size].to(device)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - torch.tensor(INCEPTION_MEAN, device=device).view(1, 3, 1, 1)) / torch.tensor(
            INCEPTION_STD, device=device
        ).view(1, 3, 1, 1)
        features.append(model(batch).float().cpu().numpy())
    return np.concatenate(features, axis=0)


def _fid_score(mu_real: np.ndarray, sigma_real: np.ndarray, mu_fake: np.ndarray, sigma_fake: np.ndarray) -> float:
    diff = mu_real - mu_fake
    cov_prod = sigma_real @ sigma_fake
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covmean = sqrtm(cov_prod)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * _FID_EPS
        covmean = sqrtm(cov_prod + offset)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean))


def compute_multispectral_fid(
    real_patches: np.ndarray | torch.Tensor,
    fake_patches: np.ndarray | torch.Tensor,
    device: torch.device | str = "cuda",
    feature: str = "inception_v3",
    batch_size: int = 32,
) -> dict:
    """Frechet Inception Distance on the true-color [B4, B3, B2] composite.

    FID = ||mu_r - mu_f||^2 + Tr(sigma_r + sigma_f - 2*(sigma_r*sigma_f)^(1/2))
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    real_rgb = to_rgb_composite(real_patches, TRUE_COLOR_BANDS)
    fake_rgb = to_rgb_composite(fake_patches, TRUE_COLOR_BANDS)
    if feature != "inception_v3":
        raise ValueError(f"unsupported feature extractor {feature!r}")
    model, source = _inception_feature_extractor(dev)

    with torch.inference_mode():
        if model.training:
            model.eval()
        real_feats = _extract_features(real_rgb, model, dev, batch_size)
        fake_feats = _extract_features(fake_rgb, model, dev, batch_size)

    mu_real, mu_fake = real_feats.mean(0), fake_feats.mean(0)
    sigma_real = np.cov(real_feats, rowvar=False)
    sigma_fake = np.cov(fake_feats, rowvar=False)
    return {
        "rgb_fid": _fid_score(mu_real, sigma_real, mu_fake, sigma_fake),
        "fid_feature_extractor": source,
        "real_mean_feature_norm": float(np.linalg.norm(mu_real)),
        "fake_mean_feature_norm": float(np.linalg.norm(mu_fake)),
    }


def compute_spectral_angle_mapper(
    real_patches: np.ndarray | torch.Tensor,
    fake_patches: np.ndarray | torch.Tensor,
) -> dict:
    """Mean Spectral Angle Mapper across all 11 bands.

    For each of the first min(N_real, N_fake) paired samples the per-patch
    spatial-mean 11-band spectrum is compared; the average angle in degrees is
    returned. The angle between the two dataset-level mean spectra is also
    reported (`spectral_angle_mean_deg`).
    """
    real = _as_array(real_patches, "real_patches")
    fake = _as_array(fake_patches, "fake_patches")
    n = min(real.shape[0], fake.shape[0])
    if n == 0:
        return {"spectral_angle_deg": None, "spectral_angle_mean_deg": None, "samples_paired": 0}

    real_spectra = real[:n].reshape(n, 11, -1).mean(axis=2)  # (n, 11)
    fake_spectra = fake[:n].reshape(n, 11, -1).mean(axis=2)

    def _angle(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosval = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0)) if denom > 0 else 1.0
        return float(np.degrees(np.arccos(cosval)))

    angles = np.array([_angle(real_spectra[i], fake_spectra[i]) for i in range(n)])
    mean_real = real_spectra.mean(axis=0)
    mean_fake = fake_spectra.mean(axis=0)
    mean_angle = _angle(mean_real, mean_fake)
    return {
        "spectral_angle_deg": float(angles.mean()),
        "spectral_angle_std_deg": float(angles.std()),
        "spectral_angle_mean_deg": mean_angle,
        "samples_paired": n,
    }


def _mean_correlation(patches: np.ndarray) -> np.ndarray:
    """Average per-sample 11x11 Pearson correlation matrix.

    Constant bands (zero variance) yield NaN entries in np.corrcoef; they are
    replaced with 0.0 (no linear correlation) so a single degenerate synthetic
    band cannot poison the whole matrix.
    """
    n = patches.shape[0]
    flat = patches.reshape(n, 11, -1)  # (n, C, H*W)
    out = np.zeros((11, 11), dtype=np.float64)
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = np.corrcoef(flat[i])  # 11x11 (NaN where a band is constant)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        out += corr
    return out / n


def compute_bandwise_correlation(
    real_patches: np.ndarray | torch.Tensor,
    fake_patches: np.ndarray | torch.Tensor,
) -> dict:
    """Inter-band correlation matrices for real vs synthetic + MSE between them.

    The MSE over the 11x11 matrices is the primary scalar; the matrices are
    embedded as nested lists so they can be replotted in the HTML inspection
    sheet.
    """
    real = _as_array(real_patches, "real_patches")
    fake = _as_array(fake_patches, "fake_patches")
    if real.shape[0] == 0 or fake.shape[0] == 0:
        return {"band_correlation_mse": None, "real_correlation": None, "fake_correlation": None}
    real_corr = _mean_correlation(real)
    fake_corr = _mean_correlation(fake)
    return {
        "band_correlation_mse": float(np.mean((real_corr - fake_corr) ** 2)),
        "real_correlation": real_corr.tolist(),
        "fake_correlation": fake_corr.tolist(),
        "band_labels": [
            "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
        ],
    }


def compute_all_metrics(
    real_patches: np.ndarray | torch.Tensor,
    fake_patches: np.ndarray | torch.Tensor,
    device: torch.device | str = "cuda",
) -> dict:
    """Compute every metric and merge the dicts into one structured report.

    Top-level keys match the names used in artifacts/gan_verification.
    """
    results: dict = {"samples_evaluated": int(min(len(real_patches), len(fake_patches)))}
    results.update(compute_bandwise_correlation(real_patches, fake_patches))
    results.update(compute_spectral_angle_mapper(real_patches, fake_patches))
    results.update(compute_multispectral_fid(real_patches, fake_patches, device=device))
    return results