"""Tests for the multi-band WGAN-GP quality metrics.

Focuses on the deterministic, network-free pieces (RGB composites, SAM,
bandwise correlation, FID score from precomputed feature moments). The
Inception-based FID pipeline is covered by the smoke test in
scripts/verify_gan_quality.py to avoid downloading weights in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.gan_metrics import (
    NIR_FALSE_COLOR_BANDS,
    TRUE_COLOR_BANDS,
    _fid_score,
    compute_bandwise_correlation,
    compute_spectral_angle_mapper,
    to_rgb_composite,
)

RNG = np.random.default_rng(0)


def _patches(n: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 11, 256, 256), dtype=np.float32)


class TestRgbComposite:
    def test_true_color_maps_b4_to_red_b3_to_green_b2_to_blue(self):
        # Build a patch where exactly one band carries a strong signal; the
        # composite channel that lights up reveals the band mapping.
        arr = np.zeros((1, 11, 8, 8), np.float32)
        arr[:, 3] = 1.0  # B4
        rgb = to_rgb_composite(arr, TRUE_COLOR_BANDS)
        assert rgb.shape == (1, 3, 8, 8)
        assert rgb[0, 0].mean() == pytest.approx(1.0)  # red from B4
        assert rgb[0, 1].mean() == pytest.approx(0.0)
        assert rgb[0, 2].mean() == pytest.approx(0.0)

    def test_percentile_normalized_to_unit_range(self):
        arr = np.zeros((1, 11, 16, 16), np.float32)
        arr[:, 3] = np.linspace(0, 100, 256).reshape(16, 16)  # B4 high dynamic range
        rgb = to_rgb_composite(arr, TRUE_COLOR_BANDS)
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0
        assert rgb[0, 0].max() > 0.99  # 98th percentile maps near 1

    def test_false_color_ir_maps_b8_to_red_b4_to_green_b3_to_blue(self):
        arr = np.zeros((1, 11, 8, 8), np.float32)
        arr[:, 7] = 1.0  # B8 (near-infrared)
        nir = to_rgb_composite(arr, NIR_FALSE_COLOR_BANDS)
        assert nir[0, 0].mean() == pytest.approx(1.0)  # red from B8
        assert nir[0, 1].mean() == pytest.approx(0.0)
        assert nir[0, 2].mean() == pytest.approx(0.0)

    def test_rejects_wrong_channel_count(self):
        with pytest.raises(ValueError):
            to_rgb_composite(np.zeros((1, 3, 8, 8), np.float32), TRUE_COLOR_BANDS)


class TestSpectralAngleMapper:
    def test_identical_patches_zero_angle(self):
        arr = _patches(3, seed=1)
        metrics = compute_spectral_angle_mapper(arr, arr)
        assert metrics["spectral_angle_deg"] == pytest.approx(0.0, abs=1e-4)
        assert metrics["spectral_angle_mean_deg"] == pytest.approx(0.0, abs=1e-4)
        assert metrics["samples_paired"] == 3

    def test_orthogonal_spectra_90_degrees(self):
        real = np.zeros((1, 11, 4, 4), np.float32)
        real[:, 0] = 1.0
        fake = np.zeros((1, 11, 4, 4), np.float32)
        fake[:, 1] = 1.0
        metrics = compute_spectral_angle_mapper(real, fake)
        assert metrics["spectral_angle_deg"] == pytest.approx(90.0, abs=1e-3)

    def test_pairs_bounded_by_smaller_set(self):
        real = _patches(5, seed=2)
        fake = _patches(3, seed=3)
        metrics = compute_spectral_angle_mapper(real, fake)
        assert metrics["samples_paired"] == 3


class TestBandwiseCorrelation:
    def test_identical_correlation_zero_mse(self):
        arr = _patches(4, seed=4)
        metrics = compute_bandwise_correlation(arr, arr)
        assert metrics["band_correlation_mse"] == pytest.approx(0.0, abs=1e-8)
        assert np.asarray(metrics["real_correlation"]).shape == (11, 11)

    def test_varied_signals_produce_positive_mse(self):
        real = _patches(4, seed=5)
        fake = np.zeros_like(real)
        fake[:, 0] = 1.0
        fake[:, 1] = -1.0
        metrics = compute_bandwise_correlation(real, fake)
        assert metrics["band_correlation_mse"] > 0.0

    def test_band_labels_match_sentinel2_order(self):
        arr = _patches(2)
        metrics = compute_bandwise_correlation(arr, arr)
        assert metrics["band_labels"] == [
            "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
        ]


class TestFidScore:
    def test_identical_distributions_zero(self):
        mu = np.zeros(64)
        sigma = np.eye(64)
        assert _fid_score(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-6)

    def test_offset_means_finite_positive(self):
        d = 64
        mu1, mu2 = np.zeros(d), np.ones(d)
        sigma1 = sigma2 = np.eye(d) * 0.5
        fid = _fid_score(mu1, sigma1, mu2, sigma2)
        assert np.isfinite(fid) and fid > 0.0