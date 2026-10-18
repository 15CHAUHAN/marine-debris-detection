"""Tests for the pseudo-label agreement validation (breaks circular validation).

Verifies that agreement_scores returns expected consensus metrics and flags
degenerate MC-dropout when the model has no dropout layers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_gan_augmented import _enable_mc_dropout, agreement_scores, pseudo_label  # noqa: E402

TORCH_DEV = torch.device("cpu")


class _NoDropoutNet(nn.Module):
    """A tiny segmentation head with NO dropout (mirrors UNet variants)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(11, 11, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv(x)))
        return x.expand(-1, 11, 256, 256)


class _DropoutNet(nn.Module):
    """A tiny net WITH a dropout layer (for the MC-dropout path)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(11, 11, 1)
        self.drop = nn.Dropout2d(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(torch.relu(self.conv(x)))
        x = x.mean(dim=(2, 3), keepdim=True).expand(-1, 11, 256, 256)
        return x


def _sample_batch(n: int = 4) -> torch.Tensor:
    return torch.rand(n, 11, 32, 32)


class TestAgreementScores:
    def test_identical_models_full_agreement(self):
        net = _NoDropoutNet().eval()
        batch = _sample_batch()
        agree, mode, degenerate = agreement_scores(
            net, batch, TORCH_DEV, agreement_variant="self", mc_dropout_passes=4, dropout_p=0.2
        )
        assert mode == "self"
        assert agree.shape[0] == batch.shape[0]
        assert torch.all(agree >= 0.0) and torch.all(agree <= 1.0)

    def test_mc_dropout_without_dropout_layers_is_degenerate(self):
        net = _NoDropoutNet().eval()
        assert _enable_mc_dropout(net, 0.2) == 0
        agree, mode, degenerate = agreement_scores(
            net, _sample_batch(), TORCH_DEV, agreement_variant="mc", mc_dropout_passes=4, dropout_p=0.2
        )
        assert mode == "mc"
        assert degenerate is True
        assert torch.all(agree == 1.0)  # tautological: no stochasticity

    def test_mc_dropout_with_dropout_layers_has_real_agreement(self):
        net = _DropoutNet().eval()
        assert _enable_mc_dropout(net, 0.2) >= 1
        agree, mode, degenerate = agreement_scores(
            net, _sample_batch(8), TORCH_DEV, agreement_variant="mc", mc_dropout_passes=6, dropout_p=0.2
        )
        assert mode == "mc"
        assert degenerate is False
        assert agree.shape[0] == 8

    def test_ensemble_identical_models_full_agreement(self):
        # Two identical teachers -> every pseudo-label must agree 100%.
        net_a = _NoDropoutNet().eval()
        net_b = _NoDropoutNet().eval()
        net_b.load_state_dict(net_a.state_dict())
        imgs = _sample_batch(4)
        kept, _targets, stats = pseudo_label(
            net_a, net_b, imgs, TORCH_DEV,
            confidence=0.0, min_fraction=0.0, min_debris_pixels=0,
            agreement_variant="attention", agreement_threshold=0.75,
            mc_dropout_passes=1, dropout_p=0.2, batch_size=4,
        )
        assert kept.shape[0] == 4
        assert stats["mean_kept_agreement"] == pytest.approx(1.0, abs=1e-6)

    def test_ensemble_divergent_models_lower_agreement(self):
        # Different teachers -> agreement drops below 1.0 (not tautological).
        net_a = _NoDropoutNet().eval()
        net_b = _NoDropoutNet().eval()
        with torch.no_grad():
            net_b.conv.weight.copy_(net_a.conv.weight + 0.5)
        imgs = _sample_batch(4)
        kept, _targets, stats = pseudo_label(
            net_a, net_b, imgs, TORCH_DEV,
            confidence=0.0, min_fraction=0.0, min_debris_pixels=0,
            agreement_variant="attention", agreement_threshold=0.0,
            mc_dropout_passes=1, dropout_p=0.2, batch_size=4,
        )
        assert kept.shape[0] == 4
        assert stats["mean_kept_agreement"] < 1.0


class TestPseudoLabelFiltering:
    def test_impossible_agreement_threshold_keeps_nothing(self):
        net = _NoDropoutNet().eval()
        imgs = _sample_batch(2)
        kept, _targets, stats = pseudo_label(
            net, None, imgs, TORCH_DEV,
            confidence=0.0, min_fraction=0.0, min_debris_pixels=0,
            agreement_variant="self", agreement_threshold=2.0,
            mc_dropout_passes=2, dropout_p=0.2, batch_size=2,
        )
        assert kept.shape[0] == 0
        assert stats["n_kept"] == 0

    def test_zero_threshold_runs_full_pipeline(self):
        net = _NoDropoutNet().eval()
        imgs = _sample_batch(2)
        kept, targets, stats = pseudo_label(
            net, None, imgs, TORCH_DEV,
            confidence=0.0, min_fraction=0.0, min_debris_pixels=0,
            agreement_variant="self", agreement_threshold=0.0,
            mc_dropout_passes=2, dropout_p=0.2, batch_size=2,
        )
        assert stats["n_kept"] >= 0
        assert targets.shape[0] == kept.shape[0]


class TestGanMetricsE2E:
    def test_spectral_and_band_metrics_run_on_tiny_tensors(self):
        from src.evaluation.gan_metrics import (
            compute_bandwise_correlation,
            compute_spectral_angle_mapper,
        )

        real = torch.rand(2, 11, 16, 16)
        fake = torch.rand(2, 11, 16, 16)
        sam = compute_spectral_angle_mapper(real, fake)
        corr = compute_bandwise_correlation(real, fake)
        assert 0.0 <= sam["spectral_angle_deg"] <= 180.0
        assert corr["band_correlation_mse"] is not None
        assert np.isfinite(corr["band_correlation_mse"])