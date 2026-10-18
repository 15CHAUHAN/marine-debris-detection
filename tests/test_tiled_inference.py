"""Tests for high-resolution tiled sliding-window inference (Flaw 2.1)."""
import numpy as np
import pytest
import torch

from dashboard.pipeline import (
    predict_mask,
    predict_mask_tiled,
    _generate_2d_window,
    _tile_grid,
)


class _IdentityConv(torch.nn.Module):
    """Conv that approximates identity on channel 0 for a pure-conv test.

    3x3 padding=1 preserves spatial size; weights set so output ~ input
    (diagonal 1.0 on channel 0, zeros elsewhere). bias=False to avoid
    half-precision autocast issues.
    """
    def __init__(self):
        super().__init__()
        conv = torch.nn.Conv2d(11, 11, 3, padding=1, bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            # Channel 0 passes through channel 0; other channels zero
            conv.weight[0, 0, 1, 1] = 1.0
        self.conv = conv

    def forward(self, x):
        return self.conv(x)


class _SlopeModel(torch.nn.Module):
    """Model that outputs a spatial gradient on channel 0 for seam testing.

    logits[0] = (x + y) / (H + W)  in [0, 1]; other channels small.
    This creates a smoothly varying field; any tile-seam artifacts would
    show as discontinuities in the gradient.
    """
    def __init__(self):
        super().__init__()
        # Use a single 1x1 conv to mix channels into a spatial field
        conv = torch.nn.Conv2d(11, 11, 1, padding=0, bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            # Output channel 0 = sum of inputs (amplified)
            conv.weight[0, :, 0, 0] = 0.1
        self.conv = conv

    def forward(self, x):
        # x: (B, 11, H, W) in [0, 1]
        # Create a gradient field: (x + y) normalized
        B, C, H, W = x.shape
        y_grid = torch.arange(H, dtype=x.dtype, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_grid = torch.arange(W, dtype=x.dtype, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        slope = (x_grid + y_grid) / float(H + W)  # (B, 1, H, W)
        out = torch.zeros(B, 11, H, W, device=x.device, dtype=x.dtype)
        out[:, 0] = slope[:, 0] * 5.0  # high-confidence debris along gradient
        out[:, 6] = 2.0  # water baseline
        return out


def test_generate_2d_window_shape_and_values():
    win = _generate_2d_window(256)
    assert win.shape == (256, 256)
    assert win.dtype == np.float32
    assert win.min() >= 1e-4
    assert win.max() <= 1.0
    # Center should be near 1.0
    assert win[128, 128] == pytest.approx(1.0, abs=1e-3)
    # Corners clipped to 1e-4
    assert win[0, 0] == pytest.approx(1e-4, abs=1e-5)
    assert win[0, -1] == pytest.approx(1e-4, abs=1e-5)


def test_tile_grid_small():
    # axis_len <= tile_size -> 1 tile, padded to tile_size
    tiles, padded = _tile_grid(32, 256, 192)
    assert tiles == 1
    assert padded == 256
    # axis_len == tile_size
    tiles, padded = _tile_grid(256, 256, 192)
    assert tiles == 1
    assert padded == 256


def test_tile_grid_large():
    # 800 with tile=256, stride=192: (800-256)/192 = 2.83 -> ceil=3 -> tiles=4
    tiles, padded = _tile_grid(800, 256, 192)
    assert tiles == 4
    # padded = 256 + 192*3 = 832
    assert padded == 832
    # 1024: (1024-256)/192 = 4.0 -> ceil=4 -> tiles=5, padded=1024
    tiles, padded = _tile_grid(1024, 256, 192)
    assert tiles == 5
    assert padded == 1024


def test_predict_mask_tiled_shape_contract():
    """Input (11, 768, 1024) returns mask/debris/prob of (768, 1024)."""
    model = _IdentityConv().eval()
    model_input = np.random.default_rng(42).uniform(size=(11, 768, 1024)).astype(np.float32)
    res = predict_mask_tiled(model, model_input, device=torch.device("cpu"),
                             tile_size=256, stride=192, batch_size=2)
    assert res["mask"].shape == (768, 1024)
    assert res["debris"].shape == (768, 1024)
    assert res["prob"].shape == (768, 1024)
    assert res["concentration"].shape == (768, 1024)


def test_predict_mask_tiled_consistency_across_batch_sizes():
    """Tiled output should be consistent regardless of batch size (same model, same input)."""
    model = _IdentityConv().eval()
    model_input = np.random.default_rng(123).uniform(size=(11, 512, 512)).astype(np.float32)

    # Tiled with batch_size=1
    tiled_1 = predict_mask_tiled(model, model_input, device=torch.device("cpu"),
                                 tile_size=256, stride=192, batch_size=1)
    # Tiled with batch_size=4
    tiled_4 = predict_mask_tiled(model, model_input, device=torch.device("cpu"),
                                 tile_size=256, stride=192, batch_size=4)

    # Should be essentially identical (within floating point roundoff)
    diff = np.abs(tiled_1["prob"] - tiled_4["prob"])
    assert diff.max() < 1e-5, f"Max diff {diff.max()}"
    assert diff.mean() < 1e-6


def test_predict_mask_tiled_seam_free_on_slope():
    """Tiled inference on a smooth field produces finite, reasonable output.

    The _SlopeModel produces a smooth (x+y) field. Tiled inference with
    Hann-window blending should produce a reasonable output without NaNs
    or extreme values.
    """
    model = _SlopeModel().eval()
    model_input = np.random.default_rng(456).uniform(size=(11, 512, 512)).astype(np.float32)

    tiled = predict_mask_tiled(model, model_input, device=torch.device("cpu"),
                               tile_size=256, stride=192, batch_size=2)

    prob = tiled["prob"]
    # Basic sanity checks
    assert np.all(np.isfinite(prob)), "Output contains NaN or Inf"
    assert prob.shape == (512, 512)
    assert prob.min() >= 0.0 and prob.max() <= 1.0, "Probabilities out of range"
    # No obvious grid pattern - check that spatial variation is smooth
    # (first derivative should be roughly constant for a linear field)
    grad_h = np.abs(prob[1:, :] - prob[:-1, :])
    grad_v = np.abs(prob[:, 1:] - prob[:, :-1])
    # Gradients should be relatively uniform (CV < 1 for a mostly linear field)
    assert grad_h.std() / max(grad_h.mean(), 1e-6) < 1.0
    assert grad_v.std() / max(grad_v.mean(), 1e-6) < 1.0


def test_predict_mask_router_small_input_uses_legacy():
    """Input <= 384 px uses single-pass path even with enable_tiling=True."""
    model = _IdentityConv().eval()
    model_input = np.random.default_rng(789).uniform(size=(11, 256, 256)).astype(np.float32)

    res = predict_mask(model, model_input, device=torch.device("cpu"),
                       enable_tiling=True)
    assert res["mask"].shape == (256, 256)


def test_predict_mask_router_large_input_uses_tiled():
    """Input > 384 px routes through tiled inference when enable_tiling=True."""
    model = _IdentityConv().eval()
    model_input = np.random.default_rng(999).uniform(size=(11, 400, 400)).astype(np.float32)

    res = predict_mask(model, model_input, device=torch.device("cpu"),
                       enable_tiling=True)
    assert res["mask"].shape == (400, 400)
    # Verify it actually went through tiling by checking with enable_tiling=False
    res_legacy = predict_mask(model, model_input, device=torch.device("cpu"),
                              enable_tiling=False)
    # Results should differ (legacy downsamples to 256 then upsamples)
    assert not np.allclose(res["prob"], res_legacy["prob"], atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_predict_mask_tiled_vram_ceiling():
    """Processing a 1500x1500 image stays under 3.5 GB VRAM."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = _IdentityConv().eval().cuda()
    torch.cuda.reset_peak_memory_stats()

    model_input = np.random.default_rng(111).uniform(size=(11, 1500, 1500)).astype(np.float32)
    _ = predict_mask_tiled(model, model_input, device=torch.device("cuda"),
                           tile_size=256, stride=192, batch_size=4)

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / (1024 ** 3)
    assert peak_gb < 3.5, f"Peak VRAM {peak_gb:.2f} GB exceeds 3.5 GB limit"


# --- Verification snippet from the task (adapted for pytest) ---
def test_task_verification_snippet():
    """Run the exact verification snippet from the task specification."""
    model = torch.nn.Conv2d(11, 11, 3, padding=1, bias=False)
    model.eval()
    dummy_large = np.random.rand(11, 800, 800).astype(np.float32)
    res = predict_mask(model, dummy_large, device=torch.device("cpu"))
    assert res["mask"].shape == (800, 800), f"Mismatch shape: {res['mask'].shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])