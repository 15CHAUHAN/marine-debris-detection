"""Tests for drone ingestion and physics-informed spectral adapter (Flaw 2.2)."""
import numpy as np
import pytest

from src.data.drone_ingestion import to_model_channels, N_MODEL_CHANNELS


def test_to_model_channels_physics_approx_preserves_rgb():
    """True Red, Green, Blue preserved in Sentinel-2 band slots (indices 3, 2, 1)."""
    rng = np.random.default_rng(42)
    dummy_rgb = rng.uniform(size=(3, 256, 256)).astype(np.float32)

    out = to_model_channels(dummy_rgb, mode="physics_approx")

    assert out.shape == (11, 256, 256)
    # Red -> B04 (index 3)
    assert np.allclose(out[3], dummy_rgb[0]), "Red band misplaced! Expected at index 3 (B04)"
    # Green -> B03 (index 2)
    assert np.allclose(out[2], dummy_rgb[1]), "Green band misplaced! Expected at index 2 (B03)"
    # Blue -> B02 (index 1)
    assert np.allclose(out[1], dummy_rgb[2]), "Blue band misplaced! Expected at index 1 (B02)"


def test_to_model_channels_swir_differentiated_from_visible():
    """SWIR channels (9, 10) are physically differentiated from visible bands."""
    dummy_rgb = np.ones((3, 64, 64), dtype=np.float32) * 0.5
    dummy_rgb[0] = 0.8  # Red
    dummy_rgb[1] = 0.6  # Green
    dummy_rgb[2] = 0.4  # Blue

    out = to_model_channels(dummy_rgb, mode="physics_approx")

    # SWIR channels should be near-zero (water absorption) not equal to Blue/Green/Red
    assert not np.allclose(out[9], out[1]), "SWIR channel 9 (B11) must not equal Blue!"
    assert not np.allclose(out[10], out[1]), "SWIR channel 10 (B12) must not equal Blue!"
    assert not np.allclose(out[9], out[2]), "SWIR channel 9 (B11) must not equal Green!"
    assert not np.allclose(out[10], out[2]), "SWIR channel 10 (B12) must not equal Green!"
    assert not np.allclose(out[9], out[3]), "SWIR channel 9 (B11) must not equal Red!"
    assert not np.allclose(out[10], out[3]), "SWIR channel 10 (B12) must not equal Red!"

    # SWIR should be much smaller than visible bands (water absorption)
    assert out[9].max() < 0.1, "SWIR1 should be near-zero over water"
    assert out[10].max() < 0.1, "SWIR2 should be near-zero over water"


def test_to_model_channels_all_finite_in_range():
    """All 11 channels have finite values in [0.0, 1.0]."""
    dummy_rgb = np.random.default_rng(123).uniform(size=(3, 128, 128)).astype(np.float32)

    out = to_model_channels(dummy_rgb, mode="physics_approx")

    assert np.all(np.isfinite(out)), "Output contains NaN or Inf"
    assert out.min() >= 0.0, f"Values below 0: min={out.min()}"
    assert out.max() <= 1.0, f"Values above 1: max={out.max()}"


def test_to_model_channels_coastal_aerosol_from_blue():
    """B01 Coastal Aerosol approximated from Rayleigh-scattered blue."""
    dummy_rgb = np.zeros((3, 32, 32), dtype=np.float32)
    dummy_rgb[2] = 1.0  # Blue = 1.0

    out = to_model_channels(dummy_rgb, mode="physics_approx")

    # B01 = 0.8 * Blue + 0.2 * mean
    expected_b01 = 0.8 * 1.0 + 0.2 * (1.0 / 3.0)
    assert np.allclose(out[0], expected_b01), f"B01 Coastal Aerosol mismatch: {out[0, 0, 0]} vs {expected_b01}"


def test_to_model_channels_red_edge_bands():
    """Red Edge bands (B05, B06, B07) are progressive transitions."""
    dummy_rgb = np.zeros((3, 16, 16), dtype=np.float32)
    dummy_rgb[0] = 1.0  # Red
    dummy_rgb[1] = 0.5  # Green

    out = to_model_channels(dummy_rgb, mode="physics_approx")

    # B05 = 0.7 * R + 0.3 * G = 0.7 * 1.0 + 0.3 * 0.5 = 0.85
    assert np.allclose(out[4], 0.85), f"B05 Red Edge 1 mismatch: {out[4, 0, 0]}"
    # B06 = 0.5 * R + 0.5 * G = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
    assert np.allclose(out[5], 0.75), f"B06 Red Edge 2 mismatch: {out[5, 0, 0]}"
    # B07 = 0.6 * R + 0.4 * G = 0.6 * 1.0 + 0.4 * 0.5 = 0.80
    assert np.allclose(out[6], 0.80), f"B07 Red Edge 3 mismatch: {out[6, 0, 0]}"


def test_to_model_channels_nir_approximation():
    """NIR bands (B08, B8A) approximate vegetation/plastic reflectance."""
    dummy_rgb = np.zeros((3, 16, 16), dtype=np.float32)
    dummy_rgb[0] = 1.0  # Red
    dummy_rgb[1] = 0.2  # Green

    out = to_model_channels(dummy_rgb, mode="physics_approx")

    # NIR approx = clip(1.2 * R - 0.2 * G, 0, 1) = clip(1.2 - 0.04, 0, 1) = 1.0
    assert np.allclose(out[7], 1.0), f"B08 NIR mismatch: {out[7, 0, 0]}"
    assert np.allclose(out[8], 1.0), f"B8A Narrow NIR mismatch: {out[8, 0, 0]}"

    # Over water-like (high green, low red): NIR should be low
    water_rgb = np.zeros((3, 16, 16), dtype=np.float32)
    water_rgb[0] = 0.1  # Red
    water_rgb[1] = 0.8  # Green
    out_water = to_model_channels(water_rgb, mode="physics_approx")
    assert out_water[7].max() < 0.5, "NIR should be low over water"


def test_to_model_channels_rgb3_passthrough():
    """rgb3 mode returns input unchanged (3 channels)."""
    dummy_rgb = np.random.default_rng(999).uniform(size=(3, 64, 64)).astype(np.float32)

    out = to_model_channels(dummy_rgb, mode="rgb3")

    assert out.shape == (3, 64, 64)
    assert np.allclose(out, dummy_rgb)


def test_to_model_channels_rgb_bands_legacy():
    """rgb_bands legacy mode places RGB in B2/B3/B4 slots."""
    dummy_rgb = np.random.default_rng(777).uniform(size=(3, 32, 32)).astype(np.float32)

    out = to_model_channels(dummy_rgb, mode="rgb_bands")

    assert out.shape == (11, 32, 32)
    assert np.allclose(out[1], dummy_rgb[2])  # B2 <- Blue
    assert np.allclose(out[2], dummy_rgb[1])  # B3 <- Green
    assert np.allclose(out[3], dummy_rgb[0])  # B4 <- Red


def test_to_model_channels_replicate_deprecated():
    """replicate mode (deprecated) fills all channels with mean."""
    dummy_rgb = np.random.default_rng(888).uniform(size=(3, 16, 16)).astype(np.float32)

    out = to_model_channels(dummy_rgb, mode="replicate")

    mean = dummy_rgb.mean(axis=0)
    assert np.allclose(out[0], mean)
    assert np.allclose(out[5], mean)
    assert np.allclose(out[10], mean)


def test_to_model_channels_zero_pad():
    """zero_pad mode puts RGB in first 3 channels, zeros elsewhere."""
    dummy_rgb = np.random.default_rng(555).uniform(size=(3, 16, 16)).astype(np.float32)

    out = to_model_channels(dummy_rgb, mode="zero_pad")

    assert np.allclose(out[:3], dummy_rgb)
    assert np.allclose(out[3:], 0.0)


def test_to_model_channels_invalid_mode():
    """Unknown mode raises ValueError."""
    dummy_rgb = np.zeros((3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown adapter mode"):
        to_model_channels(dummy_rgb, mode="invalid_mode")


def test_to_model_channels_invalid_shape():
    """Invalid input shape raises ValueError."""
    dummy_rgb = np.zeros((4, 16, 16), dtype=np.float32)  # 4 channels
    with pytest.raises(ValueError, match="expected \\(3, H, W\\) RGB"):
        to_model_channels(dummy_rgb, mode="physics_approx")


def test_to_model_channels_default_mode_is_physics_approx():
    """Default mode is physics_approx (not replicate)."""
    dummy_rgb = np.ones((3, 8, 8), dtype=np.float32)
    dummy_rgb[0] = 0.9  # Red
    dummy_rgb[1] = 0.5  # Green
    dummy_rgb[2] = 0.1  # Blue

    out = to_model_channels(dummy_rgb)  # No mode specified

    # Should use physics_approx, not replicate
    assert not np.allclose(out[0], out[1]), "Default mode should not be replicate"
    assert np.allclose(out[3], 0.9), "Red should be in B04 slot"
    assert np.allclose(out[2], 0.5), "Green should be in B03 slot"
    assert np.allclose(out[1], 0.1), "Blue should be in B02 slot"


# --- Verification snippet from the task ---
def test_task_verification_snippet():
    """Run the exact verification snippet from the task specification."""
    dummy_rgb = np.random.rand(3, 256, 256).astype(np.float32)
    out = to_model_channels(dummy_rgb, mode="physics_approx")
    assert out.shape == (11, 256, 256)
    assert np.allclose(out[3], dummy_rgb[0]), "Red band misplaced!"
    assert np.allclose(out[2], dummy_rgb[1]), "Green band misplaced!"
    assert np.allclose(out[1], dummy_rgb[2]), "Blue band misplaced!"
    assert not np.allclose(out[9], out[1]), "SWIR channel must not equal Blue!"
    print("Physics-informed drone spectral adapter verified successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])