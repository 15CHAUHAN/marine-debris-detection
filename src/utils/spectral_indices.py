"""Physics-informed spectral indices for Sentinel-2 marine debris detection.

Implements the Floating Debris Index (FDI), Floating Algae Index (FAI),
and Normalized Difference Water Index (NDWI) as defined in the remote sensing
literature for plastic debris detection.

Band indices (0-indexed, 11-channel order):
  0: B01  Coastal Aerosol (60m)    - 443 nm
  1: B02  Blue (10m)               - 490 nm
  2: B03  Green (10m)              - 560 nm
  3: B04  Red (10m)                - 665 nm
  4: B05  Red Edge 1 (20m)         - 705 nm
  5: B06  Red Edge 2 (20m)         - 740 nm
  6: B07  Red Edge 3 (20m)         - 783 nm
  7: B08  NIR Broad (10m)          - 842 nm
  8: B8A  Narrow NIR (20m)         - 865 nm
  9: B11  SWIR 1 (20m)             - 1610 nm
  10: B12  SWIR 2 (20m)            - 2190 nm
"""

from __future__ import annotations

import numpy as np


def compute_fdi(image_11ch: np.ndarray) -> np.ndarray:
    """Compute Floating Debris Index (FDI) per Biermann et al. (2020).

    FDI = NIR - NIR'
    where NIR' is the baseline NIR reflectance interpolated from Red Edge 2 and SWIR 1.

    Uses bands:
      - B06 (Red Edge 2, 740 nm) -> index 5
      - B08 (NIR Broad, 842 nm)    -> index 7
      - B11 (SWIR 1, 1610 nm)      -> index 9

    Wavelengths:
      lambda_RE2 = 740 nm (B06)
      lambda_NIR = 842 nm (B08)
      lambda_SWIR1 = 1610 nm (B11)

    lambda_ratio = (lambda_NIR - lambda_RE2) / (lambda_SWIR1 - lambda_RE2)
                 = (842 - 740) / (1610 - 740) = 102 / 870 ≈ 0.11724

    NIR' = B06 + (B11 - B06) * lambda_ratio * 10
    (The *10 factor accounts for the different spatial resolutions when
     bands are resampled to 10m; in our 11-channel stack they're already aligned.)

    For our aligned 10m/20m resampled stack:
    FDI = B08 - (B06 + (B11 - B06) * 0.11724)

    Floating plastics exhibit elevated FDI (> 0.015) while sea foam,
    whitecaps, and clear water have lower or negative FDI.
    """
    # Extract bands (already aligned to same resolution)
    b06 = image_11ch[5].astype(np.float32)  # Red Edge 2 (740 nm)
    b08 = image_11ch[7].astype(np.float32)  # NIR Broad (842 nm)
    b11 = image_11ch[9].astype(np.float32)  # SWIR 1 (1610 nm)

    # Lambda ratio for linear interpolation
    lambda_ratio = (842.0 - 740.0) / (1610.0 - 740.0)  # ≈ 0.11724

    # Baseline NIR interpolated from Red Edge 2 and SWIR 1
    nir_prime = b06 + (b11 - b06) * lambda_ratio

    # Floating Debris Index
    fdi = b08 - nir_prime

    return fdi.astype(np.float32)


def compute_ndwi(image_11ch: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index (NDWI).

    NDWI = (Green - NIR) / (Green + NIR)
         = (B03 - B08) / (B03 + B08)

    Uses bands:
      - B03 (Green, 560 nm) -> index 2
      - B08 (NIR, 842 nm)   -> index 7

    Clear water has high NDWI (positive), while debris/land has low/negative NDWI.
    Sea foam and whitecaps can have intermediate values.
    """
    b03 = image_11ch[2].astype(np.float32)  # Green (560 nm)
    b08 = image_11ch[7].astype(np.float32)  # NIR (842 nm)

    denominator = b03 + b08 + 1e-6
    ndwi = (b03 - b08) / denominator

    return ndwi.astype(np.float32)


def compute_fai(image_11ch: np.ndarray) -> np.ndarray:
    """Compute Floating Algae Index (FAI) per Hu et al. (2009) / adapted for Sentinel-2.

    FAI = NIR - (Red + (SWIR1 - Red) * lambda_ratio)
    where lambda_ratio = (lambda_NIR - lambda_Red) / (lambda_SWIR1 - lambda_Red)

    Uses bands:
      - B04 (Red, 665 nm)     -> index 3
      - B08 (NIR, 842 nm)     -> index 7
      - B11 (SWIR 1, 1610 nm) -> index 9

    Separates organic vegetation/sargassum (high FAI) from synthetic plastic.
    """
    b04 = image_11ch[3].astype(np.float32)  # Red (665 nm)
    b08 = image_11ch[7].astype(np.float32)  # NIR (842 nm)
    b11 = image_11ch[9].astype(np.float32)  # SWIR 1 (1610 nm)

    lambda_ratio = (842.0 - 665.0) / (1610.0 - 665.0)  # ≈ 0.1895

    nir_prime = b04 + (b11 - b04) * lambda_ratio
    fai = b08 - nir_prime

    return fai.astype(np.float32)


def compute_spectral_gating_mask(
    image_11ch: np.ndarray,
    debris_prob: np.ndarray,
    fdi_threshold: float = 0.015,
    foam_blue_threshold: float = 0.35,
    foam_green_threshold: float = 0.35,
    foam_red_threshold: float = 0.35,
    fdi_foam_max: float = 0.005,
    fdi_boost_threshold: float = 0.03,
    boost_factor: float = 1.3,
    foam_suppression_factor: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply physics-informed spectral gating to neural network debris probabilities.

    This function implements the spectral verification gate that:
    1. Computes FDI and NDWI from raw Sentinel-2 bands
    2. Identifies sea foam/whitecap false positives (high visible, low FDI)
    3. Identifies confirmed plastic signatures (high FDI)
    4. Calibrates the CNN probability map accordingly

    Args:
        image_11ch: (11, H, W) raw or preprocessed Sentinel-2 bands
        debris_prob: (H, W) neural network debris class probability [0, 1]
        fdi_threshold: FDI threshold for potential debris (default 0.015)
        foam_blue_threshold: Blue band threshold for foam detection
        foam_green_threshold: Green band threshold for foam detection
        foam_red_threshold: Red band threshold for foam detection
        fdi_foam_max: Maximum FDI for foam classification
        fdi_boost_threshold: FDI threshold for boosting confirmed plastic
        boost_factor: Multiplier for confirmed plastic pixels
        foam_suppression_factor: Multiplier for foam/glint pixels (strongly suppress)

    Returns:
        Tuple of (calibrated_prob, fdi, ndwi, foam_mask)
    """
    # Compute spectral indices
    fdi = compute_fdi(image_11ch)
    ndwi = compute_ndwi(image_11ch)

    # Detect sea foam / whitecaps / sun glint:
    # High reflectance in visible bands (Blue, Green, Red) with low FDI
    b02 = image_11ch[1].astype(np.float32)  # Blue
    b03 = image_11ch[2].astype(np.float32)  # Green
    b04 = image_11ch[3].astype(np.float32)  # Red

    foam_mask = (
        (b02 > foam_blue_threshold) &
        (b03 > foam_green_threshold) &
        (b04 > foam_red_threshold) &
        (fdi < fdi_foam_max)
    )

    # Calibrate probabilities
    calibrated_prob = debris_prob.copy()

    # Strongly suppress foam/glint false positives
    calibrated_prob[foam_mask] *= foam_suppression_factor

    # Boost confirmed plastic signatures (high FDI)
    plastic_mask = fdi > fdi_boost_threshold
    calibrated_prob[plastic_mask] = np.clip(
        calibrated_prob[plastic_mask] * boost_factor, 0.0, 1.0
    )

    # Also suppress pixels where NDWI indicates clear water but CNN says debris
    # (Clear water should have high NDWI, debris should have low/negative NDWI)
    water_mask = ndwi > 0.3
    calibrated_prob[water_mask] *= 0.5

    # Ensure valid range
    calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)

    return calibrated_prob, fdi, ndwi, foam_mask


def apply_spectral_gating(
    debris_prob: np.ndarray,
    fdi: np.ndarray,
    ndwi: np.ndarray,
    debris_threshold: float = 0.65,
    fdi_boost_threshold: float = 0.03,
    boost_factor: float = 1.3,
    water_ndwi_threshold: float = 0.3,
    water_suppression: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply spectral gating using pre-computed FDI and NDWI.

    Alternative entry point when indices are already computed.
    """
    calibrated_prob = debris_prob.copy()

    # Boost high-FDI pixels (confirmed plastic)
    plastic_mask = fdi > fdi_boost_threshold
    calibrated_prob[plastic_mask] = np.clip(
        calibrated_prob[plastic_mask] * boost_factor, 0.0, 1.0
    )

    # Suppress clear water false positives
    water_mask = ndwi > water_ndwi_threshold
    calibrated_prob[water_mask] *= water_suppression

    calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)

    # Generate binary debris mask
    debris_binary = (calibrated_prob >= debris_threshold).astype(np.uint8)

    return calibrated_prob, debris_binary


__all__ = [
    "compute_fdi",
    "compute_ndwi",
    "compute_fai",
    "compute_spectral_gating_mask",
    "apply_spectral_gating",
]