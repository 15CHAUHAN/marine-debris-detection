"""Lightweight domain checks that run BEFORE model inference.
All checks are purely heuristic — fast, zero ML inference cost.
Each returns a DomainCheckResult with severity, message, and allowed flag.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class DomainCheckResult:
    passed: bool            # True = safe to proceed
    severity: str           # "ok" | "warning" | "error"
    message: str            # Human-readable explanation for UI display
    details: dict           # Extra info (band_count, water_fraction, etc.)


def check_geotiff_bands(band_count: int) -> DomainCheckResult:
    """Warn if GeoTIFF has fewer than 11 Sentinel-2 bands."""
    if band_count == 11:
        return DomainCheckResult(True, "ok",
            "11-band Sentinel-2 GeoTIFF detected — full spectral information preserved.",
            {"band_count": band_count})
    if band_count == 3:
        return DomainCheckResult(True, "warning",
            f"⚠️ Only {band_count} bands found (RGB). The model requires 11 Sentinel-2 "
            "multispectral bands. Bands B05–B12 (Red Edge, NIR, SWIR) will be "
            "SYNTHESIZED via physics approximation. Detection accuracy for marine debris "
            "may be significantly reduced compared to real Sentinel-2 imagery.",
            {"band_count": band_count, "synthesis": "physics_approx"})
    return DomainCheckResult(True, "warning",
        f"⚠️ Unexpected band count ({band_count}). Expected 11 (Sentinel-2 MSI) or 3 (RGB). "
        "Results may be unreliable.",
        {"band_count": band_count})


def check_image_is_water_scene(rgb_uint8: np.ndarray) -> DomainCheckResult:
    """Heuristic check: does the image look like a marine/coastal scene?
    Uses blue-channel dominance and HSV saturation to distinguish
    open water, coastal zones, and beaches from indoor/urban scenes.
    A purely heuristic check — fast, no ML inference.
    """
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        return DomainCheckResult(True, "warning",
            "Could not validate image content (unexpected shape).",
            {})
    r = rgb_uint8[:, :, 0].astype(np.float32)
    g = rgb_uint8[:, :, 1].astype(np.float32)
    b = rgb_uint8[:, :, 2].astype(np.float32)
    # Fraction of pixels where blue is dominant channel (water heuristic)
    blue_dominant = float(((b > r) & (b > g)).mean())
    # Fraction of pixels in low-saturation range (water and sky are desaturated)
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_rgb > 10, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0.0)
    low_sat_fraction = float((saturation < 0.25).mean())
    is_likely_water = blue_dominant > 0.25 or low_sat_fraction > 0.40
    if is_likely_water:
        return DomainCheckResult(True, "ok",
            f"Image appears to contain water/coastal scene "
            f"(blue-dominant pixels: {blue_dominant:.1%}, "
            f"low-saturation: {low_sat_fraction:.1%}).",
            {"blue_dominant": round(blue_dominant, 3),
             "low_sat_fraction": round(low_sat_fraction, 3)})
    else:
        return DomainCheckResult(True, "warning",
            "⚠️ This image does not appear to be a marine or coastal scene "
            f"(blue-dominant pixels: {blue_dominant:.1%}). "
            "Marine debris models are trained on Sentinel-2 ocean scenes. "
            "Detection results on terrestrial, indoor, or urban imagery will be "
            "UNRELIABLE and should not be interpreted as environmental data. "
            "For litter detection in non-marine photos, the TACO YOLOv8 detector "
            "in the 'Drone image' source mode is more appropriate.",
            {"blue_dominant": round(blue_dominant, 3),
             "low_sat_fraction": round(low_sat_fraction, 3),
             "likely_marine": False})


def check_rgb_photo_domain(rgb_uint8: np.ndarray, taco_available: bool) -> DomainCheckResult:
    """Validate an RGB drone/smartphone upload for the TACO detection path."""
    water_check = check_image_is_water_scene(rgb_uint8)
    if not taco_available:
        return DomainCheckResult(True, "warning",
            "⚠️ TACO litter detector weights not found. Falling back to satellite "
            "segmentation backbone on RGB image — this will produce unreliable results "
            "for non-Sentinel-2 imagery. Train TACO first: "
            "`python scripts/train_taco.py`",
            {"taco_available": False})
    if not water_check.passed or water_check.severity == "warning":
        return DomainCheckResult(True, "warning",
            water_check.message +
            "\nTACO detector will still run but may have reduced accuracy on "
            "non-standard angles or lighting conditions.",
            water_check.details)
    return DomainCheckResult(True, "ok",
        "✅ TACO YOLOv8 litter detector is active and image appears to be a "
        "coastal/outdoor scene.",
        {"taco_available": True})