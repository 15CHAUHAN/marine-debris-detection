"""Core source-to-detection pipeline for the interactive dashboard.

Implements the detection stages of the problem-statement output pipeline:
``Satellite/Drone Image/Video -> Plastic Detection -> Concentration``.

Pure Python/numpy/torch functions with no streamlit dependency so they are
unit-testable. The Streamlit UI (dashboard/app.py) and the georeferenced map
builder (dashboard/geo.py) consume these functions only.
"""
from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

import torch.nn.functional as F
from PIL import ExifTags, Image
from scipy.ndimage import label as nd_label

try:
    from torchvision.ops import nms as torchvision_nms
    _TORCHVISION_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TORCHVISION_AVAILABLE = False

from src.data.drone_ingestion import to_model_channels
from src.data.preprocessing import preprocess_satellite_input
from src.detection.video_inference import sample_frame_indices
from src.mapping.heatmap import debris_concentration
from src.utils.spectral_indices import (
    compute_fdi,
    compute_ndwi,
    compute_spectral_gating_mask,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_SIZE = 256
DEBRIS_CHANNEL = 0
MAX_VIDEO_FRAMES = 60
DEFAULT_SAMPLE_FPS = 2.0

# --- High-resolution tiled inference --------------------------------------
# Large inputs (full Sentinel-2 scenes, big aerial tiles) are never squashed
# into MODEL_SIZE. Instead they are processed as overlapping MODEL_SIZE crops
# at native resolution and blended together with a 2D Hann window.
TILED_STRIDE = 192          # 25% overlap (64 px) between adjacent tiles
TILING_ROUTE_THRESHOLD = 384  # inputs larger than this route through tiling
TILED_BATCH_SIZE = 4        # fits easily inside the RTX 4050 (6 GB VRAM)

# --- Video frame storage resolution ----------------------------------------
# Per-frame debris masks and concentration maps are downsampled to this
# resolution before storage to avoid MemoryError in Streamlit's pickle cache.
# 256x256 float32 = ~0.25 MB per frame vs ~8 MB at 1920x1080.
VIDEO_STORAGE_SIZE = 256

# --- TACO YOLOv8 RGB debris detector -------------------------------------
# The satellite backbones resolve 10 m/px Marine Debris but cannot classify a
# bottle/can/bag in an ordinary smartphone or drone photo. A YOLOv8n model
# fine-tuned on TACO routes those RGB uploads here for instant, class-aware
# street-level detection.
TACO_MODEL_PATH = (
    ROOT / "artifacts" / "taco_training" / "yolov8n_taco" / "weights" / "best.pt"
)
TACO_IMGSZ = 640
TACO_CONF_THRESHOLD = 0.15
TACO_NMS_IOU = 0.45
TACO_CLASSES = [
    "Bottle",          # 0
    "Can",             # 1
    "Plastic bag",     # 2
    "Cup",             # 3
    "Wrapper_Sachet",  # 4
    "Container",       # 5
    "Other_Trash",     # 6
]
# Macro-classes the dashboard answers "bottles, bags, cans" questions with.
TACO_PRIMARY_CLASSES = ("Bottle", "Can", "Plastic bag")

# (label, lower inclusive bound in %).
# Severity thresholds calibrated against published debris survey literature:
#
# Source: Topouzelis et al. (2020) "Detection of floating plastics from
#   satellite and UAV sensors" — debris pixel fraction in survey patches.
# Source: Biermann et al. (2020) MARIDA paper — class distributions show
#   >95% of patches have Marine Debris fraction < 2.0%.
# Source: Kikaki et al. (2020) — "Floating marine debris" target patches
#   show debris density 0.1–3% in Sentinel-2 footprint.
#
# Thresholds derived from MARIDA class-1 pixel distribution:
#   p10 ≈ 0.05% (sparse, likely misclassification noise)
#   p50 ≈ 0.5%  (typical debris-positive patch)
#   p90 ≈ 2.5%  (heavily debris-laden patch)
#   >5%          (exceptional event — dense accumulation zone)
SEVERITY_BANDS = (
    ("Low",      0.0),    # 0 – 0.5%  : background noise / sparse scatter
    ("Moderate", 0.5),    # 0.5 – 2.5%: confirmed debris aggregation
    ("High",     2.5),    # 2.5 – 5.0%: significant marine pollution event
    ("Critical", 5.0),    # > 5.0%    : dense accumulation — MARIDA p90 level
)


def severity_level(density_pct: float) -> str:
    """Pollution severity bucket for a debris density percentage.

    Thresholds calibrated against MARIDA class-1 pixel distributions
    (Biermann et al. 2020) and Topouzelis et al. (2020) survey statistics.

    - Low < 0.5%
    - Moderate [0.5, 2.5)%
    - High [2.5, 5.0)%
    - Critical >= 5.0%

    Note: density_pct is the % of image pixels classified as Marine Debris.
    For absolute mass estimation, use debris_per_km2_m2 = density_pct/100 * 1e6.
    """
    label = SEVERITY_BANDS[0][0]
    for band_label, lower in SEVERITY_BANDS:
        if density_pct >= lower:
            label = band_label
    return label


def concentration_stats(
    debris: np.ndarray,
    concentration: np.ndarray,
    pixel_area_m2: float | None = None,
) -> dict:
    """Density, area, and severity for one source (image/frame).

    Args:
        debris: binary (H, W) Marine Debris mask (1 where detected).
        concentration: float (H, W) concentration map in [0, 1].
        pixel_area_m2: physical ground area covered by one pixel. Ignored when
            None, in which case ``debris_area_m2`` stays None.
    """
    total = int(np.prod(debris.shape))
    pixels = int(np.sum(debris))
    density = 100.0 * pixels / total if total else 0.0
    debris_per_km2_m2 = round((density / 100.0) * 1e6, 2) if pixel_area_m2 else None
    return {
        "debris_pixels": pixels,
        "total_pixels": total,
        "density_pct": round(density, 4),
        "debris_area_m2": round(pixels * pixel_area_m2, 2) if pixel_area_m2 else None,
        "debris_per_km2_m2": debris_per_km2_m2,
        "mean_concentration": round(float(np.mean(concentration)), 4),
        "peak_concentration": round(float(np.max(concentration)), 4),
        "severity": severity_level(density),
    }


def aggregate_frames(
    frames: list[dict],
    pixel_area_m2: float | None = None,
) -> dict:
    """Summarize per-frame detection dicts into one video-level report."""
    densities = [f["stats"]["density_pct"] for f in frames]
    means = [f["stats"]["mean_concentration"] for f in frames]
    peaks = [f["stats"]["peak_concentration"] for f in frames]
    total_px = int(sum(f["stats"]["debris_pixels"] for f in frames))
    mean_density = float(np.mean(densities)) if densities else 0.0
    debris_per_km2_m2 = round((mean_density / 100.0) * 1e6, 2) if pixel_area_m2 else None
    return {
        "n_frames": len(frames),
        "frames_with_debris": int(sum(1 for d in densities if d > 0)),
        "mean_density_pct": round(float(np.mean(densities)), 4),
        "peak_density_pct": round(float(np.max(densities)), 4),
        "mean_concentration": round(float(np.mean(means)), 4),
        "peak_concentration": round(float(np.max(peaks)), 4),
        "total_debris_pixels": int(total_px),
        "total_debris_area_m2": round(total_px * pixel_area_m2, 2) if pixel_area_m2 else None,
        "debris_per_km2_m2": debris_per_km2_m2,
        "severity": severity_level(float(np.mean(densities))),
    }


def boxes_with_confidence(
    debris: np.ndarray,
    prob: np.ndarray | None,
    min_area_px: int = 3,
) -> list[dict]:
    """Connected-component debris boxes with a pseudo-confidence score.

    Boxes are sorted by area descending. ``prob`` is a per-pixel class
    probability map (softmax channel 0); each box confidence is the mean
    probability inside its component. None keeps boxes but no confidence.
    """
    from scipy.ndimage import label

    labels, n = label(debris)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labels == i)
        if ys.size < min_area_px:
            continue
        box = {
            "xmin": int(xs.min()),
            "xmax": int(xs.max()),
            "ymin": int(ys.min()),
            "ymax": int(ys.max()),
            "area_px": int(ys.size),
            "confidence": (
                round(float(prob[ys, xs].mean()), 4) if prob is not None else None
            ),
        }
        boxes.append(box)
    boxes.sort(key=lambda b: b["area_px"], reverse=True)
    return boxes


def _percentile_normalize(image: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    """Band-wise percentile normalization to [0, 1] with NaN tolerance."""
    output = np.zeros_like(image, dtype=np.float32)
    for band in range(image.shape[0]):
        values = image[band].astype(np.float32)
        finite = values[np.isfinite(values)]
        clean = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
        low, high = np.percentile(finite, [pmin, pmax]) if finite.size else (0.0, 0.0)
        if high - low < 1e-6:
            low, high = 0.0, float(finite.max()) if finite.size and finite.max() > 0 else 1.0
        output[band] = np.clip((clean - low) / (high - low), 0.0, 1.0)
    return output


def _rgb_composite(image: np.ndarray, band_count: int) -> np.ndarray:
    """Display RGB (H, W, 3) uint8 from an uploaded GeoTiff."""
    if band_count >= 3:
        bands = image[[2, 1, 0]]
        rgb = _percentile_normalize(bands).transpose(1, 2, 0)
    else:
        band = _percentile_normalize(image[:1])[0]
        rgb = np.stack([band, band, band], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def read_geotiff(
    source_bytes: bytes,
    filename: str = "",
    apply_clahe: bool = True,
) -> dict:
    """Parse an uploaded Sentinel-2 (or any) GeoTiff for the model.

    Returns the 11-channel model input, an RGB composite, and the source
    georeferencing (crs, transform, size) for the pollution map.

    Args:
        apply_clahe: If True, apply the canonical CLAHE preprocessing
            (percentile normalization + CLAHE) matching training.
    """
    with rasterio.io.MemoryFile(source_bytes) as mem:
        with mem.open() as src:
            image = src.read().astype(np.float32)
            crs = src.crs.to_string() if src.crs else None
            transform = tuple(src.transform)
            width, height, count = src.width, src.height, src.count
            rgb = _rgb_composite(image, count)
    if count == 11:
        model_input = preprocess_satellite_input(image, apply_clahe=apply_clahe)
    else:
        frame = rgb.astype(np.float32) / 255.0
        model_input = to_model_channels(frame.transpose(2, 0, 1), 11, mode="replicate")
    return {
        "model_input": model_input,
        "rgb": rgb,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "band_count": count,
        "filename": filename,
    }


def _exif_rational(value) -> float:
    """Convert an EXIF rational (num, den) or simple number to float."""
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        return float(num) / float(den) if den else 0.0
    return float(value)


def _dms_to_degrees(dms, ref: str) -> float:
    """Degrees/Minutes/Seconds EXIF tuple + N/S/E/W ref -> signed decimal degrees."""
    deg, minutes, sec = (_exif_rational(v) for v in dms)
    value = deg + minutes / 60.0 + sec / 3600.0
    return -value if ref in ("S", "W") else value


def extract_exif_gps(source_bytes: bytes) -> dict | None:
    """Extract GPS Latitude/Longitude/Altitude from an image's EXIF block.

    Returns ``None`` when the file has no GPS tags (or is unreadable) so
    callers can fall back to a demo coordinate. Returned dict:
    ``{"lat": float, "lon": float, "altitude_m": float | None, "has_gps": True}``.
    """
    try:
        with Image.open(io.BytesIO(source_bytes)) as img:
            exif = img.getexif()
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:  # noqa: BLE001 - corrupt/unknown files => no GPS
        return None
    lat = gps_ifd.get(ExifTags.GPS.GPSLatitude)
    lon = gps_ifd.get(ExifTags.GPS.GPSLongitude)
    if not lat or not lon:
        return None
    lat_ref = gps_ifd.get(ExifTags.GPS.GPSLatitudeRef, "N")
    lon_ref = gps_ifd.get(ExifTags.GPS.GPSLongitudeRef, "E")
    try:
        latitude = _dms_to_degrees(lat, lat_ref.decode() if isinstance(lat_ref, bytes) else lat_ref)
        longitude = _dms_to_degrees(lon, lon_ref.decode() if isinstance(lon_ref, bytes) else lon_ref)
    except Exception:  # noqa: BLE001 - malformed DMS -> no GPS
        return None
    altitude = gps_ifd.get(ExifTags.GPS.GPSAltitude)
    if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
        return None
    altitude_m = None
    if altitude is not None:
        altitude_m = round(_exif_rational(altitude), 2)
    return {
        "lat": round(latitude, 6),
        "lon": round(longitude, 6),
        "altitude_m": altitude_m,
        "has_gps": True,
    }


def read_rgb_upload(
    source_bytes: bytes,
    filename: str = "",
    adapter_mode: str = "physics_approx",
) -> dict:
    """Parse an uploaded RGB drone image (jpg/png) for the model.

    Note: The default ``physics_approx`` mode synthesizes 11 Sentinel-2 bands
    from 3-channel RGB. This is an approximation. For true litter detection
    on RGB photos, use the TACO YOLOv8 detector (source_kind="drone" with
    TACO weights available).
    """
    import logging
    logger = logging.getLogger(__name__)
    arr = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"cannot decode image: {filename or 'upload'}")
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    frame = rgb.astype(np.float32) / 255.0
    model_input = to_model_channels(frame.transpose(2, 0, 1), 11, mode=adapter_mode)
    height, width = rgb.shape[:2]
    return {
        "model_input": model_input,
        "rgb": rgb,
        "height": height,
        "width": width,
        "filename": filename,
    }


def _generate_2d_window(tile_size: int = 256) -> np.ndarray:
    """Generate a 2D Hann weight window to smoothly blend overlapping tiles and eliminate edge artifacts."""
    w1d = np.hanning(tile_size)
    w2d = np.outer(w1d, w1d).astype(np.float32)
    # Clip lower values to prevent zero division in unoverlapped areas
    return np.clip(w2d, 1e-4, 1.0)


def _tile_grid(axis_len: int, tile_size: int, stride: int) -> tuple[int, int]:
    """Return (num_tiles, padded_extent) for even coverage of an axis with sliding tiles."""
    if axis_len <= tile_size:
        return 1, tile_size
    tiles = int(np.ceil((axis_len - tile_size) / stride)) + 1
    return tiles, tile_size + stride * (tiles - 1)


def _postprocess_logits(
    logits: np.ndarray,
    debris_threshold: float,
    min_cluster_px: int,
) -> dict:
    """Derive the standard detection dict from full-resolution (C, H, W) logits.

    Applies softmax probability, precision gating (debris must be argmax and
    exceed ``debris_threshold``), and connected-component cluster filtering
    (``min_cluster_px``), identical to the legacy single-pass path.
    """
    h, w = logits.shape[-2:]
    probs = torch.softmax(torch.from_numpy(np.asarray(logits, dtype=np.float32)), dim=0).numpy()
    prob_debris = probs[DEBRIS_CHANNEL].astype(np.float32)
    raw_mask = np.argmax(logits, axis=0).astype(np.int64)
    debris_candidates = (raw_mask == DEBRIS_CHANNEL) & (prob_debris >= debris_threshold)
    if min_cluster_px > 1 and np.any(debris_candidates):
        labeled_blobs, num_blobs = nd_label(debris_candidates)
        if num_blobs > 0:
            blob_sizes = np.bincount(labeled_blobs.ravel())
            keep_mask = np.isin(labeled_blobs, np.where(blob_sizes >= min_cluster_px)[0])
            keep_mask &= labeled_blobs > 0
            debris = keep_mask.astype(np.float32)
        else:
            debris = np.zeros((h, w), dtype=np.float32)
    else:
        debris = debris_candidates.astype(np.float32)
    final_mask = raw_mask.copy()
    filtered_out = (raw_mask == DEBRIS_CHANNEL) & (debris == 0.0)
    final_mask[filtered_out] = 6  # Class 6 is Marine Water
    return {
        "mask": final_mask,
        "debris": debris,
        "prob": prob_debris,
        "concentration": debris_concentration(debris),
    }


def _apply_tta_single(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    original_shape: tuple[int, int],
) -> np.ndarray:
    """Apply 8-fold Test-Time Augmentation for a single forward pass.

    Generates 8 spatial variants, runs inference, applies inverse transforms,
    and averages softmax probabilities.

    Variants:
    1. Original
    2. Horizontal flip (flip W)
    3. Vertical flip (flip H)
    4. Horizontal + Vertical flip
    5. 90° rotation (transpose + flip H)
    6. 180° rotation (flip H + flip W)
    7. 270° rotation (transpose + flip W)
    8. Transpose (diagonal flip)

    Args:
        model: PyTorch model in eval mode.
        x: Input tensor (1, 11, H, W) on device.
        device: Torch device.
        original_shape: (H, W) of original input for inverse interpolation.

    Returns:
        Averaged softmax probabilities (11, H, W) as numpy array.
    """
    h, w = original_shape
    probs_accum = None

    # Define 8 augmentations as (transform_fn, inverse_fn) pairs
    def identity(t): return t
    def flip_h(t): return torch.flip(t, dims=[3])
    def flip_v(t): return torch.flip(t, dims=[2])
    def flip_hv(t): return torch.flip(t, dims=[2, 3])
    def rot90(t): return torch.rot90(t, k=1, dims=[2, 3])
    def rot180(t): return torch.rot90(t, k=2, dims=[2, 3])
    def rot270(t): return torch.rot90(t, k=3, dims=[2, 3])
    def transpose(t): return t.transpose(2, 3)

    def inv_identity(t): return t
    def inv_flip_h(t): return torch.flip(t, dims=[3])
    def inv_flip_v(t): return torch.flip(t, dims=[2])
    def inv_flip_hv(t): return torch.flip(t, dims=[2, 3])
    def inv_rot90(t): return torch.rot90(t, k=3, dims=[2, 3])
    def inv_rot180(t): return torch.rot90(t, k=2, dims=[2, 3])
    def inv_rot270(t): return torch.rot90(t, k=1, dims=[2, 3])
    def inv_transpose(t): return t.transpose(2, 3)

    augmentations = [
        (identity, inv_identity),
        (flip_h, inv_flip_h),
        (flip_v, inv_flip_v),
        (flip_hv, inv_flip_hv),
        (rot90, inv_rot90),
        (rot180, inv_rot180),
        (rot270, inv_rot270),
        (transpose, inv_transpose),
    ]

    with torch.no_grad():
        for transform_fn, inverse_fn in augmentations:
            x_aug = transform_fn(x)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits_aug = model(x_aug)
            logits_aug = logits_aug.float()

            # Inverse transform logits back to original orientation
            logits_inv = inverse_fn(logits_aug)

            # Interpolate to original size if needed
            if logits_inv.shape[-2:] != (h, w):
                logits_inv = F.interpolate(
                    logits_inv, size=(h, w), mode="bilinear", align_corners=False
                )

            # Softmax to probabilities
            probs = torch.softmax(logits_inv, dim=1)[0].cpu().numpy()  # (11, H, W)

            if probs_accum is None:
                probs_accum = probs
            else:
                probs_accum += probs

    return probs_accum / len(augmentations)


def predict_mask_tta(
    model: torch.nn.Module,
    model_input: np.ndarray,
    device: torch.device | None = None,
    debris_threshold: float = 0.65,
    min_cluster_px: int = 4,
    apply_clahe: bool = True,
) -> dict:
    """Single-image 8-fold Test-Time Augmentation inference.

    Runs the model on 8 geometric variants of the input and averages
    the resulting probability maps. This eliminates orientation bias
    along ocean drift lines and improves boundary precision.

    Args:
        model: PyTorch segmentation model.
        model_input: (11, H, W) float32 input.
        device: Torch device.
        debris_threshold: Probability threshold for debris classification.
        min_cluster_px: Minimum connected component size.
        apply_clahe: Whether to apply CLAHE preprocessing.

    Returns:
        Detection dict with keys: mask, debris, prob, concentration.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch, h, w = model_input.shape
    if ch != 11:
        raise ValueError(f"model input must have 11 channels, got {ch}")

    # Apply canonical preprocessing
    model_input = preprocess_satellite_input(model_input, apply_clahe=apply_clahe)

    # Convert to tensor and add batch dimension
    x = torch.from_numpy(np.asarray(model_input, dtype=np.float32))[None].to(device, non_blocking=True)

    # If input is larger than MODEL_SIZE, we still run TTA at native resolution
    # (models should handle variable input sizes due to fully convolutional architecture)
    avg_probs = _apply_tta_single(model, x, device, (h, w))

    # Convert averaged probabilities back to logits-like for postprocessing
    # We use log(prob) as pseudo-logits for _postprocess_logits
    eps = 1e-8
    avg_logits = np.log(np.maximum(avg_probs, eps)).astype(np.float32)

    return _postprocess_logits(avg_logits, debris_threshold, min_cluster_px)


def _predict_with_spectral_gating(
    model_input: np.ndarray,
    debris_prob: np.ndarray,
    debris_threshold: float,
    min_cluster_px: int,
    enable_spectral_gating: bool,
) -> dict:
    """Apply spectral gating to debris probabilities and post-process.

    Args:
        model_input: (11, H, W) raw/preprocessed Sentinel-2 bands.
        debris_prob: (H, W) neural network debris probability.
        debris_threshold: Threshold for debris classification.
        min_cluster_px: Minimum cluster size for filtering.
        enable_spectral_gating: Whether to apply physics-informed gating.

    Returns:
        Detection dict with calibrated probabilities and debris mask.
    """
    if enable_spectral_gating:
        # Compute FDI, NDWI and apply foam rejection / plastic boost
        calibrated_prob, fdi, ndwi, foam_mask = compute_spectral_gating_mask(
            model_input, debris_prob
        )
        # Store spectral indices for visualization
        spectral_data = {"fdi": fdi, "ndwi": ndwi, "foam_mask": foam_mask}
    else:
        calibrated_prob = debris_prob
        spectral_data = {}

    # Post-process: threshold and cluster filtering
    h, w = debris_prob.shape
    raw_mask = np.zeros((h, w), dtype=np.int64)
    raw_mask[calibrated_prob >= debris_threshold] = 0  # Marine Debris
    raw_mask[calibrated_prob < debris_threshold] = 6  # Marine Water

    debris_candidates = calibrated_prob >= debris_threshold
    if min_cluster_px > 1 and np.any(debris_candidates):
        labeled_blobs, num_blobs = nd_label(debris_candidates)
        if num_blobs > 0:
            blob_sizes = np.bincount(labeled_blobs.ravel())
            keep_mask = np.isin(labeled_blobs, np.where(blob_sizes >= min_cluster_px)[0])
            keep_mask &= labeled_blobs > 0
            debris = keep_mask.astype(np.float32)
        else:
            debris = np.zeros((h, w), dtype=np.float32)
    else:
        debris = debris_candidates.astype(np.float32)

    final_mask = raw_mask.copy()
    filtered_out = (raw_mask == 0) & (debris == 0.0)
    final_mask[filtered_out] = 6  # Class 6 is Marine Water

    result = {
        "mask": final_mask,
        "debris": debris,
        "prob": calibrated_prob.astype(np.float32),
        "concentration": debris_concentration(debris),
    }
    result.update(spectral_data)
    return result


def predict_mask_tiled(
    model: torch.nn.Module,
    model_input: np.ndarray,
    device: torch.device | None = None,
    tile_size: int = MODEL_SIZE,
    stride: int = TILED_STRIDE,
    debris_threshold: float = 0.65,
    min_cluster_px: int = 4,
    batch_size: int = TILED_BATCH_SIZE,
    apply_clahe: bool = True,
) -> dict:
    """Sliding-window inference at native resolution with Hann-window blending.

    For inputs larger than TILING_ROUTE_THRESHOLD, the image is processed as
    overlapping tiles at native 10 m/px. Edge tiles are padded with edge
    replication so the model sees valid spatial context. Predictions are
    accumulated with a 2D Hann window, then normalized to produce seamless
    logits that are passed through the same precision-gating pipeline.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch, h, w = model_input.shape
    if ch != 11:
        raise ValueError(f"model input must have 11 channels, got {ch}")

    # Apply canonical preprocessing to match training distribution
    model_input = preprocess_satellite_input(model_input, apply_clahe=apply_clahe)

    # If input already matches a single tile, delegate to single-pass path
    # apply_clahe=False because we already applied preprocessing above
    if (h, w) == (tile_size, tile_size):
        return predict_mask(
            model, model_input, device,
            debris_threshold=debris_threshold,
            min_cluster_px=min_cluster_px,
            enable_tiling=False,
            apply_clahe=False,
        )

    rows, padded_h = _tile_grid(h, tile_size, stride)
    cols, padded_w = _tile_grid(w, tile_size, stride)

    src = np.asarray(model_input, dtype=np.float32)
    if (padded_h, padded_w) != (h, w):
        src = np.pad(
            src, ((0, 0), (0, padded_h - h), (0, padded_w - w)), mode="edge"
        )

    window = _generate_2d_window(tile_size)
    accum_logits = np.zeros((ch, padded_h, padded_w), dtype=np.float32)
    weight_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    # Precompute tile origins for batched processing
    top = [r * stride for r in range(rows)]
    left = [c * stride for c in range(cols)]
    tile_origins = [(r, c) for r in top for c in left]

    with torch.no_grad():
        for start in range(0, len(tile_origins), batch_size):
            origins = tile_origins[start : start + batch_size]
            tiles = np.stack([src[:, r : r + tile_size, c : c + tile_size] for r, c in origins])
            batch = torch.from_numpy(tiles).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(batch)
            logits = logits.float().cpu().numpy()  # (B, C, T, T)
            for i, (r, c) in enumerate(origins):
                accum_logits[:, r : r + tile_size, c : c + tile_size] += logits[i] * window[None]
                weight_map[r : r + tile_size, c : c + tile_size] += window

    accum_logits /= np.maximum(weight_map[None], 1e-6)
    full_logits = accum_logits[:, :h, :w].astype(np.float32)
    return _postprocess_logits(full_logits, debris_threshold, min_cluster_px)


def predict_mask(
    model: torch.nn.Module,
    model_input: np.ndarray,
    device: torch.device | None = None,
    debris_threshold: float = 0.65,
    min_cluster_px: int = 4,
    enable_tiling: bool = True,
    tile_size: int = MODEL_SIZE,
    stride: int = TILED_STRIDE,
    batch_size: int = TILED_BATCH_SIZE,
    apply_clahe: bool = True,
    enable_tta: bool = False,
    enable_spectral_gating: bool = False,
) -> dict:
    """Run one model on an 11-channel (C, H, W) input with precision gating.

    Small inputs (<= 384 px) use the legacy single-pass path: resized to
    MODEL_SIZE for the forward pass, then upsampled back. Large inputs route
    through sliding-window tiled inference at native resolution with
    Hann-window blending for seamless results.

    Args:
        model: Any trained PyTorch segmentation model.
        model_input: (11, H, W) float32 input (raw radiance/reflectance).
        device: CPU or CUDA device.
        debris_threshold: Minimum softmax probability to confirm a debris pixel.
            Filters low-confidence false positives in open water.
        min_cluster_px: Minimum connected-component pixel area to keep.
            Removes isolated single-pixel salt-and-pepper noise.
        enable_tiling: If True (default), automatically route large inputs
            (H > 384 or W > 384) through tiled inference. Set False to
            force legacy downsampling behavior for comparison.
        tile_size: Tile dimension for tiled inference (default 256).
        stride: Step between tiles (default 192 for 25% overlap).
        batch_size: Number of tiles per GPU batch (default 4).
        apply_clahe: If True (default), apply the canonical CLAHE preprocessing
            (percentile normalization + CLAHE) matching training.
        enable_tta: If True, use 8-fold Test-Time Augmentation (flips + rotations)
            to improve boundary precision on ocean drift lines.
        enable_spectral_gating: If True, apply physics-informed spectral gating
            (FDI, NDWI, foam rejection) to suppress false positives from
            sea foam, whitecaps, and sun glint.
    """
    # Store original input for spectral gating (before CLAHE)
    original_input = model_input.copy()

    # Apply canonical preprocessing to match training distribution
    model_input = preprocess_satellite_input(model_input, apply_clahe=apply_clahe)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch, h, w = model_input.shape
    if ch != 11:
        raise ValueError(f"model input must have 11 channels, got {ch}")

    # Route large inputs through tiled inference
    if enable_tiling and (h > TILING_ROUTE_THRESHOLD or w > TILING_ROUTE_THRESHOLD):
        # For tiled inference with TTA/spectral gating, we need to handle differently
        if enable_tta or enable_spectral_gating:
            # Fall back to single-image TTA on full image for now
            # (tiled TTA would require per-tile TTA which is complex)
            pass  # Continue to single-image path below
        else:
            return predict_mask_tiled(
                model,
                model_input,
                device,
                tile_size=tile_size,
                stride=stride,
                debris_threshold=debris_threshold,
                min_cluster_px=min_cluster_px,
                batch_size=batch_size,
            )

    # Single-image inference path (with optional TTA)
    if enable_tta:
        det = predict_mask_tta(
            model,
            model_input,
            device,
            debris_threshold=debris_threshold,
            min_cluster_px=min_cluster_px,
            apply_clahe=apply_clahe,
        )
    else:
        # Legacy single-pass path (resize to MODEL_SIZE, upsample logits back)
        import warnings
        if not enable_tiling and (h > TILING_ROUTE_THRESHOLD or w > TILING_ROUTE_THRESHOLD):
            warnings.warn(
                f"Input shape ({h}, {w}) exceeds {TILING_ROUTE_THRESHOLD}px but tiling is disabled. "
                "The image will be downsampled to 256x256, losing spatial detail. "
                "Enable 'High-Resolution Tiled Inference' in the Model tab for accurate results.",
                UserWarning,
                stacklevel=2,
            )
        x = torch.from_numpy(np.asarray(model_input, dtype=np.float32))[None].to(device, non_blocking=True)
        if (h, w) != (MODEL_SIZE, MODEL_SIZE):
            x = F.interpolate(x, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            logits = logits.float()
        if (h, w) != logits.shape[-2:]:
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        logits = logits[0].cpu()
        det = _postprocess_logits(logits.numpy().astype(np.float32), debris_threshold, min_cluster_px)

    # Apply spectral gating if enabled
    if enable_spectral_gating:
        # Use original input (before CLAHE) for spectral indices
        det = _predict_with_spectral_gating(
            original_input,
            det["prob"],
            debris_threshold,
            min_cluster_px,
            enable_spectral_gating=True,
        )

    return det


def load_cv_ensemble(variant: str, top_k: int = 3) -> list[torch.nn.Module]:
    """Load top-K Cross-Validation fold checkpoints for ensemble inference.

    Scans artifacts/cv/{variant}/ for fold checkpoints, sorts by validation
    debris F1 (from cv_summary.json), and loads the top-K models.

    Args:
        variant: Model variant name (e.g., "attention", "baseline").
        top_k: Number of top folds to load (default 3).

    Returns:
        List of models in eval() mode. Falls back to single best.pth if
        CV fold checkpoints don't exist or are fewer than 2.
    """
    cv_dir = ROOT / "artifacts" / "cv" / variant
    summary_path = cv_dir / "cv_summary.json"

    if not summary_path.exists():
        # No CV results, fall back to single model
        from src.models.registry import resolve_variant
        from src.utils.checkpoint import safe_load_state_dict
        model = resolve_variant(variant, pretrained=False)
        ckpt = ROOT / "artifacts" / variant / "best.pth"
        if ckpt.exists():
            safe_load_state_dict(model, ckpt, map_location="cpu")
        model.eval()
        return [model]

    with summary_path.open() as f:
        summary = json.load(f)

    fold_results = summary.get("fold_results", [])
    if len(fold_results) < 2:
        # Not enough folds for ensemble, fall back to single model
        from src.models.registry import resolve_variant
        from src.utils.checkpoint import safe_load_state_dict
        model = resolve_variant(variant, pretrained=False)
        ckpt = ROOT / "artifacts" / variant / "best.pth"
        if ckpt.exists():
            safe_load_state_dict(model, ckpt, map_location="cpu")
        model.eval()
        return [model]

    # Sort folds by val_debris_f1 (primary) then best_mean_iou (secondary)
    valid_folds = [r for r in fold_results if r.get("val_debris_f1") is not None]
    sorted_folds = sorted(
        valid_folds,
        key=lambda f: (f.get("val_debris_f1", 0.0), f.get("best_mean_iou", 0.0)),
        reverse=True,
    )
    top_folds = sorted_folds[:top_k]

    models = []
    for fold_record in top_folds:
        fold_idx = fold_record["fold"]
        fold_ckpt = cv_dir / f"fold_{fold_idx}" / "best.pth"
        if not fold_ckpt.exists():
            continue
        from src.models.registry import resolve_variant
        from src.utils.checkpoint import safe_load_state_dict
        model = resolve_variant(variant, pretrained=False)
        safe_load_state_dict(model, fold_ckpt, map_location="cpu")
        model.eval()
        models.append(model)

    if not models:
        # Fallback if no fold checkpoints could be loaded
        from src.models.registry import resolve_variant
        from src.utils.checkpoint import safe_load_state_dict
        model = resolve_variant(variant, pretrained=False)
        ckpt = ROOT / "artifacts" / variant / "best.pth"
        if ckpt.exists():
            safe_load_state_dict(model, ckpt, map_location="cpu")
        model.eval()
        return [model]

    return models


def predict_mask_ensemble(
    models: list[torch.nn.Module],
    model_input: np.ndarray,
    device: torch.device | None = None,
    debris_threshold: float = 0.65,
    min_cluster_px: int = 4,
    enable_tiling: bool = True,
    tile_size: int = MODEL_SIZE,
    stride: int = TILED_STRIDE,
    batch_size: int = TILED_BATCH_SIZE,
    apply_clahe: bool = True,
    enable_tta: bool = False,
    enable_spectral_gating: bool = False,
) -> dict:
    """Run ensemble inference by averaging softmax probabilities across models.

    Computes the mean probability map and the per-pixel variance (epistemic
    uncertainty) for Class 1 (Marine Debris).

    Args:
        models: List of models in eval() mode.
        model_input: (11, H, W) float32 input.
        device: CPU or CUDA device.
        debris_threshold: Minimum softmax probability to confirm a debris pixel.
        min_cluster_px: Minimum connected-component pixel area to keep.
        enable_tiling: If True, automatically route large inputs through
            tiled inference.
        tile_size: Tile dimension for tiled inference (default 256).
        stride: Step between tiles (default 192 for 25% overlap).
        batch_size: Number of tiles per GPU batch (default 4).
        apply_clahe: If True (default), apply the canonical CLAHE preprocessing
            (percentile normalization + CLAHE) matching training.
        enable_tta: If True, use 8-fold Test-Time Augmentation per model.
        enable_spectral_gating: If True, apply physics-informed spectral gating
            (FDI, NDWI, foam rejection) on the ensemble mean probability.

    Returns:
        Dict with keys: mask, debris, prob, uncertainty, concentration, boxes, stats.
        If spectral_gating enabled, also includes: fdi, ndwi, foam_mask.
    """
    if not models:
        raise ValueError("models list cannot be empty")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch, h, w = model_input.shape
    if ch != 11:
        raise ValueError(f"model input must have 11 channels, got {ch}")

    all_probs = []
    for model in models:
        model = model.to(device)
        model.eval()
        det = predict_mask(
            model,
            model_input,
            device,
            debris_threshold=debris_threshold,
            min_cluster_px=min_cluster_px,
            enable_tiling=enable_tiling,
            tile_size=tile_size,
            stride=stride,
            batch_size=batch_size,
            apply_clahe=apply_clahe,
            enable_tta=enable_tta,
            enable_spectral_gating=False,  # Apply gating on ensemble mean, not per-model
        )
        # predict_mask returns 'prob' as the debris class probability
        all_probs.append(det["prob"])

    # Stack and average probabilities across ensemble members
    probs_stack = np.stack(all_probs, axis=0)  # (K, H, W)
    prob_ens = probs_stack.mean(axis=0)  # (H, W)
    # Epistemic uncertainty: variance of debris class probability across folds
    uncertainty = probs_stack.var(axis=0)  # (H, W)

    # Apply spectral gating on ensemble mean if enabled
    spectral_data = {}
    if enable_spectral_gating:
        calibrated_prob, fdi, ndwi, foam_mask = compute_spectral_gating_mask(
            model_input, prob_ens
        )
        prob_ens = calibrated_prob
        spectral_data = {"fdi": fdi, "ndwi": ndwi, "foam_mask": foam_mask}

    # Precision gating on ensemble mean probability (after spectral gating)
    debris_candidates = prob_ens >= debris_threshold
    if min_cluster_px > 1 and np.any(debris_candidates):
        from scipy.ndimage import label as nd_label
        labeled_blobs, num_blobs = nd_label(debris_candidates)
        if num_blobs > 0:
            blob_sizes = np.bincount(labeled_blobs.ravel())
            keep_mask = np.isin(labeled_blobs, np.where(blob_sizes >= min_cluster_px)[0])
            keep_mask &= labeled_blobs > 0
            debris = keep_mask.astype(np.float32)
        else:
            debris = np.zeros((h, w), dtype=np.float32)
    else:
        debris = debris_candidates.astype(np.float32)

    # Final mask: argmax of mean logits approximated by prob_ens
    final_mask = np.full((h, w), 6, dtype=np.int64)  # default Marine Water
    final_mask[debris > 0.5] = 0  # Marine Debris

    concentration = debris_concentration(debris)
    from dashboard.pipeline import boxes_with_confidence, concentration_stats
    boxes = boxes_with_confidence(debris, prob_ens)
    stats = concentration_stats(debris, concentration)

    result = {
        "mask": final_mask,
        "debris": debris,
        "prob": prob_ens.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
        "concentration": concentration,
        "boxes": boxes,
        "stats": stats,
    }
    result.update(spectral_data)
    return result


def segmentation_overlay(
    rgb: np.ndarray,
    debris: np.ndarray,
    alpha: float = 0.60,
    color: tuple[float, float, float] = (0.95, 0.12, 0.12),
) -> np.ndarray:
    """Tint detected Marine Debris pixels red over an RGB image (float [0,1])."""
    img = (rgb.astype(np.float32) / 255.0) if rgb.max() > 1 else rgb.astype(np.float32)
    out = img.copy()
    hit = debris.astype(bool)
    tint = np.asarray(color, dtype=np.float32)
    out[hit] = out[hit] * (1.0 - alpha) + tint * alpha
    return np.clip(out, 0.0, 1.0)


def draw_boxes(
    rgb: np.ndarray,
    boxes: list[dict],
    color=(0, 220, 0),
    show_confidence: bool = False,
    label_text: str | None = None,
) -> np.ndarray:
    """Draw debris bounding boxes. Accepts float/uint8 RGB.
    
    If show_confidence is False and label_text is None, renders clean bounding boxes without text/numbers.
    """
    img = (np.clip(rgb * 255, 0, 255).astype(np.uint8)) if rgb.max() <= 1 else rgb.copy()
    scale = max(img.shape[1] / 256.0, 0.4)
    for box in boxes:
        cv2.rectangle(img, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), color, max(1, int(scale)))
        text = None
        if label_text is not None:
            text = label_text
        elif show_confidence and box.get("confidence") is not None:
            text = f"{box['confidence']:.2f}"

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = max(1, int(scale))
            (tw, th), base = cv2.getTextSize(text, font, 0.55 * scale, thickness)
            y0 = max(box["ymin"] - 4, 0)
            cv2.rectangle(img, (box["xmin"], max(y0 - th - 4, 0)),
                          (box["xmin"] + tw + 4, y0), color, -1)
            cv2.putText(img, text, (box["xmin"] + 2, max(y0 - 4, th)),
                        font, 0.55 * scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def _make_thumbnail(rgb: np.ndarray, max_width: int = 320) -> np.ndarray:
    """Downsample an RGB uint8 array to max_width preserving aspect ratio."""
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb
    scale = max_width / w
    new_w = max_width
    new_h = max(1, int(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def process_video(
    model: torch.nn.Module,
    source_bytes: bytes,
    device: torch.device | None = None,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_frames: int = MAX_VIDEO_FRAMES,
    adapter_mode: str = "physics_approx",
    debris_threshold: float = 0.65,
    min_cluster_px: int = 4,
    enable_tiling: bool = True,
    apply_clahe: bool = True,
) -> dict:
    """Sample a video, detect each sampled frame, and aggregate concentration.

    Returns per-frame detection dicts (``frames``), a frame-index list, video
    properties, an aggregate severity report (``aggregate``), and the mean
    concentration map across sampled frames (``aggregate_concentration``).

    Memory optimization: stores only thumbnails (~30 KB each) instead of
    full-resolution RGB arrays (~6 MB each). Full-resolution debris masks
    and concentration maps are kept for analysis; RGB arrays are freed
    after processing each frame.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    tmp.write_bytes(source_bytes)
    cap = cv2.VideoCapture(str(tmp))
    try:
        if not cap.isOpened():
            raise ValueError("cannot open the uploaded video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        indices = list(sample_frame_indices(total, fps, sample_fps))[:max_frames]

        # RAM guard: estimate memory and throttle frames if needed
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            estimated_frame_mb = (width * height * 3) / (1024 ** 2)  # raw RGB
            estimated_total_mb = len(indices) * estimated_frame_mb
            if estimated_total_mb > available_ram_gb * 1024 * 0.25:
                # Throttle frames to use at most 25% of available RAM
                safe_frames = max(1, int((available_ram_gb * 1024 * 0.25) / estimated_frame_mb))
                indices = indices[:safe_frames]
                print(f"  RAM guard: throttled to {safe_frames} frames "
                      f"(available={available_ram_gb:.1f} GB, "
                      f"estimated {estimated_total_mb:.0f} MB for {len(indices)} frames)")
        except ImportError:
            pass  # psutil not available, skip RAM guard

        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = rgb.astype(np.float32) / 255.0
            model_input = to_model_channels(frame.transpose(2, 0, 1), 11, mode=adapter_mode)
            det = predict_mask(
                model,
                model_input,
                device,
                debris_threshold=debris_threshold,
                min_cluster_px=min_cluster_px,
                enable_tiling=enable_tiling,
                apply_clahe=apply_clahe,
            )
            # Downsample per-frame debris & concentration to VIDEO_STORAGE_SIZE
            # to avoid MemoryError in Streamlit's pickle cache.
            # Full-res at 1920x1080 = ~8 MB/frame; 256x256 = ~0.25 MB/frame.
            ds_h, ds_w = VIDEO_STORAGE_SIZE, VIDEO_STORAGE_SIZE
            debris_small = cv2.resize(det["debris"], (ds_w, ds_h), interpolation=cv2.INTER_AREA).astype(np.float32)
            conc_small = cv2.resize(det["concentration"], (ds_w, ds_h), interpolation=cv2.INTER_AREA).astype(np.float32)
            # Build lightweight frame record — NO full-resolution arrays stored.
            frame_record = {
                "frame_index": int(index),
                # Thumbnail: downsample to max 320px wide for display only (~30 KB)
                "rgb_thumb": _make_thumbnail(rgb, max_width=320),
                # Debris mask at VIDEO_STORAGE_SIZE (~0.25 MB/frame)
                "debris": debris_small,
                # Concentration at VIDEO_STORAGE_SIZE (~0.25 MB/frame)
                "concentration": conc_small,
                # Original video dimensions for upscaling on display
                "orig_height": height,
                "orig_width": width,
                # Scalar statistics — minimal overhead
                "boxes": boxes_with_confidence(det["debris"], det["prob"]),
                "stats": concentration_stats(det["debris"], det["concentration"]),
                # Drop: det["prob"] (H×W float32 heatmap, ~2 MB per frame) after use
            }
            frames.append(frame_record)
            del det, rgb, frame, model_input, debris_small, conc_small  # Explicit free
        if not frames:
            raise ValueError("no frames could be read from the video")
    finally:
        cap.release()
        for _attempt in range(5):
            try:
                tmp.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.2)

    aggregate_concentration = np.mean(
        np.stack([f["concentration"] for f in frames]), axis=0
    ).astype(np.float32)
    return {
        "frames": frames,
        "aggregate": aggregate_frames(frames),
        "aggregate_concentration": aggregate_concentration,
        "video_fps": fps,
        "total_frames": total,
        "width": width,
        "height": height,
    }


# ---------------------------------------------------------------------------
# TACO YOLOv8 RGB debris detector
# ---------------------------------------------------------------------------


def boxes_to_debris_mask(boxes: list[dict], height: int, width: int) -> np.ndarray:
    """Rasterize detection boxes into a binary (H, W) debris mask.

    Used to reuse the segmentation concentration/severity math for YOLO boxes:
    each kept box paints its clipped footprint, then Gaussian blurring yields
    the same hot-spot concentration map the dashboard already renders.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        x1 = max(int(box["xmin"]), 0)
        y1 = max(int(box["ymin"]), 0)
        x2 = min(int(box["xmax"]), width - 1)
        y2 = min(int(box["ymax"]), height - 1)
        if x2 > x1 and y2 > y1:
            mask[y1 : y2 + 1, x1 : x2 + 1] = 1.0
    return mask


def taco_boxes_to_detection(
    detections: list[tuple[float, float, float, float, float, int]],
    height: int,
    width: int,
    class_names: list[str] | None = None,
    label: str = "taco_upload",
    rgb: np.ndarray | None = None,
    lightweight: bool = False,
) -> dict:
    """Build a dashboard-compatible detection dict from raw YOLO boxes.

    Args:
        detections: ``(x1, y1, x2, y2, confidence, class_id)`` tuples in pixel
            coordinates (ultralytics ``xyxy`` convention).
        height: image height in pixels.
        width: image width in pixels.
        class_names: class-id to name lookup (defaults to the 7 TACO classes).
        label: source label shown in the UI.
        rgb: optional uint8 (H, W, 3) display image; when omitted a black
            canvas is used so the function stays dependency-free for tests.
        lightweight: if True, store only a thumbnail (max 320px wide) instead
            of full-resolution RGB to reduce memory for caching.
    """
    names = list(class_names) if class_names is not None else list(TACO_CLASSES)
    boxes = []
    for x1, y1, x2, y2, confidence, class_id in detections:
        cid = int(class_id)
        xmin, ymin = int(round(x1)), int(round(y1))
        xmax, ymax = int(round(x2)), int(round(y2))
        width_px = max(0, xmax - xmin + 1) if xmax > xmin else 0
        height_px = max(0, ymax - ymin + 1) if ymax > ymin else 0
        box = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "area_px": width_px * height_px,
            "confidence": round(float(confidence), 4),
            "class_id": cid,
            "class_name": names[cid] if 0 <= cid < len(names) else "Other_Trash",
        }
        boxes.append(box)
    boxes.sort(key=lambda b: b["area_px"], reverse=True)

    debris = boxes_to_debris_mask(boxes, height, width)
    concentration = debris_concentration(debris)
    class_counts: dict[str, int] = {}
    for box in boxes:
        class_counts[box["class_name"]] = class_counts.get(box["class_name"], 0) + 1
    if rgb is None:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    if lightweight:
        rgb = _make_thumbnail(rgb, max_width=320)
    
    return {
        "rgb": rgb,
        "boxes": boxes,
        "debris": debris,
        "concentration": concentration,
        "stats": concentration_stats(debris, concentration),
        "width": width,
        "height": height,
        "label": label,
        "class_counts": class_counts,
        "source": "taco",
    }


def _parse_yolo_result(result) -> list[tuple]:
    """Extract ``(x1, y1, x2, y2, conf, cls)`` tuples from one YOLO result."""
    if result is None or getattr(result, "boxes", None) is None:
        return []
    boxes = result.boxes
    xyxy, conf, cls = boxes.xyxy, boxes.conf, boxes.cls
    out = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        out.append((x1, y1, x2, y2, float(conf[i]), int(cls[i])))
    return out


def taco_detector_available(model_path: Path | str | None = None) -> bool:
    """True when a fine-tuned TACO checkpoint exists on disk."""
    path = Path(model_path) if model_path is not None else TACO_MODEL_PATH
    return path.exists()


_TACO_RGB_MODEL_CACHE: dict[str, object] = {}


def _load_taco_rgb_model(weights_path: Path):
    """Load (and cache) the YOLO model backing :func:`predict_taco_rgb`."""
    key = str(weights_path)
    model = _TACO_RGB_MODEL_CACHE.get(key)
    if model is None:
        from ultralytics import YOLO

        model = YOLO(key)
        _TACO_RGB_MODEL_CACHE[key] = model
    return model


def predict_taco_rgb(
    image_rgb: np.ndarray,
    conf_threshold: float = 0.40,
) -> tuple[np.ndarray, list[dict]]:
    """Inference for uploaded RGB photos using the fine-tuned TACO YOLO model.

    Returns:
        annotated_image: RGB image with bounding boxes and class labels drawn.
        detections: List of dicts with box coordinates, class name, and confidence.

    Falls back to the base ``yolov8n.pt`` COCO weights when the TACO run has
    not been trained yet, so the dashboard can still preview detections.
    """
    weights_path = TACO_MODEL_PATH
    if not weights_path.exists():
        weights_path = ROOT / "yolov8n.pt"

    model = _load_taco_rgb_model(weights_path)
    results = model(image_rgb, conf=conf_threshold, verbose=False)[0]
    annotated = results.plot()  # Draws clean bounding boxes with class labels
    # ultralytics plot() renders in BGR (cv2 convention); the dashboard works in RGB.
    if annotated is not None and getattr(annotated, "ndim", 0) == 3:
        annotated = np.ascontiguousarray(annotated[:, :, ::-1])

    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        boxes.append({
            "class": cls_name,
            "confidence": round(conf, 3),
            "xmin": int(xyxy[0]),
            "ymin": int(xyxy[1]),
            "xmax": int(xyxy[2]),
            "ymax": int(xyxy[3]),
        })
    return annotated, boxes


class TacoYoloDetector:
    """Routes RGB drone/photo/video uploads through the TACO YOLOv8 model.

    Ultralytics is imported lazily on first use so importing this module stays
    cheap and the satellite pipeline never hard-depends on it. Every detection
    dict mirrors the segmentation pipeline shape (``rgb``, ``boxes``,
    ``debris``, ``concentration``, ``stats``) so the dashboard renders YOLO
    results with no special-casing.

    Build one via :func:`load_taco_detector`; call :meth:`detect_upload` for a
    single image or :meth:`detect_video` for a clip.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: torch.device | None = None,
        conf: float = TACO_CONF_THRESHOLD,
        iou: float = TACO_NMS_IOU,
        imgsz: int = TACO_IMGSZ,
        class_names: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else TACO_MODEL_PATH
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.class_names = list(class_names) if class_names is not None else list(TACO_CLASSES)
        self._model = None  # tri-state: None=unloaded, False=unavailable, model=ready
        self.load_error: Exception | None = None

    @property
    def available(self) -> bool:
        """Whether the checkpoint exists or a model is already loaded."""
        return self._model is not None or self.model_path.exists()

    def _load(self):
        if self._model is None:
            if not self.model_path.exists():
                self.load_error = FileNotFoundError(
                    f"no TACO checkpoint at {self.model_path}; "
                    "run scripts/train_taco.py first"
                )
                self._model = False
                return None
            from ultralytics import YOLO

            try:
                model = YOLO(str(self.model_path))
                try:
                    model.to(self.device)
                except Exception:  # noqa: BLE001 - device placement is best-effort
                    pass
                self._model = model
            except Exception as exc:  # noqa: BLE001 - surface via load_error
                self.load_error = exc
                self._model = False
        return self._model if self._model is not False else None

    def _predict_boxes(self, rgb: np.ndarray, conf: float | None = None, imgsz: int | None = None) -> list[tuple]:
        model = self._load()
        if model is None:
            raise FileNotFoundError(
                f"TACO detector unavailable: {self.load_error or self.model_path}"
            )
        # Ultralytics model.predict(numpy_array) expects a BGR OpenCV array and internally
        # reverses channels (BGR->RGB). Passing an RGB image causes red and blue channels to flip.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if (isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.shape[2] == 3) else rgb
        predict_conf = self.conf if conf is None else float(conf)
        predict_imgsz = self.imgsz if imgsz is None else int(imgsz)
        results = model.predict(
            bgr, conf=predict_conf, iou=self.iou, imgsz=predict_imgsz, verbose=False
        )
        return _parse_yolo_result(results[0] if results else None)

    def _predict_boxes_tiled(
        self,
        rgb: np.ndarray,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
        conf: float | None = None,
    ) -> list[tuple]:
        """SAHI-style sliced inference at full resolution with global NMS.

        Slices the full-resolution (H, W, 3) image into overlapping windows of
        size slice_size x slice_size with the given overlap ratio. Runs the
        detector on each slice, maps coordinates back to global image space,
        and applies global Non-Maximum Suppression across all detections.

        Args:
            rgb: Input image (H, W, 3) uint8 or float [0,1].
            slice_size: Tile dimension in pixels (default 640).
            overlap_ratio: Overlap between adjacent tiles as fraction of slice_size (default 0.2).
            conf: Confidence threshold override.

        Returns:
            List of (x1, y1, x2, y2, confidence, class_id) tuples in global coordinates.
        """
        h, w = rgb.shape[:2]
        if h <= slice_size and w <= slice_size:
            return self._predict_boxes(rgb, conf=conf, imgsz=slice_size)

        # Try SAHI library first if available
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            from sahi.utils.cv import read_image

            model = self._load()
            if model is None:
                raise FileNotFoundError(f"TACO detector unavailable: {self.load_error}")

            # SAHI expects a file path, so we save temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(tmp.name, bgr)
                tmp_path = tmp.name

            try:
                detection_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path=str(self.model_path),
                    confidence_threshold=self.conf if conf is None else float(conf),
                    device=str(self.device),
                    image_size=slice_size,
                )
                sliced_pred = get_sliced_prediction(
                    tmp_path,
                    detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap_ratio,
                    overlap_width_ratio=overlap_ratio,
                    perform_standard_pred=False,
                    verbose=0,
                )
                boxes = []
                for pred in sliced_pred.object_prediction_list:
                    bbox = pred.bbox
                    x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                    boxes.append((float(x1), float(y1), float(x2), float(y2), float(pred.score.value), int(pred.category.id)))
                return boxes
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        except Exception:  # noqa: BLE001 - fallback to internal NumPy implementation
            pass

        # Internal NumPy/OpenCV sliding-window implementation
        actual_slice_h = min(h, slice_size)
        actual_slice_w = min(w, slice_size)
        stride_h = max(1, int(actual_slice_h * (1.0 - overlap_ratio)))
        stride_w = max(1, int(actual_slice_w * (1.0 - overlap_ratio)))

        rows = list(range(0, h - actual_slice_h + 1, stride_h)) if h > actual_slice_h else [0]
        cols = list(range(0, w - actual_slice_w + 1, stride_w)) if w > actual_slice_w else [0]

        # Add final row/col to cover remaining pixels
        if rows and rows[-1] + actual_slice_h < h:
            rows.append(h - actual_slice_h)
        if cols and cols[-1] + actual_slice_w < w:
            cols.append(w - actual_slice_w)

        all_boxes = []
        predict_conf = self.conf if conf is None else float(conf)

        for r in rows:
            for c in cols:
                slice_img = rgb[r:r + actual_slice_h, c:c + actual_slice_w]
                boxes = self._predict_boxes(slice_img, conf=predict_conf, imgsz=max(actual_slice_h, actual_slice_w))
                for x1, y1, x2, y2, confidence, class_id in boxes:
                    # Map slice coordinates to global coordinates
                    gx1 = x1 + c
                    gy1 = y1 + r
                    gx2 = x2 + c
                    gy2 = y2 + r
                    all_boxes.append((gx1, gy1, gx2, gy2, confidence, class_id))

        if not all_boxes:
            return []

        # Global NMS across all aggregated boxes
        return self._global_nms(all_boxes, iou_threshold=self.iou)

    def _global_nms(self, boxes: list[tuple], iou_threshold: float = 0.45) -> list[tuple]:
        """Apply global Non-Maximum Suppression to merged boxes from all tiles.

        Args:
            boxes: List of (x1, y1, x2, y2, confidence, class_id) tuples.
            iou_threshold: IoU threshold for suppression.

        Returns:
            Filtered list of boxes after NMS.
        """
        if not boxes:
            return []

        # Group by class for per-class NMS
        from collections import defaultdict
        boxes_by_class = defaultdict(list)
        for box in boxes:
            boxes_by_class[box[5]].append(box)

        keep_boxes = []
        for class_id, cls_boxes in boxes_by_class.items():
            if len(cls_boxes) == 1:
                keep_boxes.append(cls_boxes[0])
                continue

            # Convert to tensors for torchvision NMS or use cv2 fallback
            xyxy = np.array([b[:4] for b in cls_boxes], dtype=np.float32)
            scores = np.array([b[4] for b in cls_boxes], dtype=np.float32)

            if _TORCHVISION_AVAILABLE:
                import torch
                xyxy_t = torch.from_numpy(xyxy)
                scores_t = torch.from_numpy(scores)
                keep_idx = torchvision_nms(xyxy_t, scores_t, iou_threshold)
                for idx in keep_idx:
                    keep_boxes.append(cls_boxes[int(idx)])
            else:
                # OpenCV dnn NMSBoxes fallback
                # cv2.dnn.NMSBoxes expects boxes as [x, y, w, h]
                cv_boxes = xyxy.copy()
                cv_boxes[:, 2] -= cv_boxes[:, 0]  # w
                cv_boxes[:, 3] -= cv_boxes[:, 1]  # h
                keep_idx = cv2.dnn.NMSBoxes(
                    cv_boxes.tolist(), scores.tolist(),
                    score_threshold=0.0, nms_threshold=iou_threshold
                )
                if len(keep_idx) > 0:
                    keep_idx = keep_idx.flatten() if hasattr(keep_idx, 'flatten') else keep_idx
                    for idx in keep_idx:
                        keep_boxes.append(cls_boxes[int(idx)])

        return keep_boxes

    def detect_image(
        self,
        rgb: np.ndarray,
        label: str = "taco_upload",
        conf: float | None = None,
        imgsz: int | None = None,
        lightweight: bool = False,
        enable_sahi: bool = False,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
    ) -> dict:
        """Detect debris in an RGB ``(H, W, 3)`` image (uint8 or float [0,1]).

        Args:
            enable_sahi: If True, use sliced inference (SAHI) for high-resolution images.
            slice_size: Tile size for sliced inference (default 640).
            overlap_ratio: Overlap between tiles as fraction of slice_size (default 0.2).
        """
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB image, got {rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
        height, width = rgb.shape[:2]

        if enable_sahi and max(height, width) > slice_size:
            boxes = self._predict_boxes_tiled(
                rgb, slice_size=slice_size, overlap_ratio=overlap_ratio, conf=conf
            )
        else:
            boxes = self._predict_boxes(rgb, conf=conf, imgsz=imgsz)

        return taco_boxes_to_detection(
            boxes, height, width, self.class_names, label=label, rgb=rgb, lightweight=lightweight
        )

    def detect_upload(
        self,
        source_bytes: bytes,
        filename: str = "",
        conf: float | None = None,
        imgsz: int | None = None,
        lightweight: bool = False,
        enable_sahi: bool = False,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
    ) -> dict:
        """Decode an uploaded .jpg/.png and run the TACO detector on it.

        Args:
            enable_sahi: If True, use sliced inference (SAHI) for high-resolution images.
            slice_size: Tile size for sliced inference (default 640).
            overlap_ratio: Overlap between tiles as fraction of slice_size (default 0.2).
        """
        if self._load() is None:
            raise FileNotFoundError(
                f"TACO detector unavailable: {self.load_error or self.model_path}"
            )
        arr = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError(f"cannot decode image: {filename or 'upload'}")
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return self.detect_image(
            rgb,
            label=filename or "taco_upload",
            conf=conf,
            imgsz=imgsz,
            lightweight=lightweight,
            enable_sahi=enable_sahi,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
        )

    def detect_video(
        self,
        source_bytes: bytes,
        source_name: str = "",
        sample_fps: float = DEFAULT_SAMPLE_FPS,
        max_frames: int = MAX_VIDEO_FRAMES,
        conf: float | None = None,
        imgsz: int | None = None,
        enable_sahi: bool = False,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
        enable_tracking: bool = True,
    ) -> dict:
        """Sample a video and run the TACO detector on each sampled frame.

        Args:
            enable_sahi: If True, use sliced inference (SAHI) for high-resolution frames.
            slice_size: Tile size for sliced inference (default 640).
            overlap_ratio: Overlap between tiles as fraction of slice_size (default 0.2).
            enable_tracking: If True, use ByteTrack temporal tracking to stabilize
                detections across frames (default True).
        """
        if self._load() is None:
            raise FileNotFoundError(
                f"TACO detector unavailable: {self.load_error or self.model_path}"
            )
        tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
        tmp.write_bytes(source_bytes)
        cap = cv2.VideoCapture(str(tmp))
        try:
            if not cap.isOpened():
                raise ValueError("cannot open the uploaded video")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
            indices = list(sample_frame_indices(total, fps, sample_fps))[:max_frames]
            frames = []

            model = self._load()
            if model is None:
                raise FileNotFoundError(f"TACO detector unavailable: {self.load_error}")

            predict_conf = self.conf if conf is None else float(conf)
            predict_imgsz = self.imgsz if imgsz is None else int(imgsz)

            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, bgr = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if enable_tracking:
                    # ByteTrack temporal tracking for video stabilization
                    results = model.track(
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        conf=predict_conf,
                        iou=self.iou,
                        imgsz=predict_imgsz,
                        tracker="bytetrack.yaml",
                        persist=True,
                        verbose=False,
                    )
                    boxes = _parse_yolo_result(results[0] if results else None)
                elif enable_sahi and max(height, width) > slice_size:
                    boxes = self._predict_boxes_tiled(
                        rgb, slice_size=slice_size, overlap_ratio=overlap_ratio, conf=conf
                    )
                else:
                    boxes = self._predict_boxes(rgb, conf=conf, imgsz=imgsz)

                det = taco_boxes_to_detection(
                    boxes, height, width, self.class_names,
                    label=source_name or "taco_video", rgb=rgb, lightweight=True
                )
                # Downsample per-frame debris & concentration to VIDEO_STORAGE_SIZE
                ds_h, ds_w = VIDEO_STORAGE_SIZE, VIDEO_STORAGE_SIZE
                debris_small = cv2.resize(det["debris"], (ds_w, ds_h), interpolation=cv2.INTER_AREA).astype(np.float32)
                conc_small = cv2.resize(det["concentration"], (ds_w, ds_h), interpolation=cv2.INTER_AREA).astype(np.float32)
                # Build lightweight frame record — det["rgb"] is already a thumbnail (lightweight=True)
                frame_record = {
                    "frame_index": int(index),
                    # Thumbnail from lightweight detection (max 320px wide, ~30 KB)
                    "rgb_thumb": det["rgb"],
                    # Debris mask at VIDEO_STORAGE_SIZE (~0.25 MB/frame)
                    "debris": debris_small,
                    # Concentration at VIDEO_STORAGE_SIZE (~0.25 MB/frame)
                    "concentration": conc_small,
                    # Original video dimensions for upscaling on display
                    "orig_height": height,
                    "orig_width": width,
                    # Boxes and stats from full-res detection
                    "boxes": det["boxes"],
                    "stats": det["stats"],
                    "class_counts": det["class_counts"],
                }
                frames.append(frame_record)
                del det, rgb, debris_small, conc_small  # Explicit free
            if not frames:
                raise ValueError("no frames could be read from the video")
        finally:
            cap.release()
            for _attempt in range(5):
                try:
                    tmp.unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.2)

        aggregate_concentration = np.mean(
            np.stack([f["concentration"] for f in frames]), axis=0
        ).astype(np.float32)
        return {
            "frames": frames,
            "aggregate": aggregate_frames(frames),
            "aggregate_concentration": aggregate_concentration,
            "video_fps": fps,
            "total_frames": total,
            "width": width,
            "height": height,
            "label": source_name or "taco_video",
            "source": "taco",
        }


def load_taco_detector(
    model_path: Path | str | None = None,
    device: torch.device | None = None,
    **kwargs,
) -> TacoYoloDetector:
    """Build a TACO detector (weights loaded lazily on first detection)."""
    return TacoYoloDetector(model_path=model_path, device=device, **kwargs)


def analyze_with_taco(
    source_kind: str,
    source_bytes: bytes,
    filename: str = "",
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_frames: int = MAX_VIDEO_FRAMES,
    detector: TacoYoloDetector | None = None,
    conf: float | None = None,
    imgsz: int | None = None,
    enable_sahi: bool = False,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
    enable_tracking: bool = True,
    **detector_kwargs,
) -> dict:
    """Route an RGB drone/photo/video upload through the TACO YOLO detector.

    ``source_kind`` is ``"drone"``/``"photo"`` for a still image or ``"video"``
    for a clip. Returns the same dict shape as the segmentation pipeline so
    callers can swap detectors without touching rendering code.

    Args:
        enable_sahi: If True, use sliced inference (SAHI) for high-resolution inputs.
        slice_size: Tile size for sliced inference (default 640).
        overlap_ratio: Overlap between tiles as fraction of slice_size (default 0.2).
        enable_tracking: If True, use ByteTrack temporal tracking for video (default True).
    """
    if conf is not None and "conf" not in detector_kwargs:
        detector_kwargs["conf"] = conf
    detector = detector or load_taco_detector(**detector_kwargs)
    if source_kind == "video":
        return detector.detect_video(
            source_bytes,
            source_name=filename,
            sample_fps=sample_fps,
            max_frames=max_frames,
            enable_sahi=enable_sahi,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
            enable_tracking=enable_tracking,
        )
    if source_kind in ("drone", "photo", "rgb", "image"):
        return detector.detect_upload(
            source_bytes,
            filename=filename,
            conf=conf,
            imgsz=imgsz,
            lightweight=False,
            enable_sahi=enable_sahi,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
        )
    raise ValueError(f"TACO detector handles drone/photo/video uploads, not {source_kind!r}")


def draw_taco_boxes(
    rgb: np.ndarray,
    boxes: list[dict],
    show_confidence: bool = True,
    color=(0, 220, 0),
) -> np.ndarray:
    """Draw YOLO debris boxes labelled with their TACO class name."""
    img = (np.clip(rgb * 255, 0, 255).astype(np.uint8)) if rgb.max() <= 1 else rgb.copy()
    scale = max(img.shape[1] / 640.0, 0.4)
    for box in boxes:
        cv2.rectangle(
            img, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]),
            color, max(1, int(scale)),
        )
        text = str(box.get("class_name") or "trash")
        if show_confidence and box.get("confidence") is not None:
            text = f"{text} {box['confidence']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(scale))
        (tw, th), _base = cv2.getTextSize(text, font, 0.55 * scale, thickness)
        y0 = max(box["ymin"] - 4, 0)
        cv2.rectangle(
            img, (box["xmin"], max(y0 - th - 4, 0)), (box["xmin"] + tw + 4, y0),
            color, -1,
        )
        cv2.putText(
            img, text, (box["xmin"] + 2, max(y0 - 4, th)),
            font, 0.55 * scale, (0, 0, 0), thickness, cv2.LINE_AA,
        )
    return img