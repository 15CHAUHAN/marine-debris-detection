"""MARIDA interactive dashboard - the problem-statement output pipeline.

    Satellite/Drone Image/Video
        -> Plastic Detection (segmentation + debris boxes)
        -> Concentration (density / area / severity + hot heatmap)
        -> Pollution Map (interactive folium/Leaflet GIS overlay)

Run from the project root:
    .venv\\Scripts\\python.exe -m streamlit run dashboard\\app.py

Tabs:
  1. Source Input Mode  - upload/select the input (Satellite GeoTIFF, MARIDA
                          patch, Drone image, or a Drone/Coastal video)
  2. Model Selection    - one of the 5 hybrid segmentation techniques
  3. Plastic Detection  - original vs segmentation overlay + debris boxes
  4. Concentration      - density %, area (px and m^2), severity, heatmap
  5. Interactive Map    - folium map with georeferenced colored cells + overlay

Tabs 4 and 5 also offer a one-click "official" PDF pollution incident report
(dashboard/report_generator.py), rendered fully in memory.

All heavy logic lives in dashboard/pipeline.py (detection) and
dashboard/geo.py (georeferencing/map); this file is the UI shell.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import rasterio
import rasterio.io
import streamlit as st
import torch
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import geo
from dashboard.pipeline import (
    TACO_CLASSES,
    TACO_MODEL_PATH,
    analyze_with_taco,
    boxes_with_confidence,
    concentration_stats,
    draw_boxes,
    draw_taco_boxes,
    extract_exif_gps,
    load_cv_ensemble,
    predict_mask,
    predict_mask_ensemble,
    process_video,
    read_geotiff,
    read_rgb_upload,
    segmentation_overlay,
    severity_level,
    taco_detector_available,
)
from src.utils.spectral_indices import compute_fdi, compute_ndwi
from src.utils.domain_validator import (
    check_geotiff_bands,
    check_image_is_water_scene,
    check_rgb_photo_domain,
)

TACO_METRICS_PATH = ROOT / "artifacts" / "taco_training" / "test_metrics.json"
from dashboard.report_generator import ReportRegistry, generate_pdf_report
from src.data.encoding import CLASS_NAMES
from src.data.ingestion import MaridaDataset, PatchMeta
from src.data.plastic_l2a import PlasticL2aDataset
from src.data.splits import load_official_splits
from src.mapping.georeference import write_concentration_tiff
from src.models.registry import resolve_variant

st.set_page_config(
    page_title="OceanSentinel AI · Marine Plastic Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The 5 requested hybrid techniques, trained winner first.
HYBRID_VARIANTS = ["attention", "cnn_unet", "effnet_transformer", "yolo_vit_lstm", "yolo_bilstm"]
HYBRID_LABELS = {
    "attention": "U-Net + Attention (scSE) · SOTA Winner",
    "cnn_unet": "CNN + U-Net",
    "effnet_transformer": "EfficientNet + Transformer",
    "yolo_vit_lstm": "YOLO + Vision Transformer + LSTM",
    "yolo_bilstm": "YOLO + BiLSTM",
}

SOURCE_CODES = {
    "Satellite GeoTIFF upload": "satellite",
    "MARIDA test patch": "marida",
    "Plastic L2A (Sentinel-2, 2021-06-26)": "plastic_l2a",
    "Drone image (.jpg/.png)": "drone",
    "Drone / Coastal video (.mp4/.avi)": "video",
}

SEVERITY_PILL = {
    "Low": ("#10b981", "rgba(16, 185, 129, 0.15)"),
    "Moderate": ("#f59e0b", "rgba(245, 158, 11, 0.15)"),
    "High": ("#f97316", "rgba(249, 115, 22, 0.15)"),
    "Critical": ("#ef4444", "rgba(239, 68, 68, 0.15)"),
}

SEVERITY_BANDS = (
    ("Low",      0.0),    # 0 – 0.5%  : background noise / sparse scatter
    ("Moderate", 0.5),    # 0.5 – 2.5%: confirmed debris aggregation
    ("High",     2.5),    # 2.5 – 5.0%: significant marine pollution event
    ("Critical", 5.0),    # > 5.0%    : dense accumulation — MARIDA p90 level
)

PLACEHOLDER_GEO = {"lat": float("nan"), "lon": float("nan"), "gsd_m": 0.1}
DEFAULT_GSD_M = 0.1
GPS_NOT_SET_SENTINEL = float("nan")

SOURCE_HELP = {
    "Satellite GeoTIFF upload":
        "Upload a Sentinel-2 Level-2A GeoTIFF with 11 spectral bands. "
        "3-band RGB GeoTIFFs are accepted with a warning (reduced accuracy).",
    "MARIDA test patch":
        "Select from the 1,387 official MARIDA benchmark scenes. "
        "Ground truth masks are available for direct accuracy scoring.",
    "Plastic L2A (Sentinel-2, 2021-06-26)":
        "ESA Plastic Litter Project — calibrated plastic targets in the Aegean Sea.",
    "Drone image (.jpg/.png)":
        "Best for high-resolution coastal drone or smartphone photos. "
        "Routes to TACO YOLOv8 litter detector (not the satellite model).",
    "Drone / Coastal video (.mp4/.avi)":
        "Frame-sampled video processed by TACO YOLOv8. "
        "Max 60 frames sampled at 0.5–5 FPS with temporal aggregation.",
}

MAPS_DIR = ROOT / "artifacts" / "maps"
TAB_NAV_KEY = "tab_nav"
MAP_TAB_LABEL = "5. Geospatial Map"

# File-backed repeat-pass registry feeding the PDF report section
REPORT_HISTORY = ReportRegistry(ROOT / "artifacts" / "reports" / "history.json")


# ── Modern Professional UI / UX Styling ───────────────────────────────────────
def inject_custom_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #e2e8f0;
        }

        .stApp {
            background-color: #080c14;
        }

        /* Executive Metrics Cards */
        div[data-testid="stMetric"] {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 14px 18px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
            transition: border-color 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #334155;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            font-size: 1.45rem !important;
            color: #f8fafc !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8 !important;
        }

        /* Enterprise Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: #0f172a;
            padding: 5px;
            border-radius: 8px;
            border: 1px solid #1e293b;
            margin-bottom: 22px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 38px;
            border-radius: 6px;
            padding: 0 18px;
            font-weight: 500;
            font-size: 0.88rem;
            color: #94a3b8;
            background: transparent;
            transition: all 0.15s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #f8fafc;
            background-color: #1e293b;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0284c7 !important;
            color: #ffffff !important;
            font-weight: 600;
        }

        /* Action Buttons */
        .stButton > button[kind="primary"] {
            background-color: #0284c7;
            border: 1px solid #0369a1;
            border-radius: 6px;
            padding: 10px 24px;
            font-weight: 600;
            font-size: 0.90rem;
            color: #ffffff;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
            transition: all 0.15s ease;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #0369a1;
            border-color: #075985;
        }

        /* Telemetry Chips */
        .telemetry-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            color: #cbd5e1;
        }
        .telemetry-chip.status-green {
            border-color: rgba(16, 185, 129, 0.3);
            color: #34d399;
            background-color: rgba(16, 185, 129, 0.08);
        }
        .telemetry-chip.status-blue {
            border-color: rgba(2, 132, 199, 0.3);
            color: #38bdf8;
            background-color: rgba(2, 132, 199, 0.08);
        }
        .telemetry-chip.status-slate {
            border-color: #334155;
            color: #94a3b8;
            background-color: #1e293b;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0b0f17;
            border-right: 1px solid #1e293b;
        }

        /* Image Framing */
        [data-testid="stImage"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #1e293b;
        }

        /* Expander headers */
        .streamlit-expanderHeader {
            font-size: 0.88rem;
            font-weight: 600;
            background-color: #0f172a;
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_marida_and_splits():
    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    return marida, splits


@st.cache_resource(show_spinner="Loading Plastic L2A dataset...")
def load_plastic_l2a():
    data_root = ROOT / "artifacts" / "plastic_l2a"
    if not (data_root / "patches").exists():
        return None
    return PlasticL2aDataset(data_root, allow_pending=False)


@st.cache_data(show_spinner=False)
def debris_test_patches():
    marida, splits = load_marida_and_splits()
    return [p for p in splits["test"] if int((marida.read_mask(p) == 1).sum()) > 0]


@st.cache_resource(show_spinner="Loading model weights...")
def load_model(variant: str):
    ckpt = ROOT / "artifacts" / variant / "best.pth"
    if not ckpt.exists():
        return None, None
    model = resolve_variant(variant, pretrained=False)
    from src.utils.checkpoint import safe_load_checkpoint
    state = safe_load_checkpoint(ckpt, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state.get("epoch")


@st.cache_resource(show_spinner="Loading model (untrained fallback)...")
def load_untrained_model(variant: str):
    model = resolve_variant(variant, pretrained=True)
    model.eval()
    return model


def checkpoint_status(variant: str) -> str:
    model, epoch = load_model(variant)
    if model is None:
        return "no trained checkpoint (run scripts/train_one.py --variant {})".format(variant)
    return f"checkpoint ready (epoch {epoch})" if epoch is not None else "checkpoint ready"


@st.cache_data(show_spinner=False)
def load_model_selection_manifest(variant: str) -> dict | None:
    manifest_path = ROOT / "artifacts" / variant / "model_selection_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open() as f:
        return json.load(f)


@st.cache_data(show_spinner="Running AI inference...")
def analyze_detection(source_kind: str, variant: str, source_bytes: bytes,
                      patch_id: str, sample_fps: float, source_name: str = "",
                      debris_threshold: float = 0.65,
                      min_cluster_px: int = 4,
                      enable_tiling: bool = True,
                      use_cv_ensemble: bool = False,
                      apply_clahe: bool = True,
                      lightweight: bool = False,
                      enable_sahi: bool = False,
                      slice_size: int = 640,
                      overlap_ratio: float = 0.2,
                      enable_tracking: bool = True,
                      enable_tta: bool = False,
                      enable_spectral_gating: bool = False) -> dict:
    """Detect + concentrate any source. Returns a JSON-safe result dict."""
    # ── Fast path: Route RGB drone photo & video through TACO YOLO detector ──
    if source_kind == "drone":
        if taco_detector_available():
            taco_conf = float(st.session_state.get("taco_conf", 0.15))
            taco_imgsz = 1280 if enable_tiling else 640
            return analyze_with_taco(
                "drone", source_bytes, filename=source_name, conf=taco_conf, imgsz=taco_imgsz,
                enable_sahi=enable_sahi, slice_size=slice_size, overlap_ratio=overlap_ratio
            )
        raise ValueError(
            "⛔ TACO litter detection weights not found. "
            "RGB drone photos require YOLOv8 litter weights (run `python scripts/train_taco.py`)."
        )

    if source_kind == "video":
        if taco_detector_available():
            taco_conf = float(st.session_state.get("taco_conf", 0.15))
            taco_imgsz = 1280 if enable_tiling else 640
            return analyze_with_taco(
                "video", source_bytes, filename=source_name,
                sample_fps=sample_fps, conf=taco_conf, imgsz=taco_imgsz,
                enable_sahi=enable_sahi, slice_size=slice_size, overlap_ratio=overlap_ratio,
                enable_tracking=enable_tracking
            )
        raise ValueError(
            "⛔ TACO litter detection weights not found. "
            "RGB video frames require YOLOv8 litter weights (run `python scripts/train_taco.py`)."
        )

    # ── Satellite segmentation models (MARIDA, Plastic L2A, custom GeoTIFF) ──
    if use_cv_ensemble:
        models = load_cv_ensemble(variant)
        if not models:
            st.warning("No CV ensemble checkpoints found. Falling back to single model.")
            model, _ = load_model(variant)
            untrained = model is None
            if untrained:
                model = load_untrained_model(variant)
                st.warning(f"Running with initial weights (untrained). Train via scripts/train_one.py --variant {variant}.")
            models = [model] if model else []
        if not models:
            raise RuntimeError("No model available for ensemble inference")
    else:
        model, _ = load_model(variant)
        untrained = model is None
        if untrained:
            model = load_untrained_model(variant)
            st.warning(f"Running with initial weights (untrained). Train via scripts/train_one.py --variant {variant}.")
        models = [model] if model else []

    if not models:
        raise RuntimeError("No model available for inference")

    for m in models:
        m.to(DEVICE)
        m.eval()

    if source_kind == "marida":
        marida, _ = load_marida_and_splits()
        image = marida.read_image(patch_id)
        rgb_float = np.nan_to_num(image[[2, 1, 0]], nan=0.0).transpose(1, 2, 0)
        if use_cv_ensemble and len(models) > 1:
            det = predict_mask_ensemble(models, image.astype(np.float32), DEVICE,
                           debris_threshold=debris_threshold, min_cluster_px=min_cluster_px,
                           enable_tiling=enable_tiling, apply_clahe=apply_clahe,
                           enable_tta=enable_tta, enable_spectral_gating=enable_spectral_gating)
        else:
            det = predict_mask(models[0], image.astype(np.float32), DEVICE,
                           debris_threshold=debris_threshold, min_cluster_px=min_cluster_px,
                           enable_tiling=enable_tiling, apply_clahe=apply_clahe,
                           enable_tta=enable_tta, enable_spectral_gating=enable_spectral_gating)
        det["rgb"] = np.clip(rgb_float * 255, 0, 255).astype(np.uint8)
        det["boxes"] = boxes_with_confidence(det["debris"], det["prob"])
        det["stats"] = concentration_stats(det["debris"], det["concentration"])
        meta = marida.read_meta(patch_id)
        det["geo"] = {"transform": meta.transform, "crs": meta.crs,
                      "width": meta.width, "height": meta.height}
        det["label"] = patch_id
        return det

    if source_kind == "plastic_l2a":
        plastic = load_plastic_l2a()
        if plastic is None:
            raise ValueError("Plastic L2A dataset not prepared. Run scripts/prepare_plastic_l2a.py first.")
        image = plastic.read_image(patch_id)
        rgb_float = np.nan_to_num(image[[2, 1, 0]], nan=0.0).transpose(1, 2, 0)
        if use_cv_ensemble and len(models) > 1:
            det = predict_mask_ensemble(models, image.astype(np.float32), DEVICE,
                           debris_threshold=debris_threshold, min_cluster_px=min_cluster_px,
                           enable_tiling=enable_tiling, apply_clahe=apply_clahe,
                           enable_tta=enable_tta, enable_spectral_gating=enable_spectral_gating)
        else:
            det = predict_mask(models[0], image.astype(np.float32), DEVICE,
                           debris_threshold=debris_threshold, min_cluster_px=min_cluster_px,
                           enable_tiling=enable_tiling, apply_clahe=apply_clahe,
                           enable_tta=enable_tta, enable_spectral_gating=enable_spectral_gating)
        det["rgb"] = np.clip(rgb_float * 255, 0, 255).astype(np.uint8)
        det["boxes"] = boxes_with_confidence(det["debris"], det["prob"])
        det["stats"] = concentration_stats(det["debris"], det["concentration"])
        meta = plastic.read_meta(patch_id)
        det["geo"] = {"transform": meta.transform, "crs": meta.crs,
                      "width": meta.width, "height": meta.height}
        det["label"] = patch_id
        det["ground_truth_pending"] = not plastic.has_ground_truth
        return det

    if source_kind == "satellite":
        parsed = read_geotiff(source_bytes, apply_clahe=apply_clahe)
        if use_cv_ensemble and len(models) > 1:
            det = predict_mask_ensemble(models, parsed["model_input"], DEVICE,
                           debris_threshold=debris_threshold, min_cluster_px=min_cluster_px,
                           enable_tiling=enable_tiling, apply_clahe=apply_clahe,
                           enable_tta=enable_tta, enable_spectral_gating=enable_spectral_gating)
        else:
            det = predict_mask(models[0], parsed["model_input"], DEVICE,
                           debris_threshold=debris_threshold, min_cluster_px=min_cluster_px,
                           enable_tiling=enable_tiling, apply_clahe=apply_clahe,
                           enable_tta=enable_tta, enable_spectral_gating=enable_spectral_gating)
        det["rgb"] = parsed["rgb"]
        det["boxes"] = boxes_with_confidence(det["debris"], det["prob"])
        det["stats"] = concentration_stats(det["debris"], det["concentration"])
        det["geo"] = {"transform": parsed["transform"], "crs": parsed["crs"],
                      "width": parsed["width"], "height": parsed["height"]}
        det["label"] = parsed["filename"]
        return det

    raise ValueError(f"unknown source kind: {source_kind}")


def _drone_geo_for_upload(source_bytes: bytes, filename: str) -> dict:
    gps = extract_exif_gps(source_bytes)
    if gps:
        return {"lat": gps["lat"], "lon": gps["lon"], "gsd_m": DEFAULT_GSD_M,
                "source": "exif", "altitude_m": gps.get("altitude_m")}
    return {"lat": PLACEHOLDER_GEO["lat"], "lon": PLACEHOLDER_GEO["lon"],
            "gsd_m": PLACEHOLDER_GEO["gsd_m"], "source": "placeholder", "altitude_m": None}


def polyfill_stats(det: dict, source_kind: str, geo_inputs: dict | None) -> dict:
    stats = det["stats"] if source_kind != "video" else det["aggregate"]
    px_area = get_pixel_area(det, source_kind, geo_inputs or {}, video=(source_kind == "video"))
    
    if source_kind == "video":
        mean_density = stats.get("mean_density_pct", 0.0)
        total_debris_px = stats.get("total_debris_pixels", 0)
        n_frames = stats.get("n_frames", 1)
        frame_h = det.get("height", 1)
        frame_w = det.get("width", 1)
        tot_px = n_frames * frame_h * frame_w
        deb_px = total_debris_px
        density = mean_density
        
        if px_area is not None:
            deb_m2 = deb_px * px_area
            tot_m2 = tot_px * px_area
            return {
                "debris_pixels": deb_px,
                "total_pixels": tot_px,
                "density_pct": density,
                "debris_area_m2": deb_m2,
                "total_area_m2": tot_m2,
                "debris_per_km2_m2": (deb_m2 / (tot_m2 / 1e6)) if tot_m2 > 0 else 0.0,
                "mean_concentration": stats.get("mean_concentration", 0.0),
                "peak_concentration": stats.get("peak_concentration", 0.0),
                "severity": stats.get("severity", severity_level(density)),
            }
        return {
            "debris_pixels": deb_px,
            "total_pixels": tot_px,
            "density_pct": density,
            "debris_area_m2": None,
            "total_area_m2": None,
            "debris_per_km2_m2": None,
            "mean_concentration": stats.get("mean_concentration", 0.0),
            "peak_concentration": stats.get("peak_concentration", 0.0),
            "severity": stats.get("severity", severity_level(density)),
        }

    if px_area is not None:
        deb_px = stats["debris_pixels"]
        tot_px = stats["total_pixels"]
        deb_m2 = deb_px * px_area
        tot_m2 = tot_px * px_area
        density = (deb_px / max(tot_px, 1)) * 100.0
        return {
            "debris_pixels": deb_px,
            "total_pixels": tot_px,
            "density_pct": density,
            "debris_area_m2": deb_m2,
            "total_area_m2": tot_m2,
            "debris_per_km2_m2": (deb_m2 / (tot_m2 / 1e6)) if tot_m2 > 0 else 0.0,
            "mean_concentration": stats.get("mean_concentration", 0.0),
            "peak_concentration": stats.get("peak_concentration", 0.0),
            "severity": severity_level(density),
        }
    density = (stats["debris_pixels"] / max(stats["total_pixels"], 1)) * 100.0
    return {
        "debris_pixels": stats["debris_pixels"],
        "total_pixels": stats["total_pixels"],
        "density_pct": density,
        "debris_area_m2": None,
        "total_area_m2": None,
        "debris_per_km2_m2": None,
        "mean_concentration": stats.get("mean_concentration", 0.0),
        "peak_concentration": stats.get("peak_concentration", 0.0),
        "severity": severity_level(density),
    }


def get_pixel_area(det: dict, source_kind: str, geo_inputs: dict, video: bool = False) -> float | None:
    if source_kind in ("satellite", "marida", "plastic_l2a"):
        g = det.get("geo") or {}
        return geo.pixel_area_m2_from_transform(g.get("transform"))
    gsd = (geo_inputs or {}).get("gsd_m")
    return (float(gsd) ** 2) if gsd else None


def get_geo(det: dict, source_kind: str, geo_inputs: dict) -> dict | None:
    if source_kind in ("satellite", "marida", "plastic_l2a"):
        g = det.get("geo") or {}
        if not g.get("transform"):
            return None
        return {"transform": g["transform"], "crs": g.get("crs"),
                "width": g["width"], "height": g["height"]}
    gsd = (geo_inputs or {}).get("gsd_m")
    if not gsd:
        return None
    tr, crs = geo.synthetic_meta(
        det["width"], det["height"],
        float(geo_inputs.get("center_lat", 0.0)),
        float(geo_inputs.get("center_lon", 0.0)),
        float(gsd),
    )
    return {"transform": tr, "crs": crs, "width": det["width"], "height": det["height"]}


def severity_pill(level: str) -> str:
    color, bg = SEVERITY_PILL.get(level, ("#94a3b8", "rgba(148, 163, 184, 0.15)"))
    return (
        f'<span style="background:{bg}; color:{color}; padding:4px 14px; '
        f'border-radius:20px; font-weight:700; font-size:0.85rem; border:1px solid {color}40;">● {level.upper()}</span>'
    )


def pollution_geotiff_path(label: str, variant: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in label).strip("_") or "pollution"
    safe = "_".join(part for part in safe.split("_") if part)
    return MAPS_DIR / f"{safe}_{variant}_debris_concentration.tif"


def build_concentration_meta(det: dict, source_kind: str, geo_inputs: dict) -> PatchMeta:
    g = get_geo(det, source_kind, geo_inputs)
    if g is None:
        raise ValueError("no georeference available; provide center + GSD for drone/video inputs")
    tr = tuple(g["transform"])
    crs = g["crs"] or "EPSG:4326"
    return PatchMeta(
        patch_id=det.get("label") or "upload",
        scene_id="",
        crs=crs,
        transform=tr,
        band_count=1,
        width=g["width"],
        height=g["height"],
    )


def generate_pollution_geotiff(source_kind: str, variant: str, geo_inputs: dict) -> dict:
    det = current_source_det()
    try:
        meta = build_concentration_meta(det, source_kind, geo_inputs)
    except ValueError as exc:
        st.info(str(exc))
        return {"det": det, "tiff_path": None}
    concentration = det["aggregate_concentration"] if source_kind == "video" else det["concentration"]
    label = det.get("label") or "upload"
    tiff_path = write_concentration_tiff(concentration, meta, pollution_geotiff_path(label, variant))
    return {"det": det, "tiff_path": tiff_path}


def build_overlay(
    det: dict,
    include_segmentation: bool = False,
    show_confidence: bool = False,
) -> np.ndarray:
    rgb = det.get("rgb") if det.get("rgb") is not None else det.get("rgb_thumb")
    det_w = det.get("orig_width", det.get("width"))
    det_h = det.get("orig_height", det.get("height"))
    boxes = det.get("boxes", [])
    if det_w and det_h and rgb is not None and (rgb.shape[1] != det_w or rgb.shape[0] != det_h):
        sx = rgb.shape[1] / float(det_w)
        sy = rgb.shape[0] / float(det_h)
        scaled_boxes = []
        for b in boxes:
            scaled_boxes.append({
                **b,
                "xmin": int(round(b["xmin"] * sx)),
                "xmax": int(round(b["xmax"] * sx)),
                "ymin": int(round(b["ymin"] * sy)),
                "ymax": int(round(b["ymax"] * sy)),
            })
        boxes = scaled_boxes
    base = segmentation_overlay(rgb, det["debris"]) if include_segmentation else rgb
    if det.get("source") == "taco":
        return draw_taco_boxes(base, boxes, show_confidence=show_confidence)
    return draw_boxes(base, boxes, show_confidence=show_confidence)


def build_report_geo(det: dict, source_kind: str, geo_inputs: dict, g: dict | None = None) -> dict:
    sensor = (
        "Satellite · Sentinel-2 MSI (11-band Multispectral)"
        if source_kind in ("satellite", "marida", "plastic_l2a")
        else "Aerial Drone / Coastal Vision · High-Res RGB"
    )
    if g is None:
        return {"sensor_type": sensor, "center_lat": None, "center_lon": None,
                "bounds": None, "crs": None, "pixel_area_m2": None, "resolution_m": None}
    try:
        west, south, east, north = geo.wgs84_bounds(g["width"], g["height"], g["transform"], g["crs"])
    except Exception:
        west = south = east = north = None
    area = geo.pixel_area_m2_from_transform(g["transform"])
    center_lat = center_lon = None
    if source_kind in ("drone", "video") and geo_inputs:
        center_lat = geo_inputs.get("center_lat")
        center_lon = geo_inputs.get("center_lon")
    elif west is not None and east is not None and south is not None and north is not None:
        center_lat = (south + north) / 2.0
        center_lon = (west + east) / 2.0
    return {
        "sensor_type": sensor,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "bounds": (west, south, east, north),
        "crs": g.get("crs"),
        "pixel_area_m2": area,
        "resolution_m": math.sqrt(area) if area else None,
    }


def render_report_exporter(det: dict, source_kind: str, geo_inputs: dict | None, g: dict | None,
                           variant: str, key: str) -> None:
    st.divider()
    st.markdown("### 📄 Export Official Pollution Incident Report")
    st.caption("Generates a publication-quality compliance PDF including geospatial telemetry, debris crops, and severity analytics.")
    try:
        pdf_bytes = generate_pdf_report(
            det=det,
            model_name=HYBRID_LABELS.get(variant, variant),
            geo_data=build_report_geo(det, source_kind, geo_inputs, g),
            source_name=det.get("label") or "marine_target",
            variant=variant,
            registry=REPORT_HISTORY,
        )
    except Exception as exc:
        st.error(f"Could not render the PDF report: {exc}")
        return
    label = (det.get("label") or "marine_target").replace("\\", "_").replace("/", "_").replace(":", "_")
    st.download_button(
        "📥 Download Formal Incident Report (PDF)",
        data=pdf_bytes,
        file_name=f"Marine_Pollution_Report_{label}.pdf",
        mime="application/pdf",
        type="primary",
        key=f"pdf_report_{key}",
    )


# ── TAB 1: Source Ingestion ──────────────────────────────────────────────────
def render_source_tab() -> None:
    st.markdown("### Observation Source")
    st.caption("Select satellite Earth Observation scenes or high-resolution aerial imagery.")

    if "source_kind" not in st.session_state:
        st.session_state["source_kind"] = list(SOURCE_CODES.values())[0]

    current_mode = st.session_state.get("source_kind")
    display_key = next((k for k, v in SOURCE_CODES.items() if v == current_mode), list(SOURCE_CODES.keys())[0])

    mode = st.radio(
        "Observation Platform",
        list(SOURCE_CODES.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="source_radio",
    )
    kind = SOURCE_CODES[mode]
    st.session_state["source_kind"] = kind

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    if kind == "satellite":
        st.markdown("##### Sentinel-2 GeoTIFF (.tif)")
        up = st.file_uploader("Upload multispectral GeoTIFF", type=["tif", "tiff"], label_visibility="collapsed")
        
        sample_choice = st.selectbox(
            "Load pre-configured reference scene",
            [
                "Custom Upload",
                "26-2-18_16PCC_18.tif · Confirmed Marine Debris",
                "12-12-20_16PCC_26.tif · Official Test Scene",
                "9-10-17_16PEC_44.tif · Coastal Reference Scene",
            ],
            key="sat_sample_choice",
        )
        if sample_choice != "Custom Upload" and up is None:
            filename = sample_choice.split(" · ")[0]
            sample_path = ROOT / "data" / "marida" / "data" / filename
            if sample_path.exists():
                st.session_state["source_bytes"] = sample_path.read_bytes()
                st.session_state["source_name"] = filename

        if up is not None:
            st.session_state["source_bytes"] = up.getvalue()
            st.session_state["source_name"] = up.name

        if st.session_state.get("source_bytes"):
            try:
                parsed = read_geotiff(st.session_state["source_bytes"], st.session_state.get("source_name", "upload"))
                st.success(
                    f"Target Loaded: `{st.session_state.get('source_name', 'upload')}` | "
                    f"{parsed['width']}×{parsed['height']} px | {parsed['band_count']} Spectral Bands | CRS: `{parsed['crs'] or 'Native'}`"
                )
            except Exception as exc:
                st.error(f"Could not parse GeoTIFF: {exc}")

    elif kind == "marida":
        marida, splits = load_marida_and_splits()
        patches_with_debris = debris_test_patches()
        st.markdown("##### Official MARIDA Benchmark Scene")
        c_p, c_meta = st.columns([3, 1])
        patch_id = c_p.selectbox(
            "MARIDA Test Patches (Confirmed Debris)",
            patches_with_debris,
            key="marida_select",
        )
        st.session_state["patch_id"] = patch_id
        meta = marida.read_meta(patch_id)
        c_meta.metric("Scene Reference", marida.read_scene_id(patch_id))
        st.caption(f"Extent: {meta.width}×{meta.height} px · CRS: `{meta.crs}` · 11 Spectral Bands (10m Native)")

    elif kind == "plastic_l2a":
        plastic = load_plastic_l2a()
        if plastic is None:
            st.warning("Plastic L2A dataset not found. Run `scripts/prepare_plastic_l2a.py` first.")
            return
        st.markdown("##### ESA Plastic Litter Project (PLP) Tiles")
        patch_id = st.selectbox(
            "Sentinel-2 Target Tiles",
            list(plastic.ids),
            format_func=lambda p: f"{p} ({plastic.read_meta(p).transform[2]:.0f}E, {plastic.read_meta(p).transform[5]:.0f}N)",
        )
        st.session_state["patch_id"] = patch_id
        st.caption(f"Scene: {plastic.read_scene_id(patch_id)} · CRS: EPSG:32635 · 10m Ground Resolution")

    elif kind == "drone":
        st.markdown("##### High-Resolution Aerial / Drone Photography (.jpg / .png)")
        up = st.file_uploader("Upload aerial imagery", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if up is not None:
            st.session_state["source_bytes"] = up.getvalue()
            st.session_state["source_name"] = up.name
            
            geo_defaults = _drone_geo_for_upload(up.getvalue(), up.name)
            for key, widget in (("lat", "map_lat"), ("lat", "conc_lat"),
                                ("lon", "map_lon"), ("lon", "conc_lon"),
                                ("gsd_m", "map_gsd"), ("gsd_m", "conc_gsd")):
                st.session_state[widget] = geo_defaults[key]
            if geo_defaults["source"] == "exif":
                alt = geo_defaults["altitude_m"]
                alt_txt = f", Altitude: {alt:.1f}m" if alt is not None else ""
                st.success(f"EXIF Georeference: Lat {geo_defaults['lat']:.5f}°, Lon {geo_defaults['lon']:.5f}°{alt_txt}")
            else:
                c_lat, c_lon = st.columns(2)
                st.session_state["map_lat"] = c_lat.number_input("Target Latitude (°)", value=14.5234, format="%.5f")
                st.session_state["map_lon"] = c_lon.number_input("Target Longitude (°)", value=39.8765, format="%.5f")

    elif kind == "video":
        st.markdown("##### Coastal / UAV Video Stream (.mp4 / .avi)")
        up = st.file_uploader("Upload video footage", type=["mp4", "avi", "mov"], label_visibility="collapsed")
        if up is not None:
            st.session_state["source_bytes"] = up.getvalue()
            st.session_state["source_name"] = up.name
            st.success(f"Video Stream: `{up.name}` ({len(up.getvalue()):,} bytes)")
        st.session_state["sample_fps"] = st.slider(
            "Sampling Frequency (FPS)", min_value=0.5, max_value=4.0, value=2.0, step=0.5,
            help="Frame extraction rate for temporal aggregation."
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.button(
        "Run Analysis",
        type="primary",
        key="analyze_btn",
        help="Executes neural detection pipeline, calculates concentration metrics, and generates spatial artifacts.",
    )
    if st.session_state.get("analyze_btn"):
        run_analyze_jump()
    render_analysis_result()


# ── TAB 2: Analysis Parameters ───────────────────────────────────────────────
def render_model_tab() -> None:
    st.markdown("### Analysis Configuration")
    st.caption("Neural network architectures and detection threshold calibration.")

    c_sat, c_uav = st.columns(2)

    with c_sat:
        st.markdown("##### Satellite Engine (Sentinel-2)")
        variant = st.selectbox(
            "Segmentation Architecture",
            HYBRID_VARIANTS,
            format_func=lambda v: HYBRID_LABELS[v],
            help="Select the deep learning segmentation backbone for 11-channel Sentinel-2 MSI data."
        )
        st.session_state["variant"] = variant

        c_th, c_cl = st.columns(2)
        debris_thresh = c_th.slider(
            "Confidence Cutoff (τ)",
            min_value=0.30, max_value=0.95, value=float(st.session_state.get("debris_thresh", 0.65)), step=0.05,
            help="Probability threshold for satellite marine debris classification.",
        )
        min_px = c_cl.slider(
            "Min Cluster Size (px)",
            min_value=1, max_value=10, value=int(st.session_state.get("min_cluster_px", 4)), step=1,
            help="Filters out isolated single-pixel noise spikes.",
        )
        st.session_state["debris_thresh"] = debris_thresh
        st.session_state["min_cluster_px"] = min_px

        enable_spectral_gating = st.checkbox(
            "Spectral Verification (FDI / Foam Rejection)",
            value=st.session_state.get("enable_spectral_gating", True),
            help="Applies Biermann Floating Debris Index and spectral foam filtering to reject whitecaps.",
        )
        st.session_state["enable_spectral_gating"] = enable_spectral_gating

        enable_tiling = st.checkbox(
            "High-Resolution Tiling (10m Native)",
            value=st.session_state.get("enable_tiling", True),
            help="Processes satellite scenes as overlapping tiles with Hann-window blending.",
        )
        st.session_state["enable_tiling"] = enable_tiling

    with c_uav:
        st.markdown("##### Aerial Drone & Video Engine (RGB)")
        st.text_input("Object Detector", value="YOLOv8s · TACO Marine Litter", disabled=True)

        taco_conf = st.slider(
            "Litter Detection Threshold (τ_litter)",
            min_value=0.05, max_value=0.80, value=float(st.session_state.get("taco_conf", 0.15)), step=0.05,
            help="Confidence threshold for YOLOv8 litter detection (Bottles, Bags, Containers).",
        )
        st.session_state["taco_conf"] = taco_conf

        enable_sahi = st.checkbox(
            "Sliced Inference (SAHI High-Recall)",
            value=st.session_state.get("enable_sahi", True),
            help="Slices high-res drone images into overlapping tiles to detect small litter items.",
        )
        st.session_state["enable_sahi"] = enable_sahi

        enable_tracking = st.checkbox(
            "Video Motion Stabilization (ByteTrack)",
            value=st.session_state.get("enable_tracking", True),
            help="Maintains temporal tracking across video frames to prevent wave flicker.",
        )
        st.session_state["enable_tracking"] = enable_tracking

    with st.expander("Advanced Parameters & Validation Reference", expanded=False):
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            st.markdown("###### Processing Options")
            use_clahe = st.checkbox(
                "Apply CLAHE Contrast Equalization",
                value=st.session_state.get("apply_clahe", True),
            )
            st.session_state["apply_clahe"] = use_clahe

            enable_tta = st.checkbox(
                "8-Fold Test-Time Augmentation (TTA)",
                value=st.session_state.get("enable_tta", False),
            )
            st.session_state["enable_tta"] = enable_tta

            slice_size = st.slider("SAHI Slice Dimension (px)", 320, 1280, int(st.session_state.get("slice_size", 640)), 64)
            st.session_state["slice_size"] = slice_size
            overlap_ratio = st.slider("SAHI Overlap Fraction", 0.1, 0.4, float(st.session_state.get("overlap_ratio", 0.2)), 0.05)
            st.session_state["overlap_ratio"] = overlap_ratio

        with c_adv2:
            st.markdown("###### Validation Benchmarks")
            metrics_path = ROOT / "artifacts" / variant / "test_metrics.json"
            if metrics_path.exists():
                m = json.loads(metrics_path.read_text())
                c_m1, c_m2 = st.columns(2)
                c_m1.metric("Validation mIoU", f"{m['mean_iou']:.4f}")
                c_m2.metric("Overall Accuracy", f"{m['accuracy'] * 100:.1f}%")
                c_m1.metric("Debris Class F1", f"{m['f1_class_1_marine_debris']:.4f}")
                c_m2.metric("Debris IoU", f"{m['iou_per_class'].get('Marine Debris', 0.0):.4f}")
            if TACO_METRICS_PATH.exists():
                try:
                    tm = json.loads(TACO_METRICS_PATH.read_text())
                    st.caption(f"YOLOv8s TACO: mAP50 = {tm.get('mAP50', 0.0):.4f} | Recall = {tm.get('recall', 0.0):.4f}")
                except Exception:
                    pass


def current_source_det() -> dict:
    kind = st.session_state.get("source_kind")
    variant = st.session_state.get("variant")
    if not kind or not variant:
        raise ValueError("choose a source (tab 1) and a model (tab 2) first")
    patch_id = st.session_state.get("patch_id", "")
    source_bytes = st.session_state.get("source_bytes", b"")
    if kind in ("satellite", "drone", "video") and not source_bytes:
        raise ValueError("upload a file in tab 1 first")
    if kind in ("marida", "plastic_l2a") and not patch_id:
        raise ValueError("select a dataset patch in tab 1 first")
    sample_fps = float(st.session_state.get("sample_fps", 2.0))
    debris_threshold = float(st.session_state.get("debris_thresh", 0.65))
    min_cluster_px = int(st.session_state.get("min_cluster_px", 4))
    enable_tiling = st.session_state.get("enable_tiling", True)
    use_cv_ensemble = st.session_state.get("use_cv_ensemble", False)
    apply_clahe = st.session_state.get("apply_clahe", True)
    enable_sahi = st.session_state.get("enable_sahi", True)
    slice_size = int(st.session_state.get("slice_size", 640))
    overlap_ratio = float(st.session_state.get("overlap_ratio", 0.2))
    enable_tracking = st.session_state.get("enable_tracking", True)
    enable_tta = st.session_state.get("enable_tta", False)
    enable_spectral_gating = st.session_state.get("enable_spectral_gating", True)
    return analyze_detection(kind, variant, source_bytes, patch_id, sample_fps,
                             source_name=st.session_state.get("source_name", ""),
                             debris_threshold=debris_threshold,
                             min_cluster_px=min_cluster_px,
                             enable_tiling=enable_tiling,
                             use_cv_ensemble=use_cv_ensemble,
                             apply_clahe=apply_clahe,
                             lightweight=(kind == "video"),
                             enable_sahi=enable_sahi,
                             slice_size=slice_size,
                             overlap_ratio=overlap_ratio,
                             enable_tracking=enable_tracking,
                             enable_tta=enable_tta,
                             enable_spectral_gating=enable_spectral_gating)


def _geo_defaults_from_session() -> dict:
    return {
        "center_lat": float(st.session_state.get("map_lat", st.session_state.get("conc_lat", PLACEHOLDER_GEO["lat"]))),
        "center_lon": float(st.session_state.get("map_lon", st.session_state.get("conc_lon", PLACEHOLDER_GEO["lon"]))),
        "gsd_m": float(st.session_state.get("map_gsd", st.session_state.get("conc_gsd", PLACEHOLDER_GEO["gsd_m"]))),
    }


def run_analyze_jump() -> None:
    kind = st.session_state.get("source_kind")
    variant = st.session_state.get("variant")
    if not kind or not variant:
        st.error("Choose a source (tab 1) and a model (tab 2) first.")
        return
    try:
        det = current_source_det()
    except ValueError as exc:
        st.error(str(exc))
        return
    geo_inputs = _geo_defaults_from_session() if kind in ("drone", "video") else None
    stats = polyfill_stats(det, kind, geo_inputs)
    out = generate_pollution_geotiff(kind, variant, geo_inputs or {})
    st.session_state["_analysis_result"] = {
        "kind": kind,
        "variant": variant,
        "stats": stats,
        "boxes": len(det["boxes"]) if kind != "video" else det["aggregate"]["frames_with_debris"],
        "tiff_path": str(out["tiff_path"].relative_to(ROOT)) if out.get("tiff_path") else None,
        "label": out["det"].get("label") or "upload",
    }
    st.session_state["_go_to_map"] = True
    st.rerun()


def render_analysis_result() -> None:
    result = st.session_state.get("_analysis_result")
    if not result:
        return
    stats = result["stats"]
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📊 Rapid Telemetry Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Debris Density", f"{stats['density_pct']:.3f}%")
    c2.metric("Debris Pixels", f"{stats['debris_pixels']:,} px")
    if stats.get("debris_area_m2"):
        c3.metric("Physical Area", f"{stats['debris_area_m2']:,.1f} m²")
    else:
        c3.metric("Physical Area", "Requires GSD")
    c4.metric("Pollution Severity", stats['severity'])
    
    if result.get("tiff_path"):
        tiff_abs = ROOT / result["tiff_path"]
        if tiff_abs.exists():
            st.download_button(
                "💾 Download Pollution Heatmap (GeoTIFF)",
                data=tiff_abs.read_bytes(),
                file_name=tiff_abs.name,
                mime="image/tiff",
                key="pollution_tiff_download",
            )


# ── TAB 3: Detection Viewport ────────────────────────────────────────────────
def render_detection_tab() -> None:
    st.markdown("### Detection Viewport")
    st.caption("Side-by-side comparative analysis of sensor input and neural classification.")

    try:
        det = current_source_det()
    except ValueError as exc:
        st.info(f"Target Status: {exc}")
        return

    kind = st.session_state["source_kind"]
    if kind == "video":
        frames = det["frames"]
        index = st.slider("Video Timeline Scrubber", 0, len(frames) - 1, 0,
                          help="Inspect individual frames sampled from the video stream.")
        det_target = frames[index]
    else:
        det_target = det

    n_boxes = len(det_target["boxes"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected Targets", n_boxes)
    c2.metric("Surface Classification", "Debris Positive" if n_boxes > 0 else "Clear Water")
    c3.metric("Debris Pixels", f"{det_target['stats']['debris_pixels']:,} px")
    c4.metric("Pollution Severity", det_target['stats']['severity'])

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    display_rgb = det_target.get("rgb") if det_target.get("rgb") is not None else det_target.get("rgb_thumb")
    left.image(
        display_rgb,
        use_container_width=True,
        caption=f"{det_target.get('label', 'source')} · Input Observation Stream",
    )
    overlay = build_overlay(det_target, include_segmentation=False, show_confidence=False)
    right.image(
        overlay,
        use_container_width=True,
        caption=("Debris Classification Overlay"
                 if det_target.get("source") == "taco"
                 else "Debris Localization (Sentinel-2)"),
    )

    if det_target.get("source") == "taco" and det_target.get("class_counts"):
        counts = det_target["class_counts"]
        st.markdown("##### Detected Litter Inventory")
        chip_cols = st.columns(min(6, max(len(counts), 1)))
        for i, (name, count) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1])):
            chip_cols[i % 6].metric(f"{name}", f"{count} items")

    # FDI Spectral Layer inspection for satellite sources with spectral gating
    if kind in ("satellite", "marida", "plastic_l2a") and "fdi" in det_target:
        with st.expander("Spectral Inspection: Floating Debris Index (FDI Layer)", expanded=False):
            st.caption("Biermann FDI (Red Edge 2, NIR, SWIR 1) separates floating polymers from organic sea foam and whitecaps.")
            fdi_map = det_target["fdi"]
            fdi_vis = np.clip((fdi_map - fdi_map.min()) / (fdi_map.max() - fdi_map.min() + 1e-6), 0, 1)
            fdi_colored = cv2.applyColorMap((fdi_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
            fdi_colored = cv2.cvtColor(fdi_colored, cv2.COLOR_BGR2RGB)
            st.image(fdi_colored, use_container_width=True, caption="FDI Surface Reflectance Map (Elevated = Polymer Signature)")
            
            if "foam_mask" in det_target and np.any(det_target["foam_mask"]):
                foam_mask = det_target["foam_mask"]
                st.caption(f"Spectral Foam Rejection: Filtered {int(foam_mask.sum())} candidate pixels matching whitecap signatures.")


# ── TAB 4: Density Analytics ────────────────────────────────────────────────
def render_concentration_tab() -> None:
    st.markdown("### Spatial Density & Concentration")
    st.caption("Surface accumulation metrics and thermal pollution dispersion heatmaps.")

    try:
        det = current_source_det()
    except ValueError as exc:
        st.info(f"Target Status: {exc}")
        return

    kind = st.session_state["source_kind"]
    video = kind == "video"
    concentration = det["aggregate_concentration"] if video else det["concentration"]

    geo_inputs = None
    if kind in ("drone", "video"):
        with st.expander("Ground Sampling Distance (GSD) & Sensor Calibration", expanded=False):
            c1, c2, c3 = st.columns(3)
            center_lat = c1.number_input("Center Latitude", value=PLACEHOLDER_GEO["lat"], step=0.0001, format="%.4f", key="conc_lat")
            center_lon = c2.number_input("Center Longitude", value=PLACEHOLDER_GEO["lon"], step=0.0001, format="%.4f", key="conc_lon")
            gsd_m = c3.number_input("Ground Resolution (m/px)", value=PLACEHOLDER_GEO["gsd_m"], min_value=0.001, step=0.01, key="conc_gsd")
            geo_inputs = {"center_lat": center_lat, "center_lon": center_lon, "gsd_m": gsd_m}

    stats = polyfill_stats(det, kind, geo_inputs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Debris Density", f"{stats['density_pct']:.3f}%")
    c2.metric("Total Debris Pixels", f"{stats['debris_pixels']:,} px")
    if stats.get("debris_area_m2"):
        c3.metric("Estimated Physical Area", f"{stats['debris_area_m2']:,.1f} m²")
    else:
        c3.metric("Estimated Physical Area", "Requires GSD")
    c4.metric("Peak Accumulation Index", f"{stats['peak_concentration']:.3f}")

    st.markdown(f"**Severity Tier:** {severity_pill(stats['severity'])}", unsafe_allow_html=True)
    st.caption("Calibrated against published marine debris survey benchmarks (Topouzelis et al. 2020, Biermann et al. 2020 MARIDA). Low < 0.5% | Moderate 0.5–2.5% | High 2.5–5% | Critical > 5.0%.")

    if stats.get("debris_per_km2_m2") is not None:
        c1b, c2b = st.columns(2)
        c1b.metric("Debris Density per km²", f"{stats['debris_per_km2_m2']:,.1f} m² / km²")
        if stats.get("debris_area_m2"):
            c2b.metric("Spatial Ground Resolution", f"{stats['debris_area_m2'] / max(stats['debris_pixels'], 1):.2f} m²/px")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    c_hm, _ = st.columns([2, 1])
    c_hm.image(geo.concentration_png(concentration), use_container_width=True,
               caption="Debris Concentration Heatmap (Normalized Dispersion Scale)")

    render_report_exporter(det, kind, geo_inputs, get_geo(det, kind, geo_inputs),
                           st.session_state.get("variant", ""), "concentration")


# ── TAB 5: Geospatial Mapping & Incident Reporting ───────────────────────────
def render_map_tab() -> None:
    st.markdown("### Geospatial Mapping & Incident Reporting")
    st.caption("Georeferenced satellite mapping with GIS bounds and formal compliance documentation.")

    try:
        det = current_source_det()
    except ValueError as exc:
        st.info(f"Target Status: {exc}")
        return

    kind = st.session_state["source_kind"]
    video = kind == "video"
    geo_inputs = None
    if kind in ("drone", "video"):
        col1, col2, col3 = st.columns(3)
        center_lat = col1.number_input("Center Latitude", value=PLACEHOLDER_GEO["lat"], step=0.0001, format="%.4f", key="map_lat")
        center_lon = col2.number_input("Center Longitude", value=PLACEHOLDER_GEO["lon"], step=0.0001, format="%.4f", key="map_lon")
        gsd_m = col3.number_input("GSD (m/px)", value=PLACEHOLDER_GEO["gsd_m"], min_value=0.001, step=0.01, key="map_gsd")
        geo_inputs = {"center_lat": center_lat, "center_lon": center_lon, "gsd_m": gsd_m}

    g = get_geo(det, kind, geo_inputs)
    if g is None:
        st.warning("No georeference available. For drone or video uploads, please provide center latitude, longitude, and GSD above.")
        return

    lat = st.session_state.get("map_lat", st.session_state.get("conc_lat", float("nan")))
    lon = st.session_state.get("map_lon", st.session_state.get("conc_lon", float("nan")))
    if math.isnan(lat) or math.isnan(lon):
        st.info("No GPS coordinates available for this file. Enter coordinates in Tab 1 or above to display geographic mapping.")
        return

    concentration = det["aggregate_concentration"] if video else det["concentration"]
    severity = det["aggregate"]["severity"] if video else det["stats"]["severity"]

    c_tile, c_heat = st.columns([3, 1])
    tile = c_tile.radio("Basemap Layer", ["Esri World Imagery", "OpenStreetMap"], horizontal=True)
    heatmap_on = c_heat.checkbox("Show Dispersion Heatmap Layer", value=True)

    m = geo.build_pollution_map(
        g["width"], g["height"], g["transform"], g["crs"],
        concentration, severity, tile=tile, heatmap_on=heatmap_on,
    )
    st_folium(m, width=1100, height=540, returned_objects=[], key=f"pollution_map_{kind}_{heatmap_on}")

    west, south, east, north = geo.wgs84_bounds(g["width"], g["height"], g["transform"], g["crs"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Western Boundary", f"{west:.5f}°")
    c2.metric("Southern Boundary", f"{south:.5f}°")
    c3.metric("Eastern Boundary", f"{east:.5f}°")
    c4.metric("Northern Boundary", f"{north:.5f}°")

    render_report_exporter(det, kind, geo_inputs, g, st.session_state.get("variant", ""), "map")


# ── MAIN APPLICATION ────────────────────────────────────────────────────────
def main() -> None:
    inject_custom_styles()

    # ── Sidebar Mission Workspace ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Operational Workspace")
        st.caption("Marine Debris Surveillance Platform")
        st.divider()

        st.markdown("##### System Status")
        gpu_active = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_active else "CPU"
        st.markdown(
            f"<span class='telemetry-chip status-green'>● Ready</span> "
            f"<span class='telemetry-chip status-blue'>● {gpu_name}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        active_sensor = st.session_state.get("source_kind", "satellite").upper()
        active_model = HYBRID_LABELS.get(st.session_state.get("variant", "attention"), "U-Net + Attention")
        active_target = st.session_state.get("source_name", st.session_state.get("patch_id", "Not Loaded"))

        st.markdown(f"**Target:** `{active_target}`")
        st.markdown(f"**Sensor Mode:** `{active_sensor}`")
        st.markdown(f"**Active Backbone:** `{active_model.split('·')[0].strip()}`")

        st.divider()
        with st.expander("Hardware & Execution Environment", expanded=False):
            st.caption(f"Torch Version: `{torch.__version__}`")
            st.caption(f"CUDA Available: `{torch.cuda.is_available()}`")
            if gpu_active:
                st.caption(f"VRAM Allocated: `{torch.cuda.memory_allocated() / (1024**2):.1f} MB`")
            st.caption(f"Working Directory: `{ROOT.name}`")

    # ── Executive Masthead ──────────────────────────────────────────────────
    st.markdown(
        """
        <div style="border-bottom: 1px solid #1e293b; padding-bottom: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h1 style="margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; color: #f8fafc;">
                        Maritime Debris Surveillance System
                    </h1>
                    <p style="margin: 3px 0 0 0; color: #94a3b8; font-size: 0.85rem;">
                        Operational Remote Sensing & Multi-Scale Marine Litter Telemetry · Sentinel-2 MSI & UAV
                    </p>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="telemetry-chip status-green">● Pipeline Operational</span>
                    <span class="telemetry-chip status-blue">● NVIDIA CUDA</span>
                    <span class="telemetry-chip status-slate">● Standard Release</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.pop("_go_to_map", False):
        st.session_state[TAB_NAV_KEY] = MAP_TAB_LABEL

    tab_source, tab_model, tab_detect, tab_map = st.tabs(
        [
            "1. Source Ingestion",
            "2. Analysis Parameters",
            "3. Detection Viewport",
            "4. Geospatial Map",
        ],
        key=TAB_NAV_KEY,
    )

    with tab_source:
        render_source_tab()
    with tab_model:
        render_model_tab()
    with tab_detect:
        render_detection_tab()
    with tab_map:
        render_map_tab()


if __name__ == "__main__":
    main()