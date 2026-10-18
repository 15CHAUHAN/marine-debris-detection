"""Tests for the dashboard pipeline and georeferencing modules."""
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import geo
from dashboard import pipeline
from dashboard import app


class _ZerosModel(torch.nn.Module):
    """Deterministic dummy: logits all zero -> argmax channel 0 (all debris),
    but softmax debris probability 1/11 < default threshold (gets gated out)."""

    def forward(self, x):
        return torch.zeros(x.shape[0], 11, x.shape[2], x.shape[3])


class _DebrisModel(torch.nn.Module):
    """Deterministic dummy: channel 0 (Marine Debris) dominates with high confidence."""

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 11, x.shape[2], x.shape[3])
        logits[:, 0] = 5.0
        return logits


class _OneWaterModel(torch.nn.Module):
    """Deterministic dummy: channel 6 (Marine Water) always wins, no debris."""

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 11, x.shape[2], x.shape[3])
        logits[:, 6] = 2.0
        return logits


def _jpeg_with_gps(lat_dms, lon_dms, lat_ref="N", lon_ref="E", altitude=(15, 2)) -> bytes:
    """Serialize a JPEG carrying the given GPS EXIF block."""
    from PIL import ExifTags, Image

    img = Image.new("RGB", (32, 32), (200, 150, 100))
    exif = Image.Exif()
    gps = {
        ExifTags.GPS.GPSLatitudeRef: lat_ref,
        ExifTags.GPS.GPSLatitude: lat_dms,
        ExifTags.GPS.GPSLongitudeRef: lon_ref,
        ExifTags.GPS.GPSLongitude: lon_dms,
        ExifTags.GPS.GPSAltitudeRef: 0,
        ExifTags.GPS.GPSAltitude: altitude,
    }
    exif[ExifTags.IFD.GPSInfo] = gps
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_extract_exif_gps_reads_dms_lat_lon_and_altitude():
    data = _jpeg_with_gps((12, 30, 30), (40, 15, 45))
    gps = pipeline.extract_exif_gps(data)
    assert gps is not None
    assert gps["has_gps"] is True
    assert gps["lat"] == pytest.approx(12.508333, abs=1e-5)
    assert gps["lon"] == pytest.approx(40.2625, abs=1e-5)
    assert gps["altitude_m"] is not None


def test_extract_exif_gps_honors_south_west_refs():
    data = _jpeg_with_gps((12, 30, 30), (40, 15, 45), lat_ref="S", lon_ref="W")
    gps = pipeline.extract_exif_gps(data)
    assert gps["lat"] == pytest.approx(-12.508333, abs=1e-5)
    assert gps["lon"] == pytest.approx(-40.2625, abs=1e-5)


def test_extract_exif_gps_missing_returns_none():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="PNG")
    assert pipeline.extract_exif_gps(buf.getvalue()) is None


def test_extract_exif_gps_corrupt_bytes_returns_none():
    assert pipeline.extract_exif_gps(b"not an image") is None


def test_extract_exif_gps_out_of_range_falls_back_to_none():
    data = _jpeg_with_gps((95, 0, 0), (40, 15, 45))
    assert pipeline.extract_exif_gps(data) is None


def test_severity_level_boundaries():
    assert pipeline.severity_level(0.0) == "Low"
    assert pipeline.severity_level(0.2) == "Low"
    assert pipeline.severity_level(0.499) == "Low"
    assert pipeline.severity_level(0.5) == "Moderate"
    assert pipeline.severity_level(1.0) == "Moderate"
    assert pipeline.severity_level(2.499) == "Moderate"
    assert pipeline.severity_level(2.5) == "High"
    assert pipeline.severity_level(4.999) == "High"
    assert pipeline.severity_level(5.0) == "Critical"
    assert pipeline.severity_level(100.0) == "Critical"


def test_concentration_stats_area_and_severity():
    debris = np.zeros((100, 100), dtype=np.float32)
    debris[:10] = 1.0
    concentration = np.full((100, 100), 0.5, dtype=np.float32)
    stats = pipeline.concentration_stats(debris, concentration, pixel_area_m2=4.0)
    assert stats["debris_pixels"] == 1000
    assert stats["total_pixels"] == 10000
    assert stats["density_pct"] == pytest.approx(10.0)
    assert stats["debris_area_m2"] == pytest.approx(4000.0)
    # 10% density is now Critical (>=5%)
    assert stats["severity"] == "Critical"


def test_concentration_stats_without_area():
    stats = pipeline.concentration_stats(np.zeros((4, 4)), np.zeros((4, 4)))
    assert stats["debris_area_m2"] is None
    assert stats["density_pct"] == 0.0


def test_aggregate_frames():
    frames = [
        {"stats": {"density_pct": 2.0, "mean_concentration": 0.1, "peak_concentration": 0.3, "debris_pixels": 10}},
        {"stats": {"density_pct": 4.0, "mean_concentration": 0.2, "peak_concentration": 0.5, "debris_pixels": 20}},
    ]
    agg = pipeline.aggregate_frames(frames, pixel_area_m2=25.0)
    assert agg["n_frames"] == 2
    assert agg["frames_with_debris"] == 2
    assert agg["mean_density_pct"] == pytest.approx(3.0)
    assert agg["peak_density_pct"] == pytest.approx(4.0)
    assert agg["total_debris_area_m2"] == pytest.approx(30 * 25.0)
    # Mean density is 3%, which is now High (>=2.5%)
    assert agg["severity"] == "High"


def test_boxes_with_confidence_sorted_by_area():
    debris = np.zeros((20, 20), dtype=np.float32)
    debris[2:5, 3:7] = 1.0
    debris[10:13, 10:13] = 1.0
    prob = np.full((20, 20), 0.8)
    boxes = pipeline.boxes_with_confidence(debris, prob)
    assert len(boxes) == 2
    assert boxes[0]["area_px"] == 12
    assert boxes[0]["xmin"] == 3 and boxes[0]["xmax"] == 6
    assert boxes[1]["area_px"] == 9
    assert all(b["confidence"] == pytest.approx(0.8) for b in boxes)


def test_boxes_without_probability_no_confidence():
    debris = np.zeros((10, 10), dtype=np.float32)
    debris[1:3, 1:3] = 1.0
    boxes = pipeline.boxes_with_confidence(debris, None)
    assert boxes[0]["confidence"] is None


def test_predict_mask_upsamples_to_any_resolution():
    model_input = np.random.default_rng(0).uniform(size=(11, 96, 128)).astype(np.float32)
    det = pipeline.predict_mask(_DebrisModel(), model_input, torch.device("cpu"))
    assert det["mask"].shape == (96, 128)
    assert det["debris"].sum() == 96 * 128  # high-confidence argmax channel 0 everywhere
    assert det["concentration"].shape == (96, 128)
    assert det["prob"].shape == (96, 128)


def test_predict_mask_no_debris_when_water_wins():
    model_input = np.random.default_rng(1).uniform(size=(11, 32, 32)).astype(np.float32)
    det = pipeline.predict_mask(_OneWaterModel(), model_input, torch.device("cpu"))
    assert det["debris"].sum() == 0.0


def test_predict_mask_gates_low_confidence_debris_and_falls_back_to_water():
    """All-zero logits argmax to debris but softmax p~1/11 < 0.65 default,
    so gating drops them and the mask falls back to Marine Water (class 6)."""
    model_input = np.random.default_rng(5).uniform(size=(11, 32, 32)).astype(np.float32)
    det = pipeline.predict_mask(_ZerosModel(), model_input, torch.device("cpu"))
    assert det["debris"].sum() == 0.0
    assert np.all(det["mask"] == 6)


def test_predict_mask_disabled_gating_restores_unconstrained_argmax():
    """debris_threshold <= 0 and min_cluster_px <= 1 restore old behavior."""
    model_input = np.random.default_rng(6).uniform(size=(11, 32, 32)).astype(np.float32)
    det = pipeline.predict_mask(_ZerosModel(), model_input, torch.device("cpu"),
                                debris_threshold=0.0, min_cluster_px=1)
    assert det["debris"].sum() == 32 * 32
    assert np.all(det["mask"] == 0)


def test_predict_mask_cluster_filter_removes_isolated_noise():
    """A high-confidence 1-2 pixel debris bloom in open water is dropped by the
    default min_cluster_px=4 while a compact 4px cluster is preserved."""

    class _SpottyModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 11, x.shape[2], x.shape[3])
            sig = torch.sigmoid
            # High-confidence debris where channel 0 input is > 0.9
            hot = (x[:, 0] > 0.9).float()
            logits[:, 0] = 5.0 * hot
            return logits

    rng = np.random.default_rng(7)
    model_input = rng.uniform(size=(11, 32, 32)).astype(np.float32)
    # Two isolated pixels...
    model_input[0, 5, 5] = 1.0
    model_input[0, 7, 7] = 1.0
    # ...and a compact 4-pixel cluster.
    model_input[0, 20:22, 20:22] = 1.0
    det = pipeline.predict_mask(_SpottyModel(), model_input, torch.device("cpu"))
    blob = det["debris"].astype(bool)
    # Only the 4px cluster survives.
    assert blob.sum() == 4
    assert blob[5, 5] == 0 and blob[7, 7] == 0
    assert blob[20:22, 20:22].all()
    # Filtered-out debris pixels fell back to Marine Water (class 6).
    assert det["mask"][5, 5] == 6 and det["mask"][7, 7] == 6


def test_read_rgb_upload_returns_11_channels():
    rgb = np.random.default_rng(2).integers(0, 256, size=(20, 25, 3)).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    parsed = pipeline.read_rgb_upload(encoded.tobytes(), filename="drone.png")
    assert parsed["model_input"].shape == (11, 20, 25)
    assert parsed["rgb"].shape == (20, 25, 3)
    assert parsed["height"] == 20 and parsed["width"] == 25


def test_read_geotiff_3band_adapts_to_11_channels():
    profile = {
        "driver": "GTiff", "height": 32, "width": 32, "count": 3,
        "dtype": "float32", "crs": "EPSG:32748",
        "transform": rasterio.transform.from_origin(10.0, 100.0, 10.0, 10.0),
    }
    data = np.random.default_rng(3).random((3, 32, 32)).astype(np.float32)
    with rasterio.io.MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data)
        parsed = pipeline.read_geotiff(mem.read(), filename="patch.tif")
    assert parsed["band_count"] == 3
    assert parsed["model_input"].shape == (11, 32, 32)
    assert parsed["rgb"].shape == (32, 32, 3)
    assert parsed["crs"] == "EPSG:32748"
    assert parsed["transform"][0] == pytest.approx(10.0)


def test_segmentation_overlay_tints_debris():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    debris = np.zeros((8, 8), dtype=np.float32)
    debris[0, 0] = 1.0
    overlay = pipeline.segmentation_overlay(rgb, debris)
    assert overlay[0, 0, 0] > overlay[0, 0, 1]
    assert overlay[5, 5, 0] == 0.0


def test_draw_boxes_annotation():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    boxes = [{"xmin": 2, "ymin": 2, "xmax": 10, "ymax": 8, "area_px": 30, "confidence": 0.9}]
    out = pipeline.draw_boxes(rgb, boxes)
    assert out.dtype == np.uint8
    assert (out[2:9, 2:3] > 0).any()


def test_synthetic_meta_centers_bounds_on_requested_point():
    tr, crs = geo.synthetic_meta(256, 256, -12.5, 40.5, 10.0)
    west, south, east, north = geo.wgs84_bounds(256, 256, tr, crs)
    assert crs == geo.WGS84
    assert abs((south + north) / 2 - (-12.5)) < 0.02
    assert abs((west + east) / 2 - 40.5) < 0.02
    assert east > west and north > south


def test_pixel_xy_to_wgs84_identity_for_wgs84_crs():
    tr, crs = geo.synthetic_meta(128, 96, 5.0, -60.0, 1.0)
    lons, lats = geo.pixel_xy_to_wgs84(128, 96, tr, crs, [64], [48])
    assert abs(lats[0] - 5.0) < 0.01
    assert abs(lons[0] - (-60.0)) < 0.01


def test_concentration_grid_shapes_and_values():
    concentration = np.full((256, 256), 0.5, dtype=np.float32)
    cols, rows, means = geo.concentration_grid(concentration, grid_n=16)
    assert means.shape == (16, 16)
    assert cols.shape == (16,) and rows.shape == (16,)
    assert np.allclose(means, 0.5)
    assert 0 <= cols.max() < 256 and 0 <= rows.max() < 256


def test_concentration_color_hex_format():
    for value in (0.0, 0.5, 1.0):
        c = geo.concentration_color(value)
        assert c.startswith("#") and len(c) == 7


def test_severity_colors_cover_levels():
    for level in ("Low", "Moderate", "High", "Critical"):
        assert geo.severity_color(level).startswith("#")


def test_corner_wgs84_is_closed_ring():
    tr, crs = geo.synthetic_meta(256, 256, 0.0, 0.0, 10.0)
    corners = geo.corner_wgs84(256, 256, tr, crs)
    assert len(corners) == 5
    assert corners[0] == corners[-1]


def test_wgs84_conversion_on_real_utm_patch():
    """Real MARIDA UTM GeoTiff metadata must warp to plausible lat/lon."""
    from src.data.ingestion import MaridaDataset

    marida = MaridaDataset(ROOT)
    patch_id = "12-12-20_16PCC_0"
    meta = marida.read_meta(patch_id)
    assert meta.crs.startswith("EPSG:")
    lons, lats = geo.pixel_xy_to_wgs84(
        meta.width, meta.height, meta.transform, meta.crs, [128], [128]
    )
    assert -180.0 < lons[0] < 180.0
    assert -90.0 < lats[0] < 90.0
    west, south, east, north = geo.wgs84_bounds(meta.width, meta.height, meta.transform, meta.crs)
    assert west < east and south < north
    assert abs((south + north) / 2 - lats[0]) < 0.5
    assert abs((west + east) / 2 - lons[0]) < 0.5


def test_pixel_area_from_transform():
    assert geo.pixel_area_m2_from_transform((10.0, 0.0, 100.0, 0.0, -10.0, 300.0)) == 100.0
    assert geo.pixel_area_m2_from_transform(None) is None


def test_wgs84_extraction_matches_rasterio_warp_reference():
    from rasterio.warp import transform as warp_coords, transform_bounds as warp_bounds

    crs = "EPSG:32631"  # UTM zone 31N
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)  # 10 m pixels
    width, height = 256, 256

    cols = [0, 128, 255]
    rows = [0, 128, 255]
    lons, lats = geo.pixel_xy_to_wgs84(width, height, transform, crs, cols, rows)

    affine = rasterio.transform.Affine(*transform)
    xs, ys = rasterio.transform.xy(affine, rows, cols, offset="center")
    ref_lons, ref_lats = warp_coords(crs, "EPSG:4326", xs, ys)
    np.testing.assert_allclose(lons, np.asarray(ref_lons, dtype=float), atol=1e-8)
    np.testing.assert_allclose(lats, np.asarray(ref_lats, dtype=float), atol=1e-8)

    assert 2.9 < lons[0] < 3.2  # near the 3E central meridian
    assert 40.5 < lats[0] < 40.8  # northing ~4.5M N ~ 40.65N

    west, south, east, north = geo.wgs84_bounds(width, height, transform, crs)
    ref = warp_bounds(
        crs,
        "EPSG:4326",
        500000.0,
        4500000.0 - height * 10.0,
        500000.0 + width * 10.0,
        4500000.0,
        densify_pts=65,
    )
    np.testing.assert_allclose((west, south, east, north), ref, atol=1e-6)
    assert west < east and south < north
    assert south <= lats[0] <= north and west <= lons[0] <= east


def test_concentration_rgba_has_alpha_and_hot_colors():
    concentration = np.zeros((16, 16), dtype=np.float32)
    concentration[4:12, 4:12] = 0.5
    rgba = geo.concentration_rgba(concentration, threshold=0.0)
    assert rgba.shape == (16, 16, 4)
    assert rgba.dtype == np.uint8
    assert (rgba[concentration > 0, 0] > rgba[concentration > 0, 2]).all()  # red > blue
    assert (rgba[concentration == 0, 3] == 0).all()  # transparent elsewhere


def test_build_pollution_map_renders_interactive_layers():
    folium = pytest.importorskip("folium")
    tr, crs = geo.synthetic_meta(256, 256, 0.0, 0.0, 10.0)
    concentration = np.random.default_rng(4).random((256, 256)).astype(np.float32)
    m = geo.build_pollution_map(256, 256, tr, crs, concentration, "Moderate")
    assert isinstance(m, folium.Map)
    html = m.get_root().render()
    assert "circleMarker" in html
    assert "imageOverlay" in html
    assert "polygon" in html.lower()
    assert "id=" in html


def test_pollution_geotiff_path_sanitizes_label():
    p1 = app.pollution_geotiff_path("12-12-20_16PCC_7", "attention")
    assert p1.name == "12-12-20_16PCC_7_attention_debris_concentration.tif"
    assert p1.is_absolute() and p1.parent.name == "maps"
    p2 = app.pollution_geotiff_path("weird: name/with\\chars", "baseline")
    assert p2.name == "weird_name_with_chars_baseline_debris_concentration.tif"


def test_build_concentration_meta_warps_geo_to_patchmeta():
    det = {
        "geo": {"transform": (10.0, 0.0, 100.0, 0.0, -10.0, 300.0),
                "crs": "EPSG:32631", "width": 256, "height": 256},
        "label": "12-12-20_16PCC_0",
    }
    meta = app.build_concentration_meta(det, "marida", {})
    assert meta.crs == "EPSG:32631"
    assert meta.transform[0] == 10.0
    assert meta.width == 256 and meta.height == 256


def test_build_concentration_meta_drone_synthetic_crs():
    det = {"width": 256, "height": 256, "label": "drone.jpg"}
    meta = app.build_concentration_meta(det, "drone", {"center_lat": -12.5, "center_lon": 40.5, "gsd_m": 0.05})
    assert meta.crs == geo.WGS84
    assert meta.width == 256 and meta.height == 256
    west, south, east, north = geo.wgs84_bounds(meta.width, meta.height, meta.transform, meta.crs)
    assert abs((south + north) / 2 - (-12.5)) < 0.02
    assert abs((west + east) / 2 - 40.5) < 0.02


def test_build_concentration_meta_no_geo_raises():
    det = {"width": 256, "height": 256}
    with pytest.raises(ValueError):
        app.build_concentration_meta(det, "drone", {})


def test_generate_pollution_geotiff_writes_tiff(tmp_path, monkeypatch):
    debris = np.zeros((32, 32), dtype=np.float32)
    debris[2:6, 2:6] = 1.0
    det = {
        "probability": np.full((32, 32), 0.0),
        "debris": debris,
        "concentration": np.full((32, 32), 0.5, dtype=np.float32),
        "mask": np.zeros((32, 32), dtype=np.int64),
        "width": 32,
        "height": 32,
        "label": "test_src",
    }
    monkeypatch.setattr(app, "current_source_det", lambda: det)
    monkeypatch.setattr(app, "MAPS_DIR", tmp_path)
    out = app.generate_pollution_geotiff("drone", "attention", {"center_lat": 0.0, "center_lon": 0.0, "gsd_m": 1.0})
    assert out["tiff_path"] is not None and out["tiff_path"].exists()
    with rasterio.open(out["tiff_path"]) as src:
        assert src.count == 1
        assert src.crs is not None
        assert src.width == 32 and src.height == 32


def test_generate_pollution_geotiff_video_uses_aggregate(tmp_path, monkeypatch):
    det = {
        "frames": [{"rgb": np.zeros((16, 16, 3), dtype=np.uint8)}],
        "width": 16,
        "height": 16,
        "label": "clip.mp4",
        "aggregate_concentration": np.full((16, 16), 0.75, dtype=np.float32),
    }
    monkeypatch.setattr(app, "current_source_det", lambda: det)
    monkeypatch.setattr(app, "MAPS_DIR", tmp_path)
    out = app.generate_pollution_geotiff("video", "baseline", {"center_lat": 0.0, "center_lon": 0.0, "gsd_m": 1.0})
    assert out["tiff_path"] is not None and out["tiff_path"].exists()
    with rasterio.open(out["tiff_path"]) as src:
        assert src.width == 16 and src.height == 16
        assert src.read(1).max() == pytest.approx(0.75)