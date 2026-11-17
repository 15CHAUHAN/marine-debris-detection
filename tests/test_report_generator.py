"""Tests for the official PDF pollution incident report generator."""
import base64
import io
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from reportlab.lib.units import mm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.report_generator import (  # noqa: E402
    ReportRegistry,
    _class_breakdown,
    _crunch,
    _format_coord,
    _severity_from_density,
    generate_pdf_report,
)


def _try_decode(data: bytes) -> bytes | None:
    """Undo ReportLab stream wrappers: Adobe ASCII85 then FlateDecode."""
    cleaned = data.rstrip()
    if cleaned.endswith(b"~>"):
        cleaned = cleaned[:-2]
    try:
        binary = base64.a85decode(cleaned, adobe=False)
    except Exception:
        binary = cleaned
    try:
        return zlib.decompress(binary)
    except zlib.error:
        return binary if b"Tj" in binary else None


def _pdf_text(pdf: bytes) -> str:
    """Decompress every content stream and return the join (dependency-free)."""
    chunks = []
    for match in re.finditer(rb"stream\r?\n", pdf):
        start = match.end()
        end = pdf.find(b"endstream", start)
        if end < 0:
            continue
        decoded = _try_decode(pdf[start:end])
        if decoded is not None:
            chunks.append(decoded.decode("latin-1", errors="replace"))
    return "\n".join(chunks)


def _fake_det(fraction: float = 0.02, size: int = 64, with_boxes: bool = True) -> dict:
    rng = np.random.default_rng(7)
    rgb = rng.integers(40, 220, size=(size, size, 3)).astype(np.uint8)
    debris = (rng.random((size, size)) < fraction).astype(np.float32)
    concentration = (debris * rng.random((size, size)) + 0.02).astype(np.float32)
    det = {
        "rgb": rgb,
        "debris": debris,
        "concentration": concentration,
        "label": "12-12-20_16PCC_0",
        "frames": [],
        "aggregate": {},
    }
    if with_boxes:
        det["boxes"] = [
            {"xmin": 2, "ymin": 3, "xmax": 12, "ymax": 9, "area_px": 70, "confidence": 0.93},
            {"xmin": 20, "ymin": 20, "xmax": 30, "ymax": 26, "area_px": 60, "confidence": 0.88},
        ]
    return det


_GEO = {
    "sensor_type": "Satellite · Sentinel-2 multispectral (11-band MSI)",
    "center_lat": -12.508333,
    "center_lon": 40.262500,
    "bounds": (40.2600, -12.5170, 40.2650, -12.5000),
    "crs": "EPSG:32748",
    "pixel_area_m2": 100.0,
    "resolution_m": 10.0,
}


def test_generate_returns_valid_pdf_bytes():
    pdf = generate_pdf_report(_fake_det(fraction=0.05), model_name="U-Net + Attention",
                              geo_data=_GEO, source_name="12-12-20_16PCC_0")
    assert pdf.startswith(b"%PDF")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 5000


def test_pdf_contains_report_sections():
    pdf = generate_pdf_report(_fake_det(fraction=0.05), model_name="U-Net + Attention",
                              geo_data=_GEO, source_name="12-12-20_16PCC_0")
    text = _pdf_text(pdf)
    for token in (
        "OFFICIAL", "MARINE", "POLLUTION", "REPORT",
        "SURVEILLANCE", "MPD-", "ACQUISITION", "TELEMETRY",
        "VISUAL EVIDENCE", "QUANTITATIVE ANALYTICS",
        "OPERATIONAL RECOMMENDATION", "12-12-20_16PCC_0",
        "Sentinel-2", "EPSG:32748",
        "U-Net", "Attention",
    ):
        assert token in text, f"missing token in PDF content: {token!r}"
    assert re.search(r"Page \d+ of \d+", text) is not None, "missing page-number footer"


def test_pdf_contains_severity_directive():
    pdf = generate_pdf_report(_fake_det(fraction=0.20), model_name="baseline",
                              geo_data=_GEO, source_name="marina")
    text = _pdf_text(pdf)
    for token in ("CRITICAL", "EMERGENCY", "PROTOCOL", "ACTIVATED", "containment", "boom"):
        assert token in text, f"missing directive token in PDF: {token!r}"


def test_video_det_aggregates_frames():
    def frame(i, dpix, dens):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = 30 * (i + 1)
        debris = np.zeros((16, 16), dtype=np.float32)
        debris.flat[:dpix] = 1.0
        return {
            "rgb": rgb,
            "debris": debris,
            "concentration": np.full((16, 16), dens, dtype=np.float32),
            "boxes": [],
            "frame_index": i,
            "stats": {"debris_pixels": dpix, "density_pct": 100.0 * dpix / 256.0,
                      "mean_concentration": dens, "peak_concentration": dens},
        }

    frames = [frame(0, 10, 0.2), frame(1, 40, 0.6), frame(2, 5, 0.1)]
    det = {
        "frames": frames,
        "aggregate_concentration": np.full((16, 16), 0.4, dtype=np.float32),
    }
    info = _crunch(det, None)
    assert info is not None
    assert info["stats"]["n_frames"] == 3
    assert info["stats"]["frames_with_debris"] == 3
    assert info["stats"]["debris_pixels"] == 55
    density = 100.0 * 55 / (16 * 16 * 3)
    assert info["stats"]["density_pct"] == pytest.approx(round(density, 4))
    assert info["stats"]["severity"] == _severity_from_density(density)
    assert info["rgb"].shape == (16, 16, 3)

    pdf = generate_pdf_report(det, model_name="yolo", geo_data=_GEO, source_name="clip.mp4")
    assert pdf.startswith(b"%PDF")

    text = _pdf_text(pdf)
    for token in ("Frames sampled / total", "Frames containing debris",
                  "Peak per-frame concentration", "Worst sampled frame"):
        assert token in text, f"missing video analytics token: {token!r}"


def test_no_visual_evidence_raises():
    with pytest.raises(ValueError):
        generate_pdf_report({"label": "empty", "frames": []}, model_name="x")


def test_without_geo_still_builds():
    pdf = generate_pdf_report(_fake_det(fraction=0.01), geo_data=None, source_name="no_geo")
    assert pdf.startswith(b"%PDF")
    text = _pdf_text(pdf)
    assert "n/a" in text


def test_severity_thresholds_match_pipeline():
    assert _severity_from_density(0.0) == "Low"
    assert _severity_from_density(0.2) == "Low"
    assert _severity_from_density(0.499) == "Low"
    assert _severity_from_density(0.5) == "Moderate"
    assert _severity_from_density(1.0) == "Moderate"
    assert _severity_from_density(2.499) == "Moderate"
    assert _severity_from_density(2.5) == "High"
    assert _severity_from_density(4.999) == "High"
    assert _severity_from_density(5.0) == "Critical"
    assert _severity_from_density(100.0) == "Critical"


def test_format_coord_hemispheres_and_none():
    assert _format_coord(12.5, "N", "S").endswith("N")
    assert _format_coord(-12.5, "N", "S").endswith("S")
    assert _format_coord(40.25, "E", "W").endswith("E")
    assert _format_coord(40.25, "E", "W").endswith("E")
    assert _format_coord(None, "E", "W") == "Location not provided"
    assert _format_coord(float("nan"), "E", "W") == "Location not provided"
    assert _format_coord("oops", "E", "W") == "Location not provided"


def test_crunch_none_when_rgb_missing_and_no_frames():
    assert _crunch({"concentration": np.zeros((4, 4))}, {}) is None


def test_class_breakdown_from_mask():
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[32:, 32:] = 10  # Shallow Water in the bottom-right quadrant
    rows = _class_breakdown({"mask": mask})
    assert rows is not None
    names = [name for name, _ in rows]
    assert names == ["Marine Debris", "Shallow Water"]
    assert rows[0][1] == pytest.approx(75.0)
    assert rows[1][1] == pytest.approx(25.0)
    assert _class_breakdown({"mask": None}) is None


@pytest.mark.requires_data
def test_scene_content_and_provenance_sections():
    det = _fake_det(fraction=0.05)
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[4:8, 4:8] = 4  # index 4 -> CLASS_NAMES[5] == "Ship"
    det["mask"] = mask
    pdf = generate_pdf_report(det, model_name="U-Net + Attention",
                              geo_data=_GEO, source_name="12-12-20_16PCC_0")
    text = _pdf_text(pdf)
    for token in ("SCENE CONTENT", "DATA PROVENANCE", "SHA-256",
                  "MARIDA official test split", "2020-12-12",
                  "Annotator agreement", "Ship", "Marine Debris"):
        assert token in text, f"missing provenance token: {token!r}"


@pytest.mark.requires_data
def test_model_qa_section_from_variant():
    pdf = generate_pdf_report(_fake_det(fraction=0.02), model_name="U-Net + Attention",
                              geo_data=_GEO, source_name="12-12-20_16PCC_0",
                              variant="attention")
    text = _pdf_text(pdf)
    for token in ("MODEL", "QUALITY ASSURANCE", "Architecture",
                  "Test mean IoU", "Test accuracy",
                  "Marine Debris F1", "Marine Debris IoU"):
        assert token in text, f"missing QA token: {token!r}"


def test_analytics_conversion_rows():
    pdf = generate_pdf_report(_fake_det(fraction=0.05), model_name="baseline",
                              geo_data=_GEO, source_name="12-12-20_16PCC_0")
    text = _pdf_text(pdf)
    for token in ("Surface area analyzed", "Debris areal density",
                  "Largest debris cluster"):
        assert token in text, f"missing analytics token: {token!r}"


def test_inventory_appendix_with_boxes():
    pdf = generate_pdf_report(_fake_det(fraction=0.05), model_name="baseline",
                              geo_data=_GEO, source_name="clip")
    text = _pdf_text(pdf)
    for token in ("DETECTED CLUSTER INVENTORY", "x-range", "y-range",
                  "Confidence", "#1", "#2"):
        assert token in text, f"missing inventory token: {token!r}"


def test_inventory_note_when_no_boxes():
    det = _fake_det(fraction=0.0, with_boxes=False)
    pdf = generate_pdf_report(det, model_name="baseline", geo_data=None,
                              source_name="clean_water")
    text = _pdf_text(pdf)
    assert "No discrete debris clusters were boxed in this scan." in text


def test_histogram_and_map_panels_present():
    pdf = generate_pdf_report(_fake_det(fraction=0.06), model_name="baseline",
                              geo_data=_GEO, source_name="12-12-20_16PCC_0")
    text = _pdf_text(pdf)
    for token in ("CONCENTRATION DISTRIBUTION",
                  "D - Georeferenced Pollution Map",
                  "C - Gaussian Concentration Heatmap"):
        assert token in text, f"missing panel/section token: {token!r}"


def test_qr_record_section():
    pdf = generate_pdf_report(_fake_det(fraction=0.02), model_name="baseline",
                              geo_data=_GEO, source_name="marina")
    text = _pdf_text(pdf)
    for token in ("MACHINE-READABLE REPORT RECORD",
                  "Incident reference", "MPD-", "Surface fingerprint"):
        assert token in text, f"missing QR token: {token!r}"


def test_qr_flowable_is_full_size_not_a_dot():
    from dashboard.report_generator import _QrFlowable

    qr = _QrFlowable("MPD-RECORD|MPD-2026-ABCD|site|coord|deadbeef", size=26 * mm)
    assert qr.width == pytest.approx(26 * mm)
    # A degenerate 1.5pt barcode would be invisible; require a real module grid.
    assert qr._drawing.width > 20 * mm
    assert qr._drawing.height > 20 * mm


def test_repeat_pass_registry_records_and_delta(tmp_path):
    registry = ReportRegistry(tmp_path / "history.json")
    geo = {**_GEO, "center_lat": 10.25, "center_lon": -60.5}
    det = _fake_det(fraction=0.05)
    first = generate_pdf_report(det, model_name="baseline", geo_data=geo,
                                source_name="5-5-24_SITE_3", registry=registry)
    assert "first registered scan" in _pdf_text(first)
    assert registry.path.exists()

    second = generate_pdf_report(_fake_det(fraction=0.08), model_name="baseline",
                                 geo_data=geo, source_name="5-5-24_SITE_3",
                                 registry=registry)
    text = _pdf_text(second)
    for token in ("REPEAT-PASS SURVEILLANCE", "Baseline scan date",
                  "Density change", "Scans on record"):
        assert token in text, f"missing repeat-pass token: {token!r}"

    rows = json.loads(registry.path.read_text())
    assert len(rows) == 2
    assert {r["site_key"] for r in rows} == {"coord:10.2500,-60.5000"}
    assert rows[0]["density_pct"] < rows[1]["density_pct"]
    assert rows[0]["acquired_on"] == "2024-05-05"


def test_evidence_panels_are_aligned_identical_boxes():
    from dashboard.report_generator import _evidence_strip, _styles
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Table

    det = _fake_det(fraction=0.03, size=96)
    info = _crunch(det, None)
    assert info is not None
    flowables = _evidence_strip(info, _GEO, _styles())
    widths, heights, cells_in_row = [], [], None
    for flowable in flowables:
        if isinstance(flowable, Table):
            assert len(flowable._cellvalues) == 1, "evidence rows must stay 1xN"
            assert len(flowable._cellvalues[0]) == 2, "each row must hold 2 panels"
            for cell in flowable._cellvalues[0]:
                if cells_in_row is None:
                    cells_in_row = len(cell)
                assert len(cell) == cells_in_row
                for nested in cell:
                    if isinstance(nested, RLImage):
                        widths.append(nested.drawWidth)
                        heights.append(nested.drawHeight)
    assert cells_in_row == 3, "each panel cell must stack caption + spacer + image"
    assert len(widths) == 4
    assert len(set(widths)) == 1, f"panel widths differ: {widths}"
    assert len(set(heights)) == 1, f"panel heights differ: {heights}"


def test_fit_panel_letterboxes_into_exact_box():
    from dashboard.report_generator import _fit_panel
    from PIL import Image as PILImage

    wide = PILImage.new("RGB", (200, 100), (255, 0, 0))
    buf = io.BytesIO()
    wide.save(buf, format="PNG")
    fitted = _fit_panel(buf.getvalue(), 400, 400)
    out = PILImage.open(io.BytesIO(fitted))
    assert out.size == (400, 400)
    # Wide content is letterboxed vertically inside a navy 400x400 canvas.
    assert out.getpixel((200, 10))[:3] == (12, 18, 30)
    assert out.getpixel((200, 200))[:3] == (255, 0, 0)