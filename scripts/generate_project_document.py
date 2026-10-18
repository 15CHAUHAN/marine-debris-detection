"""Generate the official project documentation as a professional Word document.

Content is sourced from the real, logged project results in docs/ and
artifacts/. Produces TECH-ATHON_Marine_Debris_Detection_Project.docx in the
project root.

Usage:
    .venv\\Scripts\\python.exe scripts\\generate_project_document.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "TECH-ATHON_Marine_Debris_Detection_Project.docx"

NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x5F, 0xA3)
TEAL = RGBColor(0x0E, 0x7C, 0x7B)
GREY = RGBColor(0x59, 0x59, 0x59)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x8E, 0x3E)
HDR_FILL = "1F3864"
ALT_FILL = "EAF0F8"

_ALT_TOGGLE = False


def set_cell_bg(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, color=None, size=None, align=None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    if size is not None:
        run.font.size = size
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)


def add_table(doc, headers, rows, col_widths=None, header_fill=HDR_FILL, zebra=True) -> None:
    global _ALT_TOGGLE
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        set_cell_bg(hdr_cells[i], header_fill)
        set_cell_text(hdr_cells[i], h, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), size=Pt(9))
    for r, row in enumerate(rows):
        cells = table.add_row().cells
        for c, val in enumerate(row):
            if zebra and r % 2 == 1:
                set_cell_bg(cells[c], ALT_FILL)
            set_cell_text(cells[c], str(val), size=Pt(9))
    if col_widths is not None:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return table


def h1(doc, text) -> None:
    p = doc.add_heading(text, level=1)
    for r in p.runs:
        r.font.color.rgb = NAVY
        r.font.size = Pt(16)
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)


def h2(doc, text) -> None:
    p = doc.add_heading(text, level=2)
    for r in p.runs:
        r.font.color.rgb = BLUE
        r.font.size = Pt(13)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)


def h3(doc, text) -> None:
    p = doc.add_heading(text, level=3)
    for r in p.runs:
        r.font.color.rgb = TEAL
        r.font.size = Pt(11.5)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)


def para(doc, text, bold=False, italic=False, size=None, color=None, align=None) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size is not None:
        run.font.size = size
    if color is not None:
        run.font.color.rgb = color
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    return p


def bullets(doc, items, style="List Bullet") -> None:
    for it in items:
        p = doc.add_paragraph(it, style=style)
        p.paragraph_format.space_after = Pt(2)


def numbered(doc, items) -> None:
    for it in items:
        p = doc.add_paragraph(it, style="List Number")
        p.paragraph_format.space_after = Pt(3)


def callout(doc, text, fill="F2F7FF") -> None:
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    set_cell_bg(cell, fill)
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = Pt(10)
    run.font.color.rgb = NAVY
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    doc.add_paragraph()


def add_toc(doc) -> None:
    p = doc.add_paragraph()
    run = p.add_run()
    fld_char = OxmlElement("w:fldChar")
    fld_char.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "separate")
    t_val = OxmlElement("w:t")
    t_val.text = "Right-click and choose Update Field (or press F9) to build the table of contents."
    fld_char3 = OxmlElement("w:fldChar")
    fld_char3.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char)
    run._r.append(instr)
    run._r.append(fld_char2)
    run._r.append(t_val)
    run._r.append(fld_char3)


def title_page(doc) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

    for _ in range(6):
        doc.add_paragraph()

    para(doc, "TECH-ATHON 2026", bold=True, size=Pt(16), color=TEAL, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "AI-Based Marine Plastic Pollution Detection", bold=True, size=Pt(30), color=NAVY,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Pixel-Level Marine Debris Segmentation from Sentinel-2 Multispectral Imagery",
         italic=True, size=Pt(15), color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_paragraph()
    para(doc, "End-to-End Detection, Concentration Mapping and Operational Dashboards",
         size=Pt(12), align=WD_ALIGN_PARAGRAPH.CENTER)

    for _ in range(5):
        doc.add_paragraph()

    info = (
        ("Dataset", "MARIDA - MARIne Debris Archive (Zenodo DOI 10.5281/zenodo.5151941)"),
        ("Platform", "NVIDIA GeForce RTX 4050 (6 GB VRAM), CUDA 12.7, Python 3.12, PyTorch cu126"),
        ("Models", "U-Net family (ResNet-34, scSE attention, from-scratch CNN, EfficientNet-B0 + Transformer), "
                   "YOLO hybrids (YOLO + ViT + LSTM, YOLO + BiLSTM), YOLOv8n detector, CNN + BiLSTM classifier, WGAN-GP"),
        ("Deliverables", "Segmentation model zoo, 10-fold scene-disjoint cross-validation, GAN augmentation study, "
                         "georeferenced pollution heatmaps, static dashboard, interactive Streamlit dashboard, "
                         "official PDF pollution incident reports"),
        ("Status", "All pipeline stages complete and verified, with all metrics logged from real training runs"),
    )
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in info:
        row = table.add_row().cells
        set_cell_bg(row[0], ALT_FILL)
        set_cell_text(row[0], k, bold=True, size=Pt(10), color=NAVY)
        set_cell_text(row[1], v, size=Pt(10))
        row[0].width = Cm(4.2)
        row[1].width = Cm(12.4)

    doc.add_paragraph()
    para(doc, "Document organized by: project context and problem statement, solving approach, dataset, "
              "pipeline workflow, trained models with accuracy and outputs, GAN augmentation, 10-fold "
              "cross-validation, and the static and interactive dashboards.",
         italic=True, size=Pt(10), color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_page_break()


def main() -> None:
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    title_page(doc)

    # ------------------------------------------------------------------ 1. exec summary
    h1(doc, "1. Executive Summary")
    para(doc,
         "This project builds and validates a complete, end-to-end system for detecting floating marine "
         "plastic pollution (the Marine Debris class) in satellite and drone imagery. The MARIDA benchmark "
         "dataset provides 1,381 Sentinel-2 patches across 63 scenes. Each patch is a 256x256 pixel, 11-band "
         "multispectral image together with a per-pixel class mask and an annotator confidence mask.")
    para(doc,
         "The core scientific comparison is a controlled A/B experiment between a plain U-Net (baseline) and a "
         "U-Net augmented with spatial and channel squeeze-and-excitation (scSE) attention gates. Both models "
         "were trained under identical conditions on the official scene-disjoint train split and evaluated on "
         "the held-out official test split. The attention model is the better Marine Debris detector: "
         "F1 0.1606 vs 0.1528, debris IoU 0.0873 vs 0.0827, and pixel accuracy 0.7718 vs 0.7615, while both "
         "models are statistically tied on mean IoU (0.2971 vs 0.2975).")
    para(doc,
         "The project was extended beyond the A/B pair into a full model zoo with six segmentation "
         "architectures, a YOLOv8n box detector, a CNN + BiLSTM patch classifier, a WGAN-GP data-augmentation "
         "study, and a 10-fold scene-disjoint cross-validation. Two operational dashboards deliver the results "
         "to end users: a static HTML dashboard for reporting, and an interactive Streamlit dashboard that runs "
         "the full pipeline from any image or video input to a georeferenced pollution map and an official PDF "
         "incident report.")

    # ------------------------------------------------------------------ 2. problem statement
    h1(doc, "2. Problem Statement")
    h2(doc, "2.1 The Environmental Challenge")
    para(doc,
         "Marine plastic pollution is one of the most pressing environmental threats of this century. Floating "
         "plastic debris damages marine ecosystems, is ingested by marine life, and enters the human food chain "
         "through microplastics. Effective response requires continuous, large-scale monitoring to locate debris "
         "accumulations, estimate their concentration, and guide clean-up operations.")
    h2(doc, "2.2 Why Satellite Monitoring")
    para(doc,
         "In-situ ship surveys and visual inspection cannot cover the ocean at scale. Optical satellites such as "
         "Sentinel-2 provide systematic, recurring multispectral imagery of coastal and open waters. However, "
         "floating marine debris appears as extremely small and rare signals in satellite imagery: on average "
         "only about 0.45% of labeled pixels belong to the Marine Debris class, and individual debris clusters "
         "span only 1 to 6 pixels. Manually reviewing every image is infeasible, so automated, pixel-precise "
         "detection is required.")
    h2(doc, "2.3 Problem Statement")
    para(doc,
         "Given an 11-band Sentinel-2 multispectral patch (256x256 pixels), produce a per-pixel classification "
         "over the 11 aggregated thematic classes, with primary focus on accurately localizing Marine Debris. "
         "The system must:")
    numbered(doc, [
        "Segment Marine Debris at pixel level despite extreme class imbalance (about 0.45% of labeled pixels).",
        "Generalize across scenes, acquisitions, and coastal regions without spatial leakage between training and test data.",
        "Estimate debris concentration and export georeferenced pollution heatmaps for use in GIS tools.",
        "Operate on multiple input sources: satellite GeoTIFFs, drone images, and coastal video streams.",
        "Surface results to decision makers through reporting dashboards and formal incident reports.",
    ])
    h2(doc, "2.4 Key Technical Difficulties")
    bullets(doc, [
        "Extreme class imbalance: Marine Debris is present in only 373 of 1,381 patches and is a tiny fraction of labeled pixels.",
        "Very small debris signatures (1 to 6 pixels) that are sub-resolution for object detection.",
        "Scene variability across 63 acquisitions with differing UTM zones, water conditions, and lighting.",
        "Data integrity concerns (duplicates, near-duplicates, leakage), which must be audited before trusting any metric.",
        "7 patches contain NaN pixels and 724 patches are dominated by a single class, which must be handled safely.",
    ])

    # ------------------------------------------------------------------ 3. approach
    h1(doc, "3. How We Are Solving the Problem")
    h2(doc, "3.1 Solution Overview")
    para(doc,
         "The solution is a complete pipeline: Satellite / Drone Image / Video, then Plastic Detection "
         "(pixel-level segmentation), then Concentration estimation, then a Pollution Map. The scientific "
         "approach is a controlled A/B study expanded into a systematically compared model zoo, with every "
         "result verified through cross-validation, a leakage audit, and an overfitting analysis.")
    h2(doc, "3.2 Approach Principles")
    numbered(doc, [
        "Follow a controlled A/B testing methodology: identical training protocol, data, and seeds for every model, so differences are attributable to architecture only.",
        "Use the official scene-disjoint MARIDA split as the reporting standard to prevent spatial leakage.",
        "Apply augmentation (CLAHE + Albumentations) only to training data, never to validation or test data.",
        "Compute every reported number from real training runs; no fabricated or assumed values.",
        "Validate robustness independently with a 10-fold scene-grouped cross-validation and a data-leakage audit.",
        "Deliver operational outputs: Gaussian-convolved concentration heatmaps, georeferenced GeoTIFFs, and both static and interactive dashboards.",
    ])
    h2(doc, "3.3 Data Preparation and Label Handling")
    para(doc,
         "Patches are read directly from the GeoTiff files with rasterio and normalized band-wise to [0, 1]. "
         "The raw 15-class annotations are mapped deterministically to an 11-class working set by grouping "
         "Waves, Cloud Shadows, Wakes, and Mixed Water into the Marine Water super-class. Two label bugs were "
         "found and fixed during development (a class-weight off-by-one and a 1-based target mapping that "
         "dropped Shallow Water); all final results are from post-fix retrains.")
    h2(doc, "3.4 Model Comparison Strategy")
    para(doc,
         "Six segmentation architectures were trained and evaluated on the official test split: the plain "
         "U-Net A/B control, the scSE-attention A/B candidate, a compact from-scratch CNN + U-Net, an "
         "EfficientNet-B0 + Transformer U-Net, and two YOLO-backbone hybrid segmenters (YOLO + ViT + LSTM and "
         "YOLO + BiLSTM). Supporting experiments evaluate a YOLOv8n box detector and a CNN + BiLSTM patch "
         "classifier to demonstrate why segmentation is the correct route for this problem. Each model was "
         "additionally stress-tested with WGAN-GP synthetic augmentation to answer whether data synthesis helps "
         "or hurts a rare-class task.")
    h2(doc, "3.5 Robustness and Quality Gates")
    bullets(doc, [
        "Scene-disjoint 10-fold cross-validation brackets the single-run test result and confirms stability across re-partitioning.",
        "A data-leakage audit confirms zero exact duplicates and characterizes near-duplicate pairs as geographic adjacency, not actionable leakage.",
        "An overfitting analysis labels each variant as healthy, overfitting, or underfitting with recommended corrections.",
        "An automated environment gate and a 175-test pytest suite guard the reproducibility of every result.",
    ])

    # ------------------------------------------------------------------ 4. dataset
    h1(doc, "4. Dataset Overview")
    h2(doc, "4.1 MARIDA Benchmark")
    para(doc,
         "MARIDA (MARIne Debris Archive) is the standard benchmark for maritime debris and water-class "
         "segmentation from Sentinel-2 imagery. It was verified on disk during this project.")
    add_table(doc,
        ["Metric", "Value"],
        [
            ["Patches", "1,381 (256 x 256, 11-band GeoTiff + class mask + confidence mask)"],
            ["Scenes", "63 Sentinel-2 acquisitions (scene names like S2_1-12-19_48MYU)"],
            ["Spectral bands", "11 (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12; 60 m bands B9/B10 excluded)"],
            ["Data type", "float32, 256 x 256, UTM CRS (EPSG varies by scene/region)"],
            ["Class set", "11 aggregated classes (15 raw classes mapped; 0 marks unlabeled)"],
            ["Target class", "Marine Debris (class 1), present in 373 of 1,381 patches (~27%)"],
            ["Official split", "Scene-disjoint: train 694 / val 328 / test 359 (36 / 12 / 15 scenes, zero overlap)"],
        ],
        col_widths=[4.5, 12.0])

    h2(doc, "4.2 Aggregated Class Mapping (15 to 11)")
    add_table(doc,
        ["Raw 15-class group", "Aggregated 11-class group", "Class index (working set)"],
        [
            ["Marine Debris", "Marine Debris", "1"],
            ["Dense Sargassum", "Dense Sargassum", "2"],
            ["Sparse Sargassum", "Sparse Sargassum", "3"],
            ["Natural Organic Material", "Natural Organic Material", "4"],
            ["Ship", "Ship", "5"],
            ["Clouds", "Clouds", "6"],
            ["Marine Water", "Marine Water", "7"],
            ["Sediment-Laden Water", "Sediment-Laden Water", "8"],
            ["Foam", "Foam", "9"],
            ["Turbid Water", "Turbid Water", "10"],
            ["Shallow Water", "Shallow Water", "11"],
            ["Waves, Cloud Shadows, Wakes, Mixed Water", "Marine Water", "7"],
        ],
        col_widths=[6.0, 5.5, 5.0])

    h2(doc, "4.3 Class Presence in the Dataset")
    para(doc, "Number of patches in which each aggregated class appears (multi-label, from labels_mapping.txt):")
    add_table(doc,
        ["Class", "Patches"],
        [
            ["Marine Water", "955"],
            ["Marine Debris", "373"],
            ["Turbid Water", "220"],
            ["Ship", "182"],
            ["Clouds", "181"],
            ["Sparse Sargassum", "106"],
            ["Natural Organic Material", "71"],
            ["Shallow Water", "64"],
            ["Foam", "59"],
            ["Sediment-Laden Water", "51"],
            ["Dense Sargassum", "49"],
        ],
        col_widths=[9.0, 7.5])
    callout(doc,
        "Marine Debris is the rarest and most valuable class. Its pixel share is only about 0.45% of labeled "
        "pixels, and debris clusters are 1 to 6 pixels in size. This rarity dominates every metric and drives "
        "the design choices in the loss function, the augmentation policy, and the interpretation of results.")

    # ------------------------------------------------------------------ 5. workflow
    h1(doc, "5. Project Workflow")
    para(doc,
         "The processing chain is deliberately off-line and deterministic except for stochastic training-time "
         "augmentation. Data flows through the following stages:")
    h2(doc, "5.1 Workflow Diagram")
    para(doc,
         "patches (GeoTiff, class mask, confidence mask) -> Ingestion -> Encoding (15 to 11) -> Splits " 
         "(official scene-disjoint) -> Augmentation (CLAHE, training only) -> Training (CE + Dice, AMP) -> "
         "Evaluation (IoU, F1, accuracy, confusion matrix) -> Mapping (Gaussian concentration heatmap, "
         "georeferenced GeoTIFF) -> Dashboards (static HTML and interactive Streamlit + PDF reports).",
         italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    h2(doc, "5.2 Stage-by-Stage Description")
    add_table(doc,
        ["Stage", "Module", "What it does", "Output"],
        [
            ["1. Ingestion", "src/data/ingestion.py",
             "Reads the 11-band GeoTiff via rasterio, applies band-wise percentile normalization to [0, 1], "
             "captures CRS, transform, and resolution metadata.",
             "float32 (C, H, W) tensor + PatchMeta"],
            ["2. Encoding", "src/data/encoding.py",
             "Deterministic 15-to-11 class lookup and a 0-based training-target mapping used by every model "
             "and metric (unlabeled pixels map to an ignore index).",
             "int64 mask in the 11-class working set"],
            ["3. Splits", "src/data/splits.py",
             "Parses the official train/val/test files and derives scene IDs for grouping; provides a "
             "scene-grouped K-fold generator (GroupKFold, group = scene).",
             "Train 694 / val 328 / test 359 patch lists"],
            ["4. Augmentation", "src/data/augment.py",
             "CLAHE (clipLimit 2.0, tile 8x8) plus Albumentations spatial/color transforms applied only to "
             "training data; NaN pixels sanitized for inert patches.",
             "Augmented image + mask pairs"],
            ["5. Training", "src/train/trainer.py",
             "Cross-entropy + Dice loss with inverse class-frequency weights, AdamW optimizer, mixed precision "
             "(AMP), early stopping, class-balanced weighting.",
             "Checkpoints, per-epoch CSV/JSON history"],
            ["6. Evaluation", "src/evaluate/metrics.py",
             "Per-class and mean IoU, F1, pixel accuracy, full confusion matrix computed strictly on the "
             "official test split (labeled pixels only).",
             "test_metrics.json, confusion matrix"],
            ["7. Mapping", "src/mapping/heatmap.py, georeference.py",
             "Convolves the predicted debris mask with a Gaussian kernel (sigma ~2 px) for a concentration "
             "map, then writes it as a georeferenced float32 GeoTIFF with the source CRS and transform.",
             "artifacts/maps/*_concentration.tif + PNG"],
            ["8. Reporting", "dashboard/build + dashboard/app.py",
             "Regenerates the static HTML dashboard from artifacts and serves the interactive Streamlit "
             "pipeline app with PDF incident report export.",
             "dashboard.html, Streamlit app, PDF reports"],
        ],
        col_widths=[2.6, 3.6, 6.3, 4.0])
    h2(doc, "5.3 Training Protocol")
    add_table(doc,
        ["Parameter", "Value"],
        [
            ["Input / output shape", "(B, 11, 256, 256) -> (B, 11, 256, 256)"],
            ["Loss", "Cross-entropy + Dice, class weights from inverse class frequency (train split only)"],
            ["Optimizer", "AdamW, learning rate 1e-4"],
            ["Batch size / epochs", "8 / 20 (hybrid YOLO variants: 8 / 5)"],
            ["Precision", "Mixed precision (AMP)"],
            ["Seed", "42 for all comparable runs"],
            ["Augmentation", "CLAHE + Albumentations, training split only"],
        ],
        col_widths=[5.5, 11.0])
    h2(doc, "5.4 Mapping Output")
    bullets(doc, [
        "The mapping stage turns the predicted Marine Debris mask into a smooth concentration surface via Gaussian convolution (sigma ~2 px).",
        "The concentration GeoTIFF preserves the source patch CRS and affine transform, so it registers 1:1 with the source imagery in any GIS viewer.",
        "Verified properties: 256 x 256 float32, values in [0, 1], 21,106 unique values (smooth gradient, not a binary mask).",
    ])

    # ------------------------------------------------------------------ 6. models
    h1(doc, "6. Models Trained (Model Zoo)")
    para(doc,
         "All segmentation variants share one interface: they consume (B, 11, 256, 256) multispectral tensors "
         "and emit (B, 11, 256, 256) per-class logits. Metrics below are computed on the official test split "
         "(359 patches, labeled pixels only) using the best epoch selected on the official validation split. "
         "Percent values are shown in the tables.")

    h2(doc, "6.1 Segmentation Model Zoo Summary")
    add_table(doc,
        ["Model", "Params", "Mean IoU %", "Pixel Acc %", "Debris F1 %", "Debris IoU %", "Best val mIoU % (ep)", "Verdict"],
        [
            ["baseline: U-Net (ResNet-34)", "24.46 M", "29.75", "76.15", "15.28", "8.27", "24.31 (17)", "A/B control"],
            ["attention: U-Net + scSE", "24.58 M", "29.71", "77.18", "16.06", "8.73", "21.82 (19)", "A/B candidate, best debris model"],
            ["cnn_unet (from scratch)", "0.489 M", "14.05", "23.40", "12.88", "6.88", "12.18 (19)", "underfit"],
            ["effnet_transformer", "8.0 M", "28.31", "72.31", "5.02", "2.58", "22.61 (20)", "good IoU, weak debris"],
            ["yolo_vit_lstm (YOLO + ViT + LSTM)", "-", "26.02", "76.46", "11.93", "6.35", "22.45 (5)", "hybrid (5-epoch smoke)"],
            ["yolo_bilstm (YOLO + BiLSTM)", "-", "25.23", "75.96", "3.89", "1.98", "22.37 (5)", "hybrid (5-epoch smoke)"],
            ["baseline + GAN data", "24.46 M", "29.52", "75.70", "13.26", "7.10", "23.14 (18)", "GAN hurts"],
            ["attention + GAN data", "24.58 M", "26.65", "75.28", "10.12", "5.33", "22.69 (18)", "GAN hurts"],
        ],
        col_widths=[4.6, 1.7, 2.0, 1.9, 1.9, 1.9, 2.4, 2.6])
    callout(doc,
        "A/B conclusion: the attention and baseline models are essentially tied on mean IoU (29.71 vs 29.75 %), "
        "but the attention model is the better Marine Debris detector (F1 16.06 vs 15.28 %, debris IoU 8.73 vs "
        "8.27 %) and the better overall pixel classifier (77.18 vs 76.15 %).")

    h2(doc, "6.2 Per-Class IoU on the Test Split (%)")
    add_table(doc,
        ["Class", "baseline", "attention", "cnn_unet", "effnet_transformer"],
        [
            ["Marine Debris", "8.27", "8.73", "6.88", "2.58"],
            ["Dense Sargassum", "5.41", "8.36", "4.22", "5.67"],
            ["Sparse Sargassum", "25.79", "14.12", "15.21", "37.13"],
            ["Natural Organic Material", "0.00", "0.85", "0.24", "0.75"],
            ["Ship", "20.64", "21.60", "20.05", "26.06"],
            ["Clouds", "65.06", "67.69", "68.70", "48.94"],
            ["Marine Water", "35.80", "41.57", "31.93", "47.09"],
            ["Sediment-Laden Water", "84.52", "86.79", "0.98", "81.21"],
            ["Foam", "9.12", "26.12", "0.34", "12.56"],
            ["Turbid Water", "46.19", "45.80", "1.03", "40.63"],
            ["Shallow Water", "26.50", "5.15", "4.99", "8.83"],
        ],
        col_widths=[5.4, 2.6, 2.7, 2.6, 3.2])

    h2(doc, "6.3 Model-by-Model Details")
    h3(doc, "6.3.1 Baseline - U-Net with ResNet-34 Encoder (control)")
    para(doc, "The A/B control model: a plain U-Net built with segmentation-models-pytorch, ResNet-34 encoder "
              "pretrained on ImageNet, first convolutional layer adapted from 3 to 11 channels.")
    bullets(doc, [
        "Key features: pretrained encoder, no attention mechanism, 24.46 M parameters.",
        "Accuracy: mean IoU 29.75 %, pixel accuracy 76.15 %, Marine Debris F1 15.28 %, Marine Debris IoU 8.27 %.",
        "Data provided: full 11-class pixel segmentation mask, per-class IoU and F1 values, confusion matrix, "
        "training curves, and a Gaussian concentration heatmap for each analyzed patch.",
    ])
    h3(doc, "6.3.2 Attention - U-Net with scSE Attention Gates (candidate)")
    para(doc, "Identical to the baseline except for concurrent spatial and channel squeeze-and-excitation "
              "(scSE) attention gates injected on the decoder feature maps.")
    bullets(doc, [
        "Key features: scSE attention re-weights spatial and channel features, 24.58 M parameters.",
        "Accuracy: mean IoU 29.71 %, pixel accuracy 77.18 %, Marine Debris F1 16.06 %, Marine Debris IoU 8.73 %.",
        "Data provided: the recommended production model; best debris localization, best pixel accuracy, and "
        "the default checkpoint in both dashboards.",
    ])
    h3(doc, "6.3.3 CNN + U-Net (from-scratch encoder)")
    para(doc, "A compact U-Net with a from-scratch CNN encoder (no pretrained weights), under 0.5 M parameters.")
    bullets(doc, [
        "Key features: tiny footprint, fully trainable from scratch, no pretraining dependency.",
        "Accuracy: mean IoU 14.05 %, pixel accuracy 23.40 %, Marine Debris F1 12.88 %, Marine Debris IoU 6.88 %.",
        "Data provided: a lightweight segmentation baseline useful when compute is heavily constrained; "
        "classified as underfitting.",
    ])
    h3(doc, "6.3.4 EfficientNet-B0 + Transformer")
    para(doc, "An EfficientNet-B0 encoder (11-channel stem) feeding a 2-layer dot-product transformer "
              "bottleneck before the U-Net decoder.")
    bullets(doc, [
        "Key features: transformer bottleneck for long-range spatial context, 8.0 M parameters.",
        "Accuracy: mean IoU 28.31 %, pixel accuracy 72.31 %, Marine Debris F1 5.02 %, Marine Debris IoU 2.58 %.",
        "Data provided: strong scores on Sparse Sargassum (37.13 %), Marine Water (47.09 %), and Ship "
        "(26.06 %), but the weakest debris detection among the four main variants.",
    ])
    h3(doc, "6.3.5 YOLO + ViT + LSTM (yolo_vit_lstm)")
    para(doc, "Hybrid: a YOLOv8n backbone (adapted to 11 channels) with a 2-layer, 4-head Vision Transformer "
              "bottleneck and a unidirectional LSTM over raster-scanned spatial tokens, followed by a 5-stage "
              "bilinear decoder.")
    bullets(doc, [
        "Key features: combines an object-detection backbone with transformer self-attention and sequential LSTM context.",
        "Accuracy (5-epoch smoke training): mean IoU 26.02 %, pixel accuracy 76.46 %, Marine Debris F1 11.93 %, "
        "Marine Debris IoU 6.35 %.",
        "Data provided: an alternative segmentation head that reuses YOLO backbone features; its short train "
        "schedule means its numbers are not directly comparable with the 20-epoch runs.",
    ])
    h3(doc, "6.3.6 YOLO + BiLSTM (yolo_bilstm)")
    para(doc, "Hybrid: a YOLOv8n backbone whose multi-scale features (strides 8/16/32) are pushed through a "
              "shared bidirectional LSTM neck at the coarsest resolution, then decoded back to 256 x 256.")
    bullets(doc, [
        "Key features: bidirectional LSTM captures channel/spectral dependencies across the feature grid.",
        "Accuracy (5-epoch smoke training): mean IoU 25.23 %, pixel accuracy 75.96 %, Marine Debris F1 3.89 %, "
        "Marine Debris IoU 1.98 %.",
        "Data provided: validation that YOLO-backbone encoders underperform the ResNet-34 U-Net family on this "
        "rare-class task at fixed resolution.",
    ])
    h3(doc, "6.3.7 YOLOv8n Box Detector (detection track)")
    para(doc, "A standard YOLOv8n fine-tuned on bounding boxes derived from the Marine Debris masks, trained at "
              "512 px input resolution with a class-balanced split over 100 epochs.")
    bullets(doc, [
        "Key features: object-detection formulation with boxes, confidence, precision, recall, and mAP metrics.",
        "Accuracy: best validation mAP50 about 0.050 (epoch 92), best recall about 0.304, best precision about "
        "0.778. On the official test split, mAP50 is about 0.0001 % (~0) with recall 33.3 %.",
        "Data provided: bounding boxes with confidence; the experiment demonstrates that 1 to 6 px debris "
        "blobs are sub-resolution for box detection, so the segmentation route is the usable one.",
    ])
    h3(doc, "6.3.8 CNN + BiLSTM Patch Classifier (classification track)")
    para(doc, "A per-patch binary Marine Debris present/absent classifier: a shared CNN over the 11-band "
              "spectral sequence read by a bidirectional LSTM.")
    bullets(doc, [
        "Key features: patch-level decision instead of pixel-level segmentation.",
        "Accuracy: patch accuracy 71.87 %, but debris precision, recall, and F1 are all 0 as the model "
        "collapses to predicting the majority class.",
        "Data provided: an honest negative result, showing a coarse classifier cannot exploit the tiny debris "
        "signal; the classifier was discarded in favor of segmentation.",
    ])

    # ------------------------------------------------------------------ 7. GAN
    h1(doc, "7. GAN Augmentation (WGAN-GP)")
    h2(doc, "7.1 Objective")
    para(doc,
         "Because Marine Debris is so rare, an experiment was designed to ask whether synthetic multispectral "
         "patches generated by a generative adversarial network can supplement the training data and improve "
         "detection. The outcome is a documented, honest negative result: GAN augmentation does not help, and "
         "in fact slightly hurts both variants. The experiment is valuable because it answers a common "
         "engineering question with rigorous evidence.")
    h2(doc, "7.2 GAN Architecture")
    add_table(doc,
        ["Component", "Configuration"],
        [
            ["Generator", "100-dim noise -> 11 x 256 x 256 (Sigmoid output), 3.6 M parameters"],
            ["Critic", "11 x 256 x 256 -> scalar Wasserstein score, 2.8 M parameters"],
            ["Loss", "Wasserstein distance + gradient penalty (lambda_gp = 10), n_critic = 5, gradient clip 1.0"],
            ["Training", "200 epochs on the official train split (694 patches)"],
            ["Artifacts", "artifacts/gan/generator.pt, checkpoint.pt, per-epoch sample grids"],
        ],
        col_widths=[3.2, 13.3])
    h2(doc, "7.3 How GAN Augmentation Was Applied")
    numbered(doc, [
        "Train a WGAN-GP generator on the 694 real train patches for 200 epochs (latent dim 100).",
        "Generate 300 synthetic 11-band, 256 x 256 patches (first-order band statistics match the real data; "
        "no mode collapse detected).",
        "Pseudo-label these synthetic patches with a teacher model (baseline or attention checkpoint) keeping "
        "only pixels with softmax confidence >= 0.8.",
        "Calibrate the minimum confident fraction against the real data (25th percentile threshold) and keep "
        "only synthetic patches that pass it. Baseline kept 42 of 300; attention kept 6 of 300.",
        "Mix the kept synthetic patches into the training set and retrain each variant with the exact original "
        "protocol, then re-evaluate on the official test split.",
    ])
    h2(doc, "7.4 Results")
    add_table(doc,
        ["Variant", "Real-only mIoU %", "+GAN mIoU %", "+GAN Debris F1 %", "Synthetic kept"],
        [
            ["baseline", "29.75", "29.52", "13.26", "42 / 300"],
            ["attention", "29.71", "26.65", "10.12", "6 / 300"],
        ],
        col_widths=[4.0, 3.4, 3.0, 3.4, 2.7])
    para(doc,
         "Interpretation: pseudo-labeled synthetic patches overwhelmingly reinforce the majority Marine Water "
         "class, diluting the already-rare debris signal. Both GAN-augmented variants perform worse than "
         "real-data-only training. Recommendation: GAN augmentation is not used for this task; the A/B "
         "conclusion stands on real data alone.")

    # ------------------------------------------------------------------ 8. CV
    h1(doc, "8. Ten-Fold Scene-Disjoint Cross-Validation")
    h2(doc, "8.1 Protocol")
    para(doc,
         "To prove the headline results are not an artifact of one lucky split, a 10-fold cross-validation was "
         "run with sklearn GroupKFold grouped by scene, over train + val (1,022 patches). The official test "
         "split (359 patches) is held out entirely. Each fold trains 20 epochs with the identical trainer "
         "configuration used for the main runs (seed 42).")
    h2(doc, "8.2 Per-Fold Results (baseline U-Net)")
    add_table(doc,
        ["Fold", "Train patches", "Val patches", "Best epoch", "Val mIoU %"],
        [
            ["1", "921", "101", "17", "28.16"],
            ["2", "918", "104", "20", "24.29"],
            ["3", "919", "103", "19", "26.43"],
            ["4", "920", "102", "18", "35.16"],
            ["5", "920", "102", "20", "21.33"],
            ["6", "919", "103", "19", "26.02"],
            ["7", "920", "102", "18", "34.28"],
            ["8", "920", "102", "20", "31.54"],
            ["9", "921", "101", "20", "24.97"],
            ["10", "920", "102", "18", "23.25"],
        ],
        col_widths=[2.5, 4.5, 4.5, 4.0, 4.0])
    h2(doc, "8.3 Summary")
    add_table(doc,
        ["Metric", "Value"],
        [
            ["Mean validation mIoU", "27.54 %"],
            ["Standard deviation", "4.45 %"],
            ["Min / Max", "21.33 % (fold 5) / 35.16 % (fold 4)"],
            ["Single-run test mIoU (official split)", "29.75 %"],
        ],
        col_widths=[8.0, 8.5])
    h2(doc, "8.4 Interpretation")
    para(doc,
         "The single-run official-test result (29.75 %) falls within one standard deviation of the "
         "cross-validation mean (27.54 +/- 4.45 %). This confirms the headline A/B numbers are stable across "
         "scene-disjoint re-partitioning and are not artifacts of one favorable split.")
    h2(doc, "8.5 Cross-Validation Across Other Variants")
    para(doc,
         "A short 5-fold / 5-epoch protocol was also run for the attention, cnn_unet, and effnet_transformer "
         "variants as a consistency check across architectures. These means are not directly comparable to the "
         "baseline 10-fold / 20-epoch numbers because of the shorter protocol.")
    add_table(doc,
        ["Variant", "Mean val mIoU", "+/- std", "Val debris F1"],
        [
            ["attention", "0.1027", "0.0514", "0.0869 +/- 0.0366"],
            ["cnn_unet", "0.1535", "0.0275", "0.1701 +/- 0.0867"],
            ["effnet_transformer", "0.1355", "0.0291", "0.1108 +/- 0.0338"],
        ],
        col_widths=[5.0, 4.0, 3.5, 4.0])

    # ------------------------------------------------------------------ 9. dashboards
    h1(doc, "9. Dashboards")
    para(doc,
         "Two complementary dashboards deliver the project results: a static HTML dashboard for reporting and "
         "an interactive Streamlit dashboard for hands-on analysis. Both are built on the same real artifacts; "
         "a single 'Analyze and Generate Pollution Map' action in the interactive app performs the whole "
         "pipeline and writes a georeferenced contamination GeoTIFF.")

    h2(doc, "9.1 Static Dashboard (dashboard.html)")
    para(doc,
         "The static dashboard is a single self-contained HTML page generated by scripts/build_dashboard.py. "
         "Every number displayed comes from a real artifact file (test_metrics.json, history.csv, "
         "confusion_matrix.json, split files, and generated maps), so the page is a trustworthy snapshot of "
         "the current results and is regenerated automatically after any new training run.")
    h3(doc, "Major features")
    bullets(doc, [
        "Dataset overview: cards for total patches, scenes, image size, spectral bands, official split sizes, "
        "debris patches, sample CRS, and class count, plus a class-presence bar chart and detail table.",
        "Sample test patches: three test-split patches with Marine Debris shown as RGB (B4, B3, B2) and as "
        "semi-transparent 11-class ground-truth overlays with debris clusters boxed in yellow.",
        "Interactive model selector: pill-style toggles that drive the comparison charts, radar, per-class "
        "views, training curves, and confusion galleries.",
        "Full model zoo summary table: every trained variant with parameter count, headline metric, mean IoU, "
        "debris F1, accuracy, mAP50, checkpoint epoch, and notes.",
        "Accuracy comparison: grouped bar chart of pixel/patch accuracy across models, plus a log-scale chart "
        "for the YOLO detection track (mAP50, mAP50-95, recall, precision).",
        "10-fold scene-disjoint CV section: per-variant summary cards (mean, best, worst fold, single-run test "
        "mIoU), per-fold bar charts with a mean line, and fold-detail tables.",
        "Per-class IoU comparison: horizontal grouped bar chart and a detail table highlighting the best model "
        "for each class.",
        "Radar charts: IoU and F1 radar plots across all 11 classes for the selected segmentation models.",
        "Training curves: train loss and validation score per epoch for each selected variant.",
        "Confusion matrix gallery: 11 x 11 row-normalized heatmaps computed on the official test split, plus "
        "the YOLO detector's confusion matrix.",
        "Mapping output gallery: all generated debris concentration map previews.",
    ])

    h2(doc, "9.2 Interactive Streamlit Dashboard (dashboard/app.py)")
    para(doc,
         "The Streamlit app is the operational front end, mirroring the problem-statement pipeline in five "
         "tabs: Source Input, Model, Plastic Detection, Concentration, and Pollution Map. It accepts four "
         "input kinds, runs inference and concentration estimation, exports georeferenced products, and "
         "renders formal PDF incident reports, all fully in memory.")
    h3(doc, "Tab 1 - Source Input")
    bullets(doc, [
        "Four input modes: Sentinel-2 GeoTIFF upload, MARIDA test patch picker (with a debris-only filter), "
        "drone image (.jpg/.png), and drone/coastal video (.mp4/.avi) with frame-rate sampling.",
        "Automatic EXIF GPS extraction for drone images (with realistic coastal demo coordinates as fallback).",
        "One-click 'Analyze & Generate Pollution Map' button that runs the full pipeline, writes the "
        "concentration heatmap GeoTIFF to artifacts/maps/, shows summary metric cards, offers a GeoTIFF "
        "download, and jumps automatically to the pollution map tab.",
    ])
    h3(doc, "Tab 2 - Model")
    bullets(doc, [
        "Selector for the five hybrid segmentation techniques (attention, cnn_unet, effnet_transformer, "
        "yolo_vit_lstm, yolo_bilstm) with the attention model as the default.",
        "Checkpoint availability status, and published test metrics (mean IoU, accuracy, Marine Debris F1/IoU) "
        "loaded live from artifacts/<variant>/test_metrics.json.",
    ])
    h3(doc, "Tab 3 - Plastic Detection")
    bullets(doc, [
        "Side-by-side original image versus detection overlay (red segmentation tint plus yellow debris boxes "
        "with per-pixel mean probability confidence).",
        "Debris pixel count, box count, density, peak concentration, and severity badge; for video, a frame "
        "slider with per-frame overlays and aggregate statistics.",
    ])
    h3(doc, "Tab 4 - Concentration")
    bullets(doc, [
        "Debris density percentage, debris pixels and physical area in m2 (when a georeference or GSD is "
        "available), mean and peak concentration, and surface density in m2 per km2.",
        "Severity pill classification: Low (< 1 %), Moderate (1-5 %), High (5-15 %), Critical (>= 15 %).",
        "Hot colormap concentration heatmap, plus optional custom georeference (center + GSD) for drone/video "
        "inputs.",
    ])
    h3(doc, "Tab 5 - Pollution Map")
    bullets(doc, [
        "Interactive Folium/Leaflet map with the patch-extent polygon, green-to-yellow-to-red concentration "
        "cells, and a toggleable heatmap overlay.",
        "Real WGS84 bounds extracted from the GeoTIFF CRS via rasterio warp (no external geodetic library), "
        "with OpenStreetMap and Esri World Imagery basemaps.",
        "Map telemetry (West/South/East/North) shown as metric cards.",
    ])
    h3(doc, "Official PDF Pollution Incident Report")
    para(doc,
         "Tabs 4 and 5 include a one-click 'Download Official Pollution Incident Report' PDF generator "
         "(dashboard/report_generator.py, rendered entirely in memory with reportlab). The formal A4 report "
         "contains:")
    bullets(doc, [
        "Navy header banner with an incident reference (MPD-YYYY-XXXX), UTC and local timestamps, and the model label.",
        "Geospatial and acquisition telemetry (center coordinates, hemisphere-formatted bounds, sensor type, CRS, ground resolution).",
        "Scene content and data provenance: 11-class coverage grid, SHA-256 surface fingerprint, MARIDA split and "
        "acquisition date, and per-pixel annotator agreement from the confidence masks.",
        "A four-panel visual evidence strip: original RGB, AI overlay with confidence boxes, concentration "
        "heatmap, and a georeferenced static map snapshot.",
        "A quantitative analytics matrix (density %, debris area m2, area analyzed km2, debris m2 per km2, "
        "largest cluster, video sampling stats) with a color-coded severity badge.",
        "A concentration-distribution histogram, model and QA metrics (checkpoint + test IoU/accuracy/debris "
        "F1 from artifacts), a severity-gated operational protocol, repeat-pass surveillance deltas against "
        "the first baseline scan, a per-cluster detection inventory, and a machine-readable QR record.",
    ])
    h3(doc, "Verified end-to-end results (attention model)")
    add_table(doc,
        ["Input", "Detected debris", "Density", "Severity", "Output"],
        [
            ["MARIDA patch 12-12-20_16PCC_0", "6,163 debris pixels", "9.4 %", "High",
             "Map anchored at Gulf of Mexico coords (West -88.335, South 16.002)"],
            ["Uploaded 512 x 512 drone PNG", "16,221 debris pixels", "6.2 %", "High",
             "Map on default coastal georeference (GSD 0.05 m/px)"],
            ["Uploaded 12-frame video", "6 frames sampled, all with debris", "3.0 % mean", "Moderate",
             "Aggregate heatmap + per-frame JSON report"],
        ],
        col_widths=[4.2, 3.4, 2.2, 1.9, 4.8])
    h2(doc, "9.3 Feature Comparison")
    add_table(doc,
        ["Aspect", "Static dashboard (dashboard.html)", "Interactive dashboard (Streamlit)"],
        [
            ["Purpose", "Snapshot reporting of the full model zoo", "Operational, per-image pipeline analysis"],
            ["Refresh", "Regenerated from artifacts via build_dashboard.py", "Live, reads checkpoints and metrics at runtime"],
            ["Input", "Pre-computed artifacts only", "GeoTIFF, MARIDA patch, drone image, video upload"],
            ["Interactivity", "Chart selection via pills", "Full five-tab pipeline with widgets and sliders"],
            ["Model insight", "Compare any subset of the zoo side by side", "Pick one model and run it on new data"],
            ["Geospatial output", "Concentration map gallery", "Interactive Folium map + GeoTIFF download"],
            ["Reporting", "Static tables and charts", "One-click official PDF incident report"],
            ["Best for", "Reviews, reports, benchmarks", "Field analysis, demos, incident response"],
        ],
        col_widths=[3.0, 6.7, 6.8])

    # ------------------------------------------------------------------ 10. findings
    h1(doc, "10. Key Findings and Conclusions")
    numbered(doc, [
        "Attention vs baseline: essentially tied on mean IoU (29.71 vs 29.75 %), but the scSE attention "
        "U-Net is the better Marine Debris detector (F1 16.06 vs 15.28 %, debris IoU 8.73 vs 8.27 %) and the "
        "better pixel classifier (77.18 vs 76.15 %).",
        "Stability: 10-fold scene-disjoint cross-validation for the baseline yields 27.54 +/- 4.45 % "
        "validation mIoU, bracketing the single-run test result (29.75 %). The results are robust to "
        "re-partitioning.",
        "GAN augmentation is a negative result: pseudo-labeled synthetic patches dilute the rare debris "
        "signal, so data synthesis is not recommended for this task.",
        "Rare-class ceiling: Marine Debris is about 0.45 % of labeled pixels and 1 to 6 px in size; all "
        "segmentation models land in the 2 to 9 % debris-IoU band, and box detection (YOLO) is infeasible at "
        "this resolution.",
        "Only the baseline overfits; the attention, from-scratch CNN, and EfficientNet-transformer variants "
        "converge cleanly.",
        "The data is clean: zero exact duplicates; near-duplicate geographic pairs are adjacency artifacts, "
        "not leakage.",
        "The operational pipeline works: a single click converts a GeoTIFF, drone image, or video into a "
        "georeferenced debris concentration GeoTIFF, an interactive pollution map, and a formal PDF incident "
        "report.",
    ])

    # ------------------------------------------------------------------ 11. quality
    h1(doc, "11. Quality Assurance and Reproducibility")
    h2(doc, "11.1 Automated Testing")
    para(doc,
         "A pytest suite of 175 tests guards every module: encoding, ingestion, splits, dataset, augmentation, "
         "metrics, models, detection, drone/video ingestion, mapping, cross-validation, dashboard logic, and "
         "the PDF report generator. All tests pass on the pinned environment.")
    h2(doc, "11.2 Data Integrity and Leakage Audit")
    add_table(doc,
        ["Check", "Result"],
        [
            ["Exact duplicates (whole dataset, content-hash)", "0"],
            ["Near-duplicate pairs (similarity >= 0.985)", "74 pairs"],
            ["Cross-split near-duplicate pairs", "30 pairs (adjacent UTM tile geography, not copied imagery)"],
            ["Suspicious features (single-class dominance > 95 %)", "727 patches (expected for homogeneous ocean tiles)"],
            ["Patch NaN handling", "7 NaN patches sanitized; inert in training"],
        ],
        col_widths=[9.5, 7.0])
    h2(doc, "11.3 Key Bug Fixes")
    para(doc,
         "Two label bugs were found and fixed on 2026-09-12: Bug 1, a class-weight array shifted by one "
         "position that assigned Marine Debris the background weight instead of its true inverse frequency "
         "(about 79); and Bug 2, 1-based training targets that left channel 0 unsupervised and silently "
         "dropped Shallow Water. After both fixes, Marine Debris F1 rose from exactly 0.0 to the reported "
         "15 to 16 % range. All final numbers come from post-fix retrains; pre-fix and GAN-contaminated runs "
         "were quarantined.")
    h2(doc, "11.4 Environment and Dependencies")
    bullets(doc, [
        "Python 3.12 virtual environment, PyTorch 2.14.0 + cu126 on CUDA 12.7 (RTX 4050, 6 GB VRAM).",
        "Pinned requirements.txt with rasterio, segmentation-models-pytorch, albumentations, opencv, "
        "scikit-learn, matplotlib, streamlit, folium, and reportlab.",
        "scripts/verify_env.py provides a repeatable environment and dataset integrity gate.",
    ])

    # ------------------------------------------------------------------ 12. artifacts
    h1(doc, "12. Artifact Index (Selected Outputs)")
    add_table(doc,
        ["Content", "Path"],
        [
            ["Domain glossary and vocabulary", "CONTEXT.md"],
            ["Implementation plan with execution status", "docs/PLAN.md"],
            ["Development log with real numbers", "docs/DEVELOPMENT_LOG.md"],
            ["Pipeline documentation", "docs/PIPELINE.md"],
            ["Project status (single source of truth)", "docs/PROJECT_STATUS.md"],
            ["Static dashboard (HTML)", "dashboard.html (+ scripts/build_dashboard.py)"],
            ["Interactive dashboard (Streamlit)", "dashboard/app.py"],
            ["Test split metrics per variant", "artifacts/<variant>/test_metrics.json"],
            ["10-fold CV summary (baseline)", "artifacts/cv/baseline/cv_summary.json"],
            ["WGAN-GP weights", "artifacts/gan/generator.pt, checkpoint.pt"],
            ["Overfitting report", "artifacts/overfitting_report.json"],
            ["Leakage audit", "artifacts/leakage_audit/report.json"],
            ["Concentration maps", "artifacts/maps/*_concentration.tif (+ PNG)"],
            ["Repeat-pass report history", "artifacts/reports/history.json"],
        ],
        col_widths=[7.5, 9.0])

    h1(doc, "13. References")
    bullets(doc, [
        "MARIDA dataset: Zenodo DOI 10.5281/zenodo.5151941 (Kikaki et al., marine debris segmentation benchmark).",
        "Sentinel-2: ESA/EU Copernicus multispectral mission, 10 m and 20 m spatial resolution bands.",
        "Segmentation Models PyTorch: Yakubovskiy, Qubvel, U-Net with pretrained encoders and scSE blocks.",
        "Project documentation set: CONTEXT.md and docs/* under the project root.",
    ])

    doc.add_paragraph()
    para(doc, "End of document.", italic=True, size=Pt(10), color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.core_properties.title = "AI-Based Marine Plastic Pollution Detection - Project Documentation"
    doc.core_properties.subject = "TECH-ATHON: Marine Debris Segmentation from Sentinel-2"
    doc.core_properties.author = "TECH-ATHON Project Team"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()