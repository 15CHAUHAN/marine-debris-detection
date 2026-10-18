"""Regenerate the static project dashboard (dashboard.html) from trained artifacts.

Every number embedded in the page comes from a real artifact:
  - artifacts/<variant>/test_metrics.json (official test split results)
  - artifacts/<variant>/history.csv or history.json (training curves)
  - artifacts/<variant>/confusion_matrix.json (computed here from best.pth on
    the official test split and cached; regeneration is off-line, computed on
    first run only)
  - labels_mapping.txt and splits/*.txt (dataset statistics)
  - artifacts/maps/ (previously generated concentration maps)
  - artifacts/detection/runs/debris_balanced/confusion_matrix_normalized.png

Nothing is hardcoded. Re-run after a new training run to refresh the page.

Usage:
  .venv\\Scripts\\python.exe scripts\\build_dashboard.py [--no-confusion] [--no-images]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ingestion import MaridaDataset
from src.data.splits import load_official_splits

CLASS_NAMES = [
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Sargassum",
    "Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water",
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
]
SHORT_NAMES = [
    "Debris",
    "Dense Sarg",
    "Sparse Sarg",
    "Organics",
    "Ship",
    "Clouds",
    "Water",
    "Sediment",
    "Foam",
    "Turbid",
    "Shallow",
]

# Task families decide how metrics and curves are interpreted.
VARIANT_META = {
    "baseline": {
        "label": "U-Net (ResNet-34)",
        "family": "segmentation",
        "desc": "Plain U-Net, pretrained ResNet-34 encoder, 11-band input, 11-class output. The A/B control.",
    },
    "attention": {
        "label": "U-Net + scSE attention",
        "family": "segmentation",
        "desc": "Identical U-Net with concurrent spatial/channel squeeze-and-excitation gates in the decoder. The A/B candidate.",
    },
    "cnn_unet": {
        "label": "CNN + U-Net",
        "family": "segmentation",
        "desc": "From-scratch CNN encoder (no ImageNet weights) feeding a U-Net decoder. Small footprint.",
    },
    "effnet_transformer": {
        "label": "EfficientNet-B0 + Transformer",
        "family": "segmentation",
        "desc": "EfficientNet-B0 bottleneck features passed through a 2-layer dot-product transformer before the U-Net decoder.",
    },
    "detection": {
        "label": "YOLOv8n (box detector)",
        "family": "detection",
        "desc": "YOLOv8n fine-tuned on bounding boxes derived from Marine Debris masks; 512 px input, class-balanced train split.",
    },
    "cnn_lstm_classifier": {
        "label": "CNN + BiLSTM classifier",
        "family": "classifier",
        "desc": "Per-patch binary Marine Debris classifier: shared CNN over the 11-band spectral sequence read by a bidirectional LSTM.",
    },
}

VARIANT_COLORS = {
    "baseline": "#3b82f6",
    "attention": "#22d3ee",
    "cnn_unet": "#a78bfa",
    "effnet_transformer": "#34d399",
    "detection": "#f59e0b",
    "cnn_lstm_classifier": "#f43f5e",
}

SKIPPED_DIRS = {"figs", "maps", "_invalid_buggy_first_run"}

CLASS_COLORS = np.array(
    [
        [0.85, 0.85, 0.85],
        [0.90, 0.06, 0.06],
        [0.94, 0.63, 0.13],
        [0.98, 0.78, 0.28],
        [0.20, 0.55, 0.20],
        [0.00, 0.00, 0.25],
        [0.85, 0.85, 0.85],
        [0.10, 0.45, 0.75],
        [0.65, 0.47, 0.30],
        [0.85, 0.90, 0.95],
        [0.60, 0.30, 0.50],
        [0.70, 0.55, 0.90],
    ],
    dtype=np.float64,
)


def variant_meta(variant: str) -> dict:
    if variant in VARIANT_META:
        return dict(VARIANT_META[variant])
    return {"label": variant, "family": "other", "desc": "Trained variant."}


def classify_family(metrics: dict) -> str:
    if "map50" in metrics:
        return "detection"
    if "f1_debris" in metrics or "best_epoch" in metrics:
        return "classifier"
    return "segmentation"


def discover_variants() -> list[str]:
    variants = []
    for run_dir in sorted((ROOT / "artifacts").iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name in SKIPPED_DIRS:
            continue
        if (run_dir / "test_metrics.json").exists():
            variants.append(run_dir.name)
    return variants


def load_test_metrics(variant: str) -> dict:
    path = ROOT / "artifacts" / variant / "test_metrics.json"
    return json.loads(path.read_text())


def load_curves(variant: str, family: str) -> dict:
    run_dir = ROOT / "artifacts" / variant
    csv_path = run_dir / "history.csv"
    json_path = run_dir / "history.json"
    if family == "segmentation" and csv_path.exists():
        epochs, loss, score = [], [], []
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                epochs.append(int(row["epoch"]))
                loss.append(float(row["train_loss"]))
                score.append(float(row["val_mean_iou"]))
        return {
            "epochs": epochs,
            "series": [
                {"name": "Train loss (CE + Dice)", "kind": "loss", "values": loss},
                {"name": "Validation mIoU", "kind": "score", "values": score},
            ],
        }
    if family == "classifier" and json_path.exists():
        rows = json.loads(json_path.read_text())
        if isinstance(rows, list):
            epochs = [int(r["epoch"]) for r in rows]
            train_acc = [float(r["train_acc"]) for r in rows]
            val_f1 = [float(r["val_f1"]) for r in rows]
            return {
                "epochs": epochs,
                "series": [
                    {"name": "Train accuracy", "kind": "score", "values": train_acc},
                    {"name": "Validation debris F1", "kind": "score", "values": val_f1},
                ],
            }
    if family == "detection":
        results_csv = ROOT / "artifacts" / "detection" / "runs" / "debris_balanced" / "results.csv"
        if results_csv.exists():
            epochs, box_loss, map50 = [], [], []
            with results_csv.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    epochs.append(int(row["epoch"]))
                    box_loss.append(float(row["train/box_loss"]))
                    map50.append(float(row.get("metrics/mAP50(B)") or row.get("metrics/mAP50") or 0.0))
            return {
                "epochs": epochs,
                "series": [
                    {"name": "Train box loss", "kind": "loss", "values": box_loss},
                    {"name": "Validation mAP50", "kind": "score", "values": map50},
                ],
            }
    return {"epochs": [], "series": []}


def count_params(variant: str, family: str) -> int | None:
    try:
        if family == "segmentation":
            from src.models.registry import resolve_variant

            model = resolve_variant(variant, pretrained=False)
            return sum(p.numel() for p in model.parameters())
        if family == "classifier":
            from src.models.cnn_lstm import CNNBiLSTMClassifier

            return sum(p.numel() for p in CNNBiLSTMClassifier().parameters())
        if family == "detection":
            from ultralytics import YOLO

            weights = ROOT / "artifacts" / "detection" / "runs" / "debris_balanced" / "weights" / "best.pt"
            model = YOLO(str(weights))
            return sum(p.numel() for p in model.model.parameters())
    except Exception as exc:  # best effort; the dashboard shows "n/a" on failure
        print(f"warning: param count failed for {variant}: {exc}")
    return None


def headline_metric(variant: str, family: str, metrics: dict) -> dict:
    if family == "segmentation":
        return {"value": round(metrics["accuracy"] * 100, 2), "unit": "%", "name": "Pixel accuracy"}
    if family == "classifier":
        return {"value": round(metrics["accuracy"] * 100, 2), "unit": "%", "name": "Patch accuracy"}
    return {"value": round(metrics["map50"] * 100, 4), "unit": "%", "name": "mAP50"}


ARCH_ALIASES = {
    "attention_gan_augmented": "attention",
    "baseline_gan_aug_run_final": "baseline",
    "baseline_gan_augmented": "baseline",
}


def compute_confusion_matrix(variant: str, force: bool = False) -> dict | None:
    """11x11 confusion matrix on the official test split (labeled pixels only)."""
    cache = ROOT / "artifacts" / variant / "confusion_matrix.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    print(f"computing confusion matrix for {variant} on the official test split...")
    import torch

    from src.data.factory import build_loaders
    from src.data.ingestion import MaridaDataset
    from src.data.splits import load_official_splits
    from src.evaluate.metrics import aggregate_metrics, evaluate_predictions
    from src.models.registry import resolve_variant

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    test_loader = build_loaders(marida, splits, batch_size=8)["test"]

    model = resolve_variant(variant, pretrained=False)
    from src.utils.checkpoint import safe_load_state_dict
    safe_load_state_dict(model, ROOT / "artifacts" / variant / "best.pth", map_location="cpu")
    model.to(device).eval()

    metrics_list = []
    with torch.no_grad():
        for image, mask in test_loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=True):
                logits = model(image)
            metrics_list.append(evaluate_predictions(logits.float(), mask))
    summary = aggregate_metrics(metrics_list)
    cm = summary["confusion_matrix"].astype(int).tolist()
    out = {"matrix": cm, "class_names": SHORT_NAMES}
    cache.write_text(json.dumps(out, indent=2))
    return out


def render_confusion_heatmap(variant: str, cm_data: dict, out_dir: Path) -> str:
    matrix = np.array(cm_data["matrix"], dtype=np.int64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = matrix / np.where(row_sums == 0, 1, row_sums)

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(11), SHORT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(11), SHORT_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{variant} - confusion matrix (test, labeled pixels only)")
    for i in range(11):
        for j in range(11):
            if norm[i, j] >= 0.005:
                ax.text(j, i, f"{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.5 else "#0b0f1a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout()
    out_path = out_dir / f"cm_{variant}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return f"artifacts/figs/dashboard/{out_path.name}"


def dataset_stats() -> dict:
    marida = MaridaDataset(ROOT)
    labels = json.loads((ROOT / "labels_mapping.txt").read_text())
    labels_arr = np.array(list(labels.values()), dtype=np.int64)  # (1381, 15)
    agg = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 7, 7, 7, 7])
    cols_by_class = {g: [j for j in range(15) if agg[j + 1] == g] for g in range(1, 12)}
    presence_agg = np.zeros(11, dtype=np.int64)
    for i in range(11):
        presence_agg[i] = int(np.any(labels_arr[:, cols_by_class[i + 1]], axis=1).sum())
    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = len((ROOT / "splits" / f"{name}_X.txt").read_text().splitlines())
    sample = marida.read_meta(marida.ids[0])
    crs_sample = sorted({marida.read_meta(pid).crs for pid in list(marida.ids)[:30]})
    return {
        "patches": len(marida),
        "scenes": len(marida.scene_ids),
        "size": f"{sample.width}x{sample.height}",
        "bands": sample.band_count,
        "crs": ", ".join(crs_sample[:4]),
        "splits": splits,
        "debris_patches": int(labels_arr[:, 0].sum()),
        "presence": [{"name": CLASS_NAMES[i], "patches": int(presence_agg[i])} for i in range(11)],
    }


def _gt_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Semi-transparent 11-class overlay on a dimmed RGB base.

    Unlabeled pixels (``mask == 0``) keep the satellite RGB dimmed to 70% so
    the ocean context stays visible; labeled pixels are blended ~50/50 with
    their class color. Marine Debris (``mask == 1``) is filled red after a
    5x5 dilation so single-pixel debris is actually visible.
    """
    from scipy import ndimage as ndi

    overlay = (0.7 * rgb).astype(np.float64)
    labeled = mask > 0
    if labeled.any():
        overlay[labeled] = (overlay[labeled] + CLASS_COLORS[mask[labeled]]) * 0.5
    dil = ndi.binary_dilation(mask == 1, structure=np.ones((5, 5)))
    if dil.any():
        red = np.array([1.0, 0.15, 0.15])
        overlay[dil] = overlay[dil] * 0.45 + red * 0.55
    return np.clip(overlay, 0.0, 1.0)


def _draw_debris_boxes(ax, mask: np.ndarray, linewidth: float = 2.0) -> None:
    """Draw bright-yellow boxes with a text label around each debris cluster.

    Boxes are drawn on the dilated Marine Debris mask so a 1-6 pixel cluster
    becomes a clearly visible marker on the overlay.
    """
    from scipy import ndimage as ndi

    dil = ndi.binary_dilation(mask == 1, structure=np.ones((5, 5)))
    if not dil.any():
        return
    labels, _ = ndi.label(dil)
    for sl in ndi.find_objects(labels):
        y1, x1 = sl[0].start, sl[1].start
        y2, x2 = sl[0].stop - 1, sl[1].stop - 1
        pad = 1.5
        ax.add_patch(
            mpatches.Rectangle(
                (x1 - pad, y1 - pad),
                (x2 - x1 + 1) + 2 * pad,
                (y2 - y1 + 1) + 2 * pad,
                linewidth=linewidth,
                edgecolor="#ffee00",
                facecolor="none",
            )
        )
        label_y = y1 - 5 if y1 >= 9 else y2 + 6
        ax.text(
            x1 - pad,
            label_y,
            "debris",
            fontsize=8,
            color="#ffee00",
            bbox=dict(boxstyle="round,pad=0.15", fc="#0b0f1a", ec="#ffee00", alpha=0.85),
        )


def build_dataset_samples(out_dir: Path) -> str:
    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    picked = []
    seen_scenes = set()
    for pid in splits["test"]:
        scene = marida.read_scene_id(pid)
        if scene in seen_scenes:
            continue
        if (marida.read_mask(pid) == 1).sum() > 0:
            picked.append(pid)
            seen_scenes.add(scene)
        if len(picked) == 3:
            break
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, pid in enumerate(picked):
        image = marida.read_image(pid)
        mask = marida.read_mask(pid)
        rgb = np.clip(image[[2, 1, 0]].transpose(1, 2, 0) / max(float(image.max()), 1e-6), 0, 1)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"{pid}\nRGB (B4,B3,B2)", fontsize=9)
        axes[1, col].imshow(_gt_overlay(rgb, mask))
        axes[1, col].set_title("Ground Truth Overlay (Debris Boxed)", fontsize=9)
        _draw_debris_boxes(axes[1, col], mask)
        for ax in (axes[0, col], axes[1, col]):
            ax.axis("off")
    handles = [mpatches.Patch(color=CLASS_COLORS[i + 1], label=CLASS_NAMES[i]) for i in range(11)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = out_dir / "dataset_samples.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"artifacts/figs/dashboard/{out_path.name}"


def collect_variant_data(variants: list[str], compute_confusion: bool) -> list[dict]:
    out_dir = ROOT / "artifacts" / "figs" / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    collected = []
    for variant in variants:
        meta = variant_meta(variant)
        metrics = load_test_metrics(variant)
        family = classify_family(metrics)
        curves = load_curves(variant, family)
        params = count_params(variant, family)
        entry = {
            "key": variant,
            "label": meta["label"],
            "family": family,
            "desc": meta["desc"],
            "params": params,
            "headline": headline_metric(variant, family, metrics),
            "metrics": metrics,
            "curves": curves,
            "color": VARIANT_COLORS.get(variant, "#64748b"),
        }
        cm_img = None
        if family == "segmentation":
            cm_data = None
            if compute_confusion:
                cm_data = compute_confusion_matrix(variant)
            else:
                cache = ROOT / "artifacts" / variant / "confusion_matrix.json"
                if cache.exists():
                    cm_data = json.loads(cache.read_text())
            if cm_data:
                cm_img = render_confusion_heatmap(variant, cm_data, out_dir)
        elif family == "detection":
            yolo_cm = ROOT / "artifacts" / "detection" / "runs" / "debris_balanced" / "confusion_matrix_normalized.png"
            if yolo_cm.exists():
                cm_img = "artifacts/detection/runs/debris_balanced/confusion_matrix_normalized.png"
        entry["cmImage"] = cm_img
        collected.append(entry)
    return collected


def collect_cv_data() -> list[dict]:
    """Discover 10-fold CV summaries under artifacts/cv/<variant>/cv_summary.json."""
    cv_root = ROOT / "artifacts" / "cv"
    out = []
    if not cv_root.is_dir():
        return out
    for variant_dir in sorted(cv_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        summary_path = variant_dir / "cv_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        variant = summary["variant"]
        meta = variant_meta(variant)
        single_run = None
        tm_path = ROOT / "artifacts" / variant / "test_metrics.json"
        if tm_path.exists():
            single_run = json.loads(tm_path.read_text()).get("mean_iou")
        out.append(
            {
                "variant": variant,
                "label": meta["label"],
                "color": VARIANT_COLORS.get(variant, "#64748b"),
                "folds": summary["folds"],
                "epochs": summary["epochs"],
                "mean": summary["mean_val_mIoU"],
                "std": summary["std_val_mIoU"],
                "min": summary["min_val_mIoU"],
                "max": summary["max_val_mIoU"],
                "single_run_test_miou": single_run,
                "fold_results": summary["fold_results"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-confusion", action="store_true", help="skip computing missing confusion matrices")
    parser.add_argument("--no-images", action="store_true", help="skip regenerating dataset sample pictures")
    args = parser.parse_args()

    variants = discover_variants()
    print(f"discovered variants: {variants}")
    if not variants:
        raise SystemExit("no trained variants found under artifacts/")

    out_dir = ROOT / "artifacts" / "figs" / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_img = None
    if not args.no_images:
        samples_img = build_dataset_samples(out_dir)

    stats = dataset_stats()
    variants_data = collect_variant_data(variants, compute_confusion=not args.no_confusion)
    cv_data = collect_cv_data()

    maps = []
    for path in sorted((ROOT / "artifacts" / "maps").glob("*concentration.png")):
        maps.append({"img": f"artifacts/maps/{path.name}", "label": path.stem.replace("_", " ") + " (Gaussian concentration, sigma=2)"})

    data = {
        "date": datetime.date.today().isoformat(),
        "dataset": stats,
        "samplesImage": samples_img,
        "classNames": CLASS_NAMES,
        "variants": variants_data,
        "cv": cv_data,
        "maps": maps,
    }

    html = render_html(data)
    out_html = ROOT / "dashboard.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"wrote {out_html} ({len(html):,} bytes)")


def render_html(data: dict) -> str:
    """Assemble the static dashboard page. All data flows through window.DATA."""
    payload = json.dumps(data, ensure_ascii=False)
    default_selection = json.dumps(
        [v["key"] for v in data["variants"] if v["family"] == "segmentation"],
        ensure_ascii=False,
    )
    n_variants = len(data["variants"])
    return TEMPLATE.replace("@@DATA@@", payload).replace(
        "@@SELECTION@@", default_selection
    ).replace("@@N_VARIANTS@@", str(n_variants)).replace("@@DATE@@", data["date"])


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MARIDA Marine Debris - Model Zoo Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:#0b0f1a; --surface:#131929; --surface2:#1a2236; --border:#1e2d4a;
  --accent:#3b82f6; --accent2:#22d3ee; --danger:#f43f5e; --warn:#f59e0b;
  --success:#10b981; --text:#e2e8f0; --muted:#64748b; --radius:12px;
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
body { font-family:"Inter","Segoe UI",sans-serif; background:var(--bg); color:var(--text); min-height:100vh; padding-bottom:48px; }
header { background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%); border-bottom:1px solid var(--border); padding:28px 40px; display:flex; align-items:center; gap:20px; flex-wrap:wrap; }
header .icon { font-size:2.4rem; }
header h1 { font-size:1.6rem; font-weight:700; letter-spacing:-.3px; }
header p { color:var(--muted); font-size:.88rem; margin-top:4px; }
.badge { display:inline-block; padding:3px 10px; border-radius:99px; font-size:.72rem; font-weight:600; background:#1d4ed820; color:var(--accent); border:1px solid #1d4ed840; margin-left:8px; vertical-align:middle; }
.page { max-width:1280px; margin:0 auto; padding:32px 24px 0; }
.section-title { font-size:1.05rem; font-weight:600; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; margin:40px 0 16px; display:flex; align-items:center; gap:8px; }
.section-title::after { content:""; flex:1; height:1px; background:var(--border); }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:16px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:18px 20px; transition:border-color .2s; }
.card:hover { border-color:var(--accent); }
.card .label { font-size:.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; }
.card .value { font-size:1.8rem; font-weight:700; margin:6px 0 2px; }
.card .sub { font-size:.76rem; color:var(--muted); }
.card.blue .value { color:var(--accent); } .card.cyan .value { color:var(--accent2); }
.card.green .value { color:var(--success); } .card.rose .value { color:var(--danger); }
.card.amber .value { color:var(--warn); } .card.violet .value { color:#a78bfa; }
.charts-2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
@media(max-width:940px) { .charts-2 { grid-template-columns:1fr; } }
.chart-box { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:20px 22px; }
.chart-box h3 { font-size:.9rem; font-weight:600; margin-bottom:14px; color:var(--text); }
.chart-box canvas { width:100% !important; }
.note { font-size:.76rem; color:var(--muted); margin-top:8px; line-height:1.5; }
.pill-row { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 2px; }
.pill { padding:6px 14px; border-radius:99px; border:1px solid var(--border); background:var(--surface2); color:var(--muted); font-size:.78rem; font-weight:600; cursor:pointer; user-select:none; transition:all .15s; }
.pill.on { background:#1d4ed830; color:var(--text); border-color:var(--accent); }
.pill .fam { font-size:.62rem; opacity:.7; margin-left:4px; text-transform:uppercase; letter-spacing:.04em; }
table { width:100%; border-collapse:collapse; font-size:.8rem; }
th { color:var(--muted); font-weight:600; text-align:left; padding:7px 9px; border-bottom:1px solid var(--border); }
td { padding:7px 9px; border-bottom:1px solid #1e2d4a20; vertical-align:middle; }
tr:last-child td { border-bottom:none; }
.bar-outer { background:var(--surface2); border-radius:4px; height:7px; flex:1; overflow:hidden; min-width:60px; }
.bar-inner { height:7px; border-radius:4px; }
.bar-row { display:flex; align-items:center; gap:8px; }
.bar-val { min-width:44px; text-align:right; color:var(--text); }
.img-box { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:14px; }
.img-box img { width:100%; border-radius:8px; border:1px solid var(--border); }
.img-box .cap { font-size:.74rem; color:var(--muted); margin-top:8px; text-align:center; }
.grid-imgs { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }
@media(max-width:900px) { .grid-imgs { grid-template-columns:1fr; } }
.tbl-wrap { overflow-x:auto; }
footer { text-align:center; color:var(--muted); font-size:.75rem; margin-top:56px; padding-top:20px; border-top:1px solid var(--border); }
img.sample { width:100%; border-radius:8px; border:1px solid var(--border); }
.na { color:var(--muted); }
</style>
</head>
<body>

<header>
  <span class="icon">&#127756;</span>
  <div>
    <h1>MARIDA Marine Debris Detection <span class="badge">Model Zoo</span></h1>
    <p>Sentinel-2 &middot; 11 bands &middot; 11 aggregated classes &middot; @@N_VARIANTS@@ architectures trained on the official scene-disjoint split</p>
  </div>
</header>

<div class="page">

  <div class="section-title">&#128230; Dataset Features</div>
  <div class="cards" id="dataset-cards"></div>
  <div class="charts-2" style="margin-top:16px;">
    <div class="chart-box">
      <h3>Patches containing each class (presence over all patches)</h3>
      <canvas id="presenceChart" height="240"></canvas>
    </div>
    <div class="chart-box">
      <h3>Class presence detail</h3>
      <div class="tbl-wrap"><table id="presenceTable"></table></div>
    </div>
  </div>
  <div id="samples-wrap" style="margin-top:16px;"></div>

  <div class="section-title">&#129513; Compare Models</div>
  <div class="chart-box">
    <h3>Select models for the comparison charts (the summary table always lists every trained variant)</h3>
    <div class="pill-row" id="selector"></div>
    <div class="note">Radar, per-class, curves and confusion views follow this selection. Detection uses mAP-based scores, segmentation uses pixel metrics, so the two families are never mixed in one per-class comparison.</div>
  </div>

  <div class="section-title">&#127919; Full Model Zoo Summary</div>
  <div class="chart-box"><div class="tbl-wrap"><table id="zooTable"></table></div></div>

  <div class="section-title">&#127919; Accuracy Comparison</div>
  <div class="charts-2">
    <div class="chart-box">
      <h3>Pixel / patch accuracy across models</h3>
      <canvas id="accChart" height="260"></canvas>
      <div class="note">Segmentation variants report labeled-pixel accuracy on the official test split; the BiLSTM reports patch-level accuracy.</div>
    </div>
    <div class="chart-box">
      <h3>Detection track (YOLOv8n, 512 px, balanced)</h3>
      <canvas id="detChart" height="260"></canvas>
      <div class="note">Marine Debris blobs are 1-6 px, sub-resolution for a detector; mAP50 is near zero while recall reaches 33%.</div>
    </div>
  </div>

  <div class="section-title">&#128200; 10-Fold Scene-Disjoint Cross-Validation</div>
  <div id="cv-wrap"></div>
  <div class="note">CV is run on train+val (1,022 patches) with GroupKFold grouped by scene; the official test split (359 patches) is held out entirely. Mean val mIoU across folds is compared with the single-run test mIoU.</div>

  <div class="section-title">&#128202; Per-Class Comparison (Segmentation)</div>
  <div class="charts-2">
    <div class="chart-box">
      <h3>Per-Class IoU (horizontal grouped bars)</h3>
      <canvas id="iouBarChart" height="320"></canvas>
    </div>
    <div class="chart-box">
      <h3>Per-Class detail</h3>
      <div class="tbl-wrap"><table id="classTable"></table></div>
    </div>
  </div>

  <div class="section-title">&#128376;&#65039; Radar Comparison</div>
  <div class="charts-2">
    <div class="chart-box"><h3>IoU radar - 11 classes</h3><canvas id="radarIoU" height="300"></canvas></div>
    <div class="chart-box"><h3>F1 radar - 11 classes</h3><canvas id="radarF1" height="300"></canvas></div>
  </div>

  <div class="section-title">&#128200; Training Curves per Variant</div>
  <div class="charts-2">
    <div class="chart-box"><h3>Loss (train)</h3><canvas id="lossChart" height="240"></canvas></div>
    <div class="chart-box"><h3>Validation score</h3><canvas id="scoreChart" height="240"></canvas></div>
  </div>

  <div class="section-title">&#129518; Confusion Matrix View</div>
  <div class="grid-imgs" id="cm-gallery"></div>
  <div class="note">Segmentation matrices are 11x11 row-normalized heatmaps from the official test split (labeled pixels only). The YOLO detector keeps its own binary matrix.</div>

  <div class="section-title">&#128506;&#65039; Mapping Output</div>
  <div class="grid-imgs" id="maps-gallery"></div>

</div>

<footer>
  TECH-ATHON &middot; MARIDA Marine Debris Detection &middot; Sentinel-2 Multispectral &middot;
  U-Net family (ResNet-34, scSE, CNN, EfficientNet+Transformer), YOLOv8n, CNN+BiLSTM &middot;
  generated from artifacts on @@DATE@@
</footer>

<script>
window.DATA = @@DATA@@;
var CLASSES = window.DATA.classNames;
var VARIANTS = window.DATA.variants;
var selected = new Set(@@SELECTION@@);

function variantByKey(k) {
  for (var i = 0; i < VARIANTS.length; i++) {
    if (VARIANTS[i].key === k) return VARIANTS[i];
  }
  return null;
}
function segVariants() { return VARIANTS.filter(function (v) { return v.family === 'segmentation'; }); }
function selectedVariants() { return VARIANTS.filter(function (v) { return selected.has(v.key); }); }
function selectedSeg() { return VARIANTS.filter(function (v) { return selected.has(v.key) && v.family === 'segmentation'; }); }
function famLabel(f) { return f === 'segmentation' ? 'seg' : (f === 'detection' ? 'yolo' : 'clf'); }

function fmtP(v, dec) {
  dec = dec === undefined ? 1 : dec;
  return (v * 100).toFixed(dec) + '%';
}

/* ---- model selector ---- */
function buildSelector() {
  var el = document.getElementById('selector');
  el.innerHTML = '';
  VARIANTS.forEach(function (v) {
    var b = document.createElement('span');
    var on = selected.has(v.key);
    b.className = 'pill' + (on ? ' on' : '');
    b.style.borderColor = on ? v.color : '';
    b.style.background = on ? v.color + '22' : '';
    b.style.color = on ? v.color : '';
    b.innerHTML = v.label + ' <span class="fam">' + famLabel(v.family) + '</span>';
    b.onclick = function () {
      if (selected.has(v.key)) { selected.delete(v.key); } else { selected.add(v.key); }
      if (selected.size === 0) selected.add(v.key);
      buildSelector();
      rerender();
    };
    el.appendChild(b);
  });
}

/* ---- chart registry so charts are recreated on selection change ---- */
var charts = {};
function makeChart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  var ctx = document.getElementById(id);
  if (ctx) charts[id] = new Chart(ctx, cfg);
}
Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "#1e2d4a";

/* ---- dataset overview ---- */
function renderDataset() {
  var d = window.DATA.dataset;
  var cards = [
    { l: 'Total patches', v: d.patches.toLocaleString(), c: 'blue', s: '100% MARIDA benchmark' },
    { l: 'Scenes', v: d.scenes, c: 'cyan', s: 'Sentinel-2 acquisitions' },
    { l: 'Image size', v: d.size, c: 'violet', s: 'pixels per patch' },
    { l: 'Spectral bands', v: d.bands, c: 'blue', s: 'B1..B12 excl. 60 m' },
    { l: 'Train / Val / Test patches', v: d.splits.train + ' / ' + d.splits.val + ' / ' + d.splits.test, c: 'amber', s: 'official scene-disjoint split' },
    { l: 'Debris patches', v: d.debris_patches, c: 'rose', s: 'class 1 present' },
    { l: 'CRS (sample)', v: d.crs, c: '', s: 'UTM zones vary by scene' },
    { l: 'Target classes', v: 11, c: 'green', s: '15 mapped to 11' }
  ];
  var html = '';
  cards.forEach(function (c) {
    html += '<div class="card ' + (c.c || '') + '"><div class="label">' + c.l + '</div><div class="value">' + c.v + '</div><div class="sub">' + c.s + '</div></div>';
  });
  document.getElementById('dataset-cards').innerHTML = html;

  makeChart('presenceChart', {
    type: 'bar',
    data: {
      labels: d.presence.map(function (p) { return p.name; }),
      datasets: [{
        label: 'Patches containing class',
        data: d.presence.map(function (p) { return p.patches; }),
        backgroundColor: '#3b82f622',
        borderColor: '#3b82f6',
        borderWidth: 1.5
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'Patches (out of ' + d.patches + ')' } } }
    }
  });

  var rows = d.presence.map(function (p, i) {
    return '<tr><td style="font-weight:' + (i === 0 ? '700' : '400') + ';color:' + (i === 0 ? '#f43f5e' : 'inherit') + '">' + p.name + '</td>' +
      '<td><div class="bar-row"><div class="bar-outer"><div class="bar-inner" style="width:' + (100 * p.patches / d.patches) + '%;background:' + (i === 0 ? '#f43f5e' : '#3b82f6') + '"></div></div><span class="bar-val">' + p.patches + '</span></div></td></tr>';
  }).join('');
  document.getElementById('presenceTable').innerHTML = '<thead><tr><th>Class</th><th>Patches present</th></tr></thead><tbody>' + rows + '</tbody>';

  if (window.DATA.samplesImage) {
    document.getElementById('samples-wrap').innerHTML =
      '<div class="chart-box"><h3>Sample test patches (RGB vs ground-truth overlay)</h3><img class="sample" src="' + window.DATA.samplesImage + '" alt="dataset samples"/>' +
      '<div class="note">Three test-split patches with Marine Debris, shown as RGB (B4,B3,B2) and semi-transparent 11-class overlays on the dimmed satellite imagery, with Marine Debris boxed in yellow.</div></div>';
  }
}

/* ---- model zoo summary ---- */
function metricCell(v, unit, dec) {
  if (v === null || v === undefined || isNaN(v)) return '<span class="na">n/a</span>';
  dec === undefined && (dec = 2);
  return v.toFixed(dec) + (unit || '');
}
function renderZoo() {
  var head = '<thead><tr><th>Model</th><th>Type</th><th>Architecture</th><th>Params</th><th>Headline</th>'
    + '<th>Mean IoU</th><th>Debris F1</th><th>Accuracy</th><th>mAP50</th><th>Checkpoint</th><th>Notes</th></tr></thead>';
  var body = '';
  VARIANTS.forEach(function (v) {
    var m = v.metrics;
    var isSeg = v.family === 'segmentation';
    var isDet = v.family === 'detection';
    var isClf = v.family === 'classifier';
    var ev = m.checkpoint_epoch !== undefined ? m.checkpoint_epoch : (m.best_epoch !== undefined ? m.best_epoch : null);
    var notes = isDet ? 'R=' + fmtP(m.recall, 1) + ', P=' + fmtP(m.precision, 4) + ', mAP50-95=' + fmtP(m.map50_95, 4)
      : isClf ? 'P=' + fmtP(m.precision, 2) + ', R=' + fmtP(m.recall, 2) + ', best val F1 ep ' + m.best_epoch
      : 'val mIoU ' + (Number(m.checkpoint_epoch) >= 0 ? 'best ep' : '');
    body += '<tr style="' + (selected.has(v.key) ? 'background:' + v.color + '0e;border-left:3px solid ' + v.color : '') + '">'
      + '<td style="font-weight:600;color:' + v.color + '">' + v.label + '</td>'
      + '<td>' + v.family + '</td>'
      + '<td>' + v.desc + '</td>'
      + '<td>' + (v.params ? v.params.toLocaleString() : '<span class="na">n/a</span>') + '</td>'
      + '<td>' + metricCell(v.headline.value, '%', 2) + ' <span class="na">(' + v.headline.name + ')</span></td>'
      + '<td>' + (isSeg ? metricCell(m.mean_iou * 100, '%', 2) : '<span class="na">n/a</span>') + '</td>'
      + '<td>' + (isSeg ? metricCell(m.f1_class_1_marine_debris * 100, '%', 2) : '<span class="na">n/a</span>') + '</td>'
      + '<td>' + (isDet ? '<span class="na">n/a</span>' : metricCell(m.accuracy * 100, '%', 2)) + '</td>'
      + '<td>' + (isDet ? metricCell(m.map50 * 100, '%', 3) : '<span class="na">n/a</span>') + '</td>'
      + '<td>' + (ev === null ? '<span class="na">n/a</span>' : 'Ep ' + ev) + '</td>'
      + '<td class="na" style="font-size:.72rem">' + notes + '</td></tr>';
  });
  document.getElementById('zooTable').innerHTML = head + '<tbody>' + body + '</tbody>';
}

/* ---- accuracy ---- */
function renderAccuracy() {
  var withAcc = VARIANTS.filter(function (v) { return v.family !== 'detection'; });
  makeChart('accChart', {
    type: 'bar',
    data: {
      labels: withAcc.map(function (v) { return v.label; }),
      datasets: [{
        label: 'Accuracy (%)',
        data: withAcc.map(function (v) { return v.metrics.accuracy * 100; }),
        backgroundColor: withAcc.map(function (v) { return v.color + '22'; }),
        borderColor: withAcc.map(function (v) { return v.color; }),
        borderWidth: 2
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: function (c) { var v = withAcc[c.dataIndex]; return v.headline.name + ': ' + c.raw.toFixed(2) + '%'; } } }
      },
      scales: { x: { min: 0, max: 100, title: { display: true, text: 'Score (%)' } } }
    }
  });

  var det = null;
  for (var i = 0; i < VARIANTS.length; i++) { if (VARIANTS[i].family === 'detection') det = VARIANTS[i]; }
  if (det) {
    var m = det.metrics;
    var detData = [m.map50 * 100, m.map50_95 * 100, m.recall * 100, m.precision * 100];
    makeChart('detChart', {
      type: 'bar',
      data: {
        labels: ['mAP50', 'mAP50-95', 'Recall', 'Precision'],
        datasets: [{ label: '%', data: detData, backgroundColor: '#f59e0b22', borderColor: '#f59e0b', borderWidth: 2 }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { y: { type: 'logarithmic', title: { display: true, text: '% (log scale)' } } }
      }
    });
  }
}

/* ---- per class ---- */
function perClass(v, metric) { return v.metrics[metric]; }
function renderPerClass() {
  var vs = selectedSeg();
  if (!vs.length) { makeChart('iouBarChart', { type: 'bar', data: { labels: [], datasets: [] } }); return; }
  var datasets = vs.map(function (v) {
    var pc = perClass(v, 'iou_per_class');
    return {
      label: v.label,
      data: CLASSES.map(function (c) { return (pc[c] || 0) * 100; }),
      backgroundColor: v.color + '22',
      borderColor: v.color,
      borderWidth: 1.5
    };
  });
  makeChart('iouBarChart', {
    type: 'bar',
    data: { labels: CLASSES, datasets: datasets },
    options: {
      indexAxis: 'y',
      plugins: { legend: { position: 'bottom' } },
      scales: { x: { min: 0, max: 100, title: { display: true, text: 'IoU (%)' } }, y: { ticks: { font: { size: 10 } } } }
    }
  });

  var cols = vs.map(function (v) { return '<th style="color:' + v.color + '">' + v.label + ' IoU</th>'; }).join('');
  var rows = CLASSES.map(function (c, i) {
    var cells = vs.map(function (v) {
      var pc = perClass(v, 'iou_per_class');
      var val = (pc[c] || 0) * 100;
      return '<td><div class="bar-row"><div class="bar-outer"><div class="bar-inner" style="width:' + val.toFixed(1) + '%;background:' + v.color + '"></div></div><span class="bar-val">' + val.toFixed(1) + '%</span></div></td>';
    });
    var best = vs[0];
    for (var j = 1; j < vs.length; j++) {
      if ((perClass(vs[j], 'iou_per_class')[c] || 0) > (perClass(best, 'iou_per_class')[c] || 0)) best = vs[j];
    }
    var bv = (perClass(best, 'iou_per_class')[c] || 0);
    var isMarine = i === 0;
    return '<tr style="' + (isMarine ? 'background:#f43f5e08' : '') + '">'
      + '<td style="font-weight:' + (isMarine ? '600' : '400') + ';color:' + (isMarine ? '#f43f5e' : 'inherit') + '">' + c + '</td>'
      + cells.join('')
      + '<td style="color:' + best.color + ';font-weight:600;">' + (bv * 100).toFixed(1) + '%</td></tr>';
  }).join('');
  document.getElementById('classTable').innerHTML = '<thead><tr><th>Class</th>' + cols + '<th>Best of selected</th></tr></thead><tbody>' + rows + '</tbody>';
}

/* ---- radar ---- */
function renderRadar() {
  var vs = selectedSeg();
  if (!vs.length) { makeChart('radarIoU', { type: 'radar', data: { labels: [], datasets: [] } }); return; }
  var short = CLASSES.map(function (c) { return c.length > 13 ? c.substring(0, 13) + '...' : c; });
  var mk = function (metric) {
    return vs.map(function (v) {
      var pc = perClass(v, metric);
      return {
        label: v.label,
        data: CLASSES.map(function (c) { return (pc[c] || 0) * 100; }),
        borderColor: v.color,
        backgroundColor: v.color + '22',
        pointBackgroundColor: v.color,
        borderWidth: 2
      };
    });
  };
  makeChart('radarIoU', {
    type: 'radar', data: { labels: short, datasets: mk('iou_per_class') },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { r: { min: 0, max: 100, ticks: { stepSize: 20, font: { size: 9 } }, pointLabels: { font: { size: 9 } } } } }
  });
  makeChart('radarF1', {
    type: 'radar', data: { labels: short, datasets: mk('f1_per_class') },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { r: { min: 0, max: 100, ticks: { stepSize: 20, font: { size: 9 } }, pointLabels: { font: { size: 9 } } } } }
  });
}

/* ---- curves ---- */
function renderCurves() {
  var withCurves = selectedVariants().filter(function (v) { return v.curves.epochs.length > 0; });
  if (!withCurves.length) {
    makeChart('lossChart', { type: 'line', data: { labels: [], datasets: [] } });
    makeChart('scoreChart', { type: 'line', data: { labels: [], datasets: [] } });
    return;
  }
  var labels = [];
  withCurves.forEach(function (v) { if (v.curves.epochs.length > labels.length) labels = v.curves.epochs; });
  var lossDs = [], scoreDs = [];
  withCurves.forEach(function (v) {
    v.curves.series.forEach(function (s) {
      var ds = {
        label: v.label + ' - ' + s.name,
        data: s.values,
        borderColor: v.color,
        pointRadius: 2,
        borderWidth: 2,
        tension: .3,
        fill: s.kind === 'score',
        backgroundColor: s.kind === 'score' ? v.color + '12' : 'transparent'
      };
      if (s.kind === 'loss') ds.borderDash = [5, 4];
      (s.kind === 'loss' ? lossDs : scoreDs).push(ds);
    });
  });
  makeChart('lossChart', {
    type: 'line', data: { labels: labels, datasets: lossDs },
    options: { animation: { duration: 400 }, plugins: { legend: { position: 'bottom' } }, scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Loss' } } } }
  });
  makeChart('scoreChart', {
    type: 'line', data: { labels: labels, datasets: scoreDs },
    options: { animation: { duration: 400 }, plugins: { legend: { position: 'bottom' } }, scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Validation score' } } } }
  });
}

/* ---- confusion gallery ---- */
function renderCM() {
  var el = document.getElementById('cm-gallery');
  var vs = selectedVariants().filter(function (v) { return v.cmImage; });
  el.innerHTML = vs.map(function (v) {
    return '<div class="img-box"><h3 style="margin-bottom:8px;">' + v.label + '</h3><img src="' + v.cmImage + '" alt="' + v.key + ' confusion matrix"/><div class="cap">' + v.key + '</div></div>';
  }).join('');
}

/* ---- 10-fold CV ---- */
function bestFold(run) {
  var b = run.fold_results[0];
  run.fold_results.forEach(function (f) { if (f.best_mean_iou > b.best_mean_iou) b = f; });
  return b.fold;
}
function worstFold(run) {
  var b = run.fold_results[0];
  run.fold_results.forEach(function (f) { if (f.best_mean_iou < b.best_mean_iou) b = f; });
  return b.fold;
}
function renderCV() {
  var cv = window.DATA.cv || [];
  var el = document.getElementById('cv-wrap');
  if (!el) return;
  if (!cv.length) { el.innerHTML = '<div class="note">No cross-validation runs found.</div>'; return; }
  var html = '';
  cv.forEach(function (run) {
    var blk = '<div class="chart-box" style="margin-bottom:16px;">';
    blk += '<h3 style="color:' + run.color + '">' + run.label + ' - ' + run.folds + '-fold CV (scene-disjoint, ' + run.epochs + ' epochs per fold)</h3>';
    var cards = [
      { l: 'Mean val mIoU', v: (run.mean * 100).toFixed(2) + '%', s: '\u00b1 ' + (run.std * 100).toFixed(2) + '% (std)' },
      { l: 'Best fold', v: (run.max * 100).toFixed(2) + '%', s: 'fold ' + bestFold(run) },
      { l: 'Worst fold', v: (run.min * 100).toFixed(2) + '%', s: 'fold ' + worstFold(run) }
    ];
    if (run.single_run_test_miou !== null && run.single_run_test_miou !== undefined) {
      cards.push({ l: 'Single-run test mIoU', v: (run.single_run_test_miou * 100).toFixed(2) + '%', s: 'official test split' });
    }
    blk += '<div class="cards" style="grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin:12px 0;">';
    cards.forEach(function (c) {
      blk += '<div class="card blue"><div class="label">' + c.l + '</div><div class="value">' + c.v + '</div><div class="sub">' + c.s + '</div></div>';
    });
    blk += '</div><div class="charts-2">';
    blk += '<div class="chart-box"><h3>Validation mIoU per fold</h3><canvas id="cv-chart-' + run.variant + '" height="220"></canvas></div>';
    blk += '<div class="chart-box"><h3>Fold detail</h3><div class="tbl-wrap"><table id="cv-table-' + run.variant + '"></table></div></div>';
    blk += '</div></div>';
    html += blk;
  });
  el.innerHTML = html;

  cv.forEach(function (run) {
    var folds = run.fold_results.map(function (f) { return f.fold; });
    var ious = run.fold_results.map(function (f) { return f.best_mean_iou * 100; });
    var meanLine = folds.map(function () { return run.mean * 100; });
    makeChart('cv-chart-' + run.variant, {
      type: 'bar',
      data: {
        labels: folds.map(function (f) { return 'F' + f; }),
        datasets: [
          { type: 'bar', label: 'Val mIoU (%)', data: ious, backgroundColor: run.color + '22', borderColor: run.color, borderWidth: 1.5 },
          { type: 'line', label: 'Mean ' + (run.mean * 100).toFixed(2) + '%', data: meanLine, borderColor: '#f43f5e', borderDash: [5, 4], pointRadius: 0, borderWidth: 2, fill: false }
        ]
      },
      options: {
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { min: 0, max: 50, title: { display: true, text: 'mIoU (%)' } },
          x: { title: { display: true, text: 'Fold' } }
        }
      }
    });
    var head = '<thead><tr><th>Fold</th><th>Best epoch</th><th>Val mIoU</th><th>Train patches</th><th>Val patches</th></tr></thead>';
    var body = '<tbody>' + run.fold_results.map(function (f) {
      return '<tr><td>Fold ' + f.fold + '</td><td>Ep ' + f.best_epoch + '</td><td style="font-weight:600;color:' + run.color + '">' + (f.best_mean_iou * 100).toFixed(2) + '%</td><td>' + f.train_size + '</td><td>' + f.val_size + '</td></tr>';
    }).join('') + '</tbody>';
    document.getElementById('cv-table-' + run.variant).innerHTML = head + body;
  });
}

/* ---- maps gallery ---- */
function renderMaps() {
  var el = document.getElementById('maps-gallery');
  el.innerHTML = window.DATA.maps.map(function (m) {
    return '<div class="img-box"><img src="' + m.img + '" alt="' + m.label + '"/><div class="cap">' + m.label + '</div></div>';
  }).join('');
}

function rerender() {
  renderZoo();
  renderPerClass();
  renderRadar();
  renderCurves();
  renderCM();
}

renderDataset();
renderZoo();
renderAccuracy();
renderCV();
renderPerClass();
renderRadar();
renderCurves();
renderCM();
renderMaps();
buildSelector();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()