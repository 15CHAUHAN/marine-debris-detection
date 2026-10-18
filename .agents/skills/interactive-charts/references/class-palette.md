# Reference: Class palette & metric chart recipes

## 1. Palette module and helpers

```python
"""src/utils/palette.py - single source of truth for class colors."""
from __future__ import annotations

import numpy as np

CLASS_NAMES = [
    "Background", "Marine Debris", "Sargassum", "Natural-Organic", "Ship",
    "Clouds", "Water", "Foam", "Sediment", "Shore", "Other",
]
CLASS_COLORS = [
    "#d9d9d9", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00",
    "#a6cee3", "#1f78b4", "#ffffff", "#b15928", "#fdc086", "#666666",
]
MARINE_DEBRIS = 1
NUM_CLASSES = len(CLASS_NAMES)

assert len(CLASS_COLORS) == NUM_CLASSES

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))   # type: ignore[return-value]

def palette_lut() -> np.ndarray:
    """(256, 3) uint8 LUT compatible with PIL 'P' mode images."""
    lut = np.zeros((256, 3), np.uint8)
    for i, h in enumerate(CLASS_COLORS):
        lut[i] = hex_to_rgb(h)
    return lut

def decode_mask(mask: np.ndarray) -> np.ndarray:
    """(H, W) class ids -> (H, W, 3) uint8 RGB."""
    if mask.max() >= NUM_CLASSES:
        raise ValueError(f"class id {int(mask.max())} outside 0..{NUM_CLASSES - 1}")
    return palette_lut()[mask.astype(np.uint8)]

def encode_palette_png(mask: np.ndarray, palette=None) -> bytes:
    """Palette PNG bytes for st.image (cache these; do not re-encode per rerun)."""
    import io
    from PIL import Image
    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(palette_lut().flatten().tolist())
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
```

## 2. Per-class F1 bar chart with class 1 highlighted

```python
import altair as alt
import pandas as pd
import streamlit as st
from src.utils.palette import CLASS_COLORS, CLASS_NAMES, MARINE_DEBRIS

def per_class_f1_chart(f1: list[float]) -> alt.Chart:
    df = pd.DataFrame({
        "class": CLASS_NAMES,
        "f1": f1,
        "highlight": [i == MARINE_DEBRIS for i in range(len(CLASS_NAMES))],
    }).sort_values("f1", ascending=False)

    return (
        alt.Chart(df, height=320)
        .mark_bar()
        .encode(
            x=alt.X("f1:Q", scale=alt.Scale(domain=[0, 1]), title="F1 score"),
            y=alt.Y("class:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.highlight,
                alt.value(CLASS_COLORS[MARINE_DEBRIS]),
                alt.value("#c9c9c9"),
            ),
            tooltip=["class:N", alt.Tooltip("f1:Q", format=".3f")],
        )
        .configure_view(stroke=None)
    )
```

## 3. Confusion matrix (row-normalized, annotated)

```python
import matplotlib.pyplot as plt
import numpy as np
from src.utils.palette import CLASS_NAMES

def confusion_figure(cm: np.ndarray, normalize: bool = True):
    if cm.shape != (len(CLASS_NAMES),) * 2:
        raise ValueError(f"expected {len(CLASS_NAMES)}x{len(CLASS_NAMES)}, got {cm.shape}")
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=110)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max() or 1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix (row-normalized)" if normalize else "Confusion matrix (counts)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            v = shown[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(cm[i, j])}",
                        ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 * shown.max() else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class" if normalize else "count")
    fig.tight_layout()
    return fig
```

## 4. Loss curves that survive grayscale

```python
def loss_figure(history: list[dict]):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history],
            "-", color="#1f78b4", label="train")
    ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history],
            "--", color="#e41a1c", label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
```
Distinct color **and** linestyle: the report may be printed in black and white.

## 5. Density heatmap (delegates to the mapping module)

```python
# src/mapping/heatmap.py already builds the density surface; the dashboard only styles it
from src.mapping.heatmap import build_density
import folium
from folium.plugins import HeatMap

def debris_heatmap(m, points: list[tuple[float, float]], radius: int = 12) -> folium.Map:
    if points:
        HeatMap(points, radius=radius, blur=15, min_opacity=0.35).add_to(m)
    return m
```

## 6. Accessibility check

- Contrast: `#ffffff` (foam) on a light background is invisible — outline its
  legend swatch with a 1 px dark stroke.
- Never rely on red/green alone; class 1 red vs class 2 green is the risky pair, so
  always show the legend with names and, in tables, the numeric value.
- Provide a text summary beside every chart (top-3 classes by F1, the class-1 value)
  so the information is available without color perception.
