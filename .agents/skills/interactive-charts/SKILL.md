---
name: interactive-charts
description: Chart library choice, the fixed 11-class color convention, and interaction/export rules for MARIDA results and dashboard figures.
---

# Interactive Charts & Visualization

## 1. Library Selection
Choose the lightest tool that does the job:

| Need | Use |
|---|---|
| Quick bar/line summary in the dashboard | `st.bar_chart`, `st.line_chart`, `st.scatter_chart` |
| Declarative, layered, linked selections | Altair via `st.altair_chart` |
| Rich zoom/hover/3-D, exports to HTML | Plotly via `st.plotly_chart` |
| Final figure for the PDF/report and docs | matplotlib → `savefig` PNG |

Reach for Plotly only when its interaction is actually used; it ships a large JS
bundle and slows first paint.

## 2. The 11-Class Color Convention (non-negotiable)
The label space is fixed at 11 classes with **class 1 = marine debris**. Keeping
one shared mapping means the legend looks identical in the dashboard, in exported
plots, and in the report.

```python
# src/utils/palette.py  (single source of truth)
CLASS_NAMES = [
    "Background", "Marine Debris", "Sargassum", "Natural-Organic", "Ship",
    "Clouds", "Water", "Foam", "Sediment", "Shore", "Other",
]
CLASS_COLORS = [
    "#d9d9d9",  # 0  background  - neutral, must recede
    "#e41a1c",  # 1  marine debris - most salient, reserved color
    "#4daf4a",  # 2  sargassum
    "#984ea3",  # 3  natural-organic
    "#ff7f00",  # 4  ship
    "#a6cee3",  # 5  clouds
    "#1f78b4",  # 6  water
    "#ffffff",  # 7  foam
    "#b15928",  # 8  sediment
    "#fdc086",  # 9  shore
    "#666666",  # 10 other
]
```

Rules:
- Class 1 keeps `#e41a1c` (red) everywhere. Never reuse that hex for other classes.
- Background stays low-saturation so it visually recedes behind detections.
- Do not use `jet`/`rainbow` for class maps or confusion matrices — not
  colorblind-safe and creates false gradient impressions between unrelated classes.
- Verify adjacent classes stay distinguishable in grayscale (the splittable
  palette above is colorblind-friendly; foam/white is why §3 exists).
- Class 1 F1 is always rendered in its own callout, not folded into macro averages.

## 3. Plot-Type Rules
| Plot | Rule |
|---|---|
| Confusion matrix | single-hue sequential (`Blues`), annotate counts, normalize row-wise and say which axis is normalized |
| Per-class F1 | horizontal bars sorted descending, class 1 highlighted in `#e41a1c`, others gray |
| Class map overlay | categorical palette + explicit legend; never continuous colormap for class ids |
| Loss curves | train/val in the same axes, distinct line styles so they survive grayscale printing |
| Precision/recall vs threshold | mark the chosen operating point explicitly |
| Geo coverage | Folium `CircleMarker`/`Rectangle`; heatmap for density (see `src/mapping/heatmap.py`) |

## 4. Interaction
- Filters: `st.multiselect` for classes, `st.select_slider` for confidence threshold.
- Linked brushing across views: Altair `selection` bound to the same field, or keep
  the selection in `st.session_state` and re-render.
- Every interactive chart state (selected class, threshold) lives in
  `session_state` so tab switches preserve it.
- Tooltips must carry real values (class name, F1, support count) — not just "value".
- Never encode information by color alone; pair with a label, marker shape, or
  direct annotation.

## 5. Performance
- Downsample before plotting: >10k points → decimate in numpy or bin into a hexbin.
- `st.plotly_chart(fig, use_container_width=True)` and avoid rebuilding the figure
  on each rerun (cache the figure or its data — see `streamlit-performance`).
- Plot the aggregated confusion matrix / curve, not per-patch series, in summary view.
- Reuse the palette module; do not build color dicts inline in `dashboard/app.py`.

## 6. Axis & Labeling Discipline
- Always label axes with units — band name and wavelength for spectral plots,
  patch count for histograms, confidence for thresholds.
- Fix axis limits when comparing runs, so a visually "better" curve is not just a
  different scale.
- Show class support (counts) next to F1 so a rare class's score is interpretable.
- State the metric definition in the caption (macro vs micro, per-class).

## 7. Export for the Report
- matplotlib remains the export path (`dashboard/report_generator.py`,
  `scripts/generate_project_document.py`, ReportLab).
- Write to `artifacts/<run_id>/plots/` with a deterministic filename so the report
  build is reproducible.
- Use `dpi=200` for report figures, `bbox_inches="tight"`, and embed the run_id in
  the caption.

## 8. Anti-Patterns
- A different color for the same class in two views.
- Rainbow colormaps for categorical classes.
- Pie charts for 11-class distributions (use a sorted bar or a stacked bar).
- Interactive Plotly inside a loop (one figure per patch × hundreds of rows).
- Charting metrics that came from different split manifests as if comparable.
