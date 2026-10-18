# Reference: Caching & rerun patterns

## 1. Correct cache boundary in `dashboard/pipeline.py`

```python
from pathlib import Path
import numpy as np
import streamlit as st
import torch

# expensensive, shared, mutable-ish -> resource
@st.cache_resource(show_spinner="Loading model...")
def load_model(weights_path: str, device: str = "cuda") -> torch.nn.Module:
    from src.models.registry import build_model          # adjust to real API
    model = build_model(Path(weights_path).stem)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state))
    return model.to(device).eval()

# pure, cheap to copy -> data
@st.cache_data(show_spinner=False, max_entries=32)
def list_patches(patches_dir: str, split: str) -> list[str]:
    return sorted(p.stem for p in Path(patches_dir).glob(f"{split}_*.npz"))

# bytes, so cache_data copies kilobytes not megabytes
@st.cache_data(show_spinner=False, max_entries=128)
def mask_png_bytes(mask_id: str, masks_dir: str, palette: tuple) -> bytes:
    arr = np.load(Path(masks_dir) / f"{mask_id}.npy")
    return encode_palette_png(arr, palette)
```

## 2. Why an unhashable argument kills the cache

```python
@st.cache_data
def slow(arr: np.ndarray) -> int:      # NEVER
    return int(arr.sum())
```
Streamlit logs nothing and recomputes every call. Symptoms: the spinner appears on
every interaction even though the code "is cached". Fix by passing a hashable key:

```python
@st.cache_data
def slow_cached(content_hash: str, npy_path: str) -> int:
    return int(np.load(npy_path).sum())
```

## 3. Explicit invalidation control

```python
with st.sidebar:
    st.caption("Data loaded from disk is cached.")
    if st.button("Reload data", help="Clear cached patches, metrics and maps"):
        list_patches.clear()
        load_metrics.clear()
        build_map.clear()
        st.rerun()
```

## 4. Deferred compute pattern (avoids rerun cost)

```python
patch_id = st.selectbox("Patch", list_patches(str(ROOT / "patches"), "test"))

if st.button("Run inference", type="primary"):
    model = load_model(str(weights_path))
    with torch.inference_mode():
        result = run_inference(model, load_patch(patch_id))
    st.session_state["result"] = {"patch_id": patch_id, **result}

# render from session state: no recompute on unrelated widget changes
if (res := st.session_state.get("result")):
    if res["patch_id"] == patch_id:
        st.image(res["overlay"], use_container_width=True)
```

## 5. Figure lifecycle

```python
import matplotlib.pyplot as plt

def render_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)   # fresh every time
    try:
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        st.pyplot(fig, clear_figure=True)
    finally:
        plt.close(fig)                                 # prevents the leak
```

## 6. Map construction, cached and minimal

```python
import folium
import streamlit_folium

@st.cache_resource(show_spinner=False, max_entries=8)
def build_map(patch_id: str, bounds: tuple, opacity: float, basemap: str) -> folium.Map:
    m = folium.Map(location=((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2),
                   zoom_start=14, tiles=basemap)
    folium.Rectangle(bounds=[[bounds[0], bounds[2]], [bounds[1], bounds[3]]],
                     color="#e41a1c", fill_opacity=opacity).add_to(m)
    return m

# only the objects you use
streamlit_folium.st_folium(build_map(pid, b, op, base), height=520, returned_objects=[])
```

## 7. Rerun budget checklist

| Interaction | Allowed cost | Mechanism |
|---|---|---|
| Change display toggle | <100 ms | render-only, cached data |
| Switch patch | <300 ms | cached decode + cached PNG |
| Switch tab | <300 ms | `session_state["tab_nav"]` + `st.rerun()`, build map lazily |
| Run inference | ~1–3 s | button-gated, `inference_mode`, batched |
| Cold start | <5 s | cached model, lazy imports of heavy modules |

## 8. Troubleshooting slow dashboards

1. Check for unhashable cache args (§2) — most common cause.
2. Check for a global `plt.figure()` reused across reruns (memory growth).
3. Check whether a map is built for an inactive tab.
4. Run with `logging` at DEBUG and the `Phase` timer around `run_inference`.
5. Confirm `torch.cuda.empty_cache()` is not being called per rerun (it forces
   synchronization and is slower than leaving allocations pooled).
