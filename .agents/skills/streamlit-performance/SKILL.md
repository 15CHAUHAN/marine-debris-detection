---
name: streamlit-performance
description: Caching boundaries, rerun avoidance, and map/image rendering rules to keep the MARIDA/TACO Streamlit dashboard responsive.
---

# Streamlit Performance

## 1. Caching Decision Table
Pick by what the function returns, not by how expensive it feels.

| Situation | Use |
|---|---|
| Model weights, sessions, Folium map objects, GPU handles | `@st.cache_resource` |
| Pure data: patch lists, DataFrames, metrics dicts, decoded masks | `@st.cache_data` |
| Everything else (per-interaction state, widgets) | `st.session_state` |
| Nothing — recompute is cheaper than hashing | no cache |

`@st.cache_data` serializes and copies its return value; `@st.cache_resource`
shares one object. Caching a large numpy array with `cache_data` copies it on every
call — for big masks prefer returning a path or bytes, or use `cache_resource` with
an explicit immutability comment.

## 2. Cache Keys: The Silent Killer
A cached function whose arguments are unhashable is **silently disabled** — you get
no warning, just slowness.

- Pass `str(path)`, never a `Path` or an open file handle.
- Never pass `np.ndarray`, `torch.Tensor`, or a DataFrame as a cache argument.
  Pass an id + a content hash instead.
- Sort any list/set argument before passing it (order changes the key).
- Keyword vs positional calls produce different keys — be consistent.

```python
@st.cache_data(show_spinner=False)
def load_patch_list(split: str, patches_dir: str) -> list[str]:
    return sorted(p.stem for p in Path(patches_dir).glob(f"{split}_*.npz"))

# caller
ids = load_patch_list("val", str(ROOT / "patches"))
```

## 3. Invalidation
- `ttl=` only for data that genuinely changes during a session (live metrics).
- `max_entries=` for map/figure caches to bound memory on a 6 GB machine.
- Provide an explicit "Reload data" button that calls `.clear()` on the relevant
  cached function — users need an escape hatch that is not a server restart.

## 4. Keep Reruns Cheap
- Streamlit reruns the whole script on every widget interaction. Expensive work
  must be behind a cache boundary or a button, never inline in the render path.
- Precompute on load, render on interaction: do inference when the user selects a
  patch and presses a button, then store the result in `st.session_state`.
- Avoid `st.rerun()` in loops; the app already uses it for tab switching — keep it
  to one call per user action.
- Keep widgets that do not affect heavy output outside the computation's dependency
  chain where possible (e.g. slide a display toggle without re-running inference).

## 5. Figures and Images
- Never reuse a global `plt.figure()`; create a fresh figure per render and close
  it (`plt.close(fig)`) — Streamlit keeps references and leaks memory otherwise.
- Downscale arrays before `st.image`: cap the long edge (e.g. 1024) and pass
  `use_container_width=True` rather than relying on CSS.
- Convert 11-class mask arrays to a palette PNG **once** and cache the bytes;
  re-rendering a raw `(11, H, W)` stack per rerun is the usual dashboard lag cause.
- Avoid `st.image` in a loop over hundreds of patches; paginate.

## 6. Folium / Leaflet
- Build the map once per `(patch_id, opacity, basemap)` key and cache it.
- Always call `streamlit_folium.st_folium(m, returned_objects=[])` and request only
  the returned objects you actually use — each returned object round-trips through
  React serialization.
- Set explicit `width`/`height` (the project uses height=520) to avoid layout
  thrash.
- Prefer a single FeatureGroup with many children over many separate layers.
- Do not create a map for a tab that is not active; gate construction on the
  current tab.

## 7. Inference Path
- `torch.inference_mode()`, not just `no_grad()`.
- Move the model to the device once (cached resource), not per call.
- Batch when the user requests several patches; single-patch calls waste the GPU.
- `num_workers=0` when reading patches inside the dashboard (Windows spawn).
- Cap tile/chip size; sliding-window inference should expose the window size as a
  config value, not a hardcoded constant.

## 8. Measuring
- Streamlit's "Rerun" timing is in the UI menu; correlate with
  `@st.cache_data` hit/miss behavior.
- For a suspected slow call, wrap it in the `Phase` context manager from
  `python-profiling` and log to the server console.
- Target numbers: cold start < 5 s, cached interaction < 300 ms, map render < 1 s.

## 9. Anti-Patterns
- `@st.cache_data` on a function that takes a DataFrame or tensor (silently off).
- Caching a model with `cache_data` — it deep-copies hundreds of MB each call.
- Loading weights on every tab switch.
- Rebuilding the Folium map on every rerun regardless of whether inputs changed.
- Reading the full 11-band GeoTIFF to display one patch preview.
