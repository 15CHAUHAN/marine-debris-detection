---
name: streamlit-dashboard-patterns
description: Development patterns and constraints for the MARIDA/TACO Streamlit interactive dashboard.
---

# Streamlit Interactive Dashboard Guidelines

1. **Architecture Separation of Concerns:**
   - `dashboard/app.py`: UI rendering and user interactions ONLY.
   - `dashboard/pipeline.py`: Model loading, inference, precision gating, bounding box drawing, and metrics.
   - `dashboard/geo.py`: Georeferencing, bounds calculation, Folium map construction, and GeoTIFF export.
   - Never import heavy training scripts directly into UI components.

2. **Caching Strategy:**
   - Model weights must be cached using `@st.cache_resource` to avoid reloading hundreds of MBs on every click.
   - Dataset metadata and patch lists must be cached using `@st.cache_data`.

3. **Session State Management:**
   - Persist loaded images, detections, and georeference coordinates across tabs using `st.session_state`.
   - When jumping between tabs programmatically, use `st.session_state["tab_nav"]` followed by `st.rerun()`.

4. **Leaflet / Folium Map Rendering:**
   - Wrap interactive maps in `streamlit_folium.st_folium(m, width=..., height=520, returned_objects=[])`.
   - Minimize returned objects to prevent slow bidirectional React-Streamlit serialization.
