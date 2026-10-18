"""Georeferencing helpers and the interactive folium pollution map.

Converts pixel locations to real WGS84 coordinates through the source GeoTiff
CRS/transform using rasterio.warp (bundled PROJ). No pyproj/geopandas needed.

Map geometry functions are pure (no streamlit) and unit-testable; the folium
builder depends only on folium, imported lazily so tests can skip it.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
import rasterio
from rasterio.warp import transform as warp_coords, transform_bounds as warp_bounds

WGS84 = "EPSG:4326"

SEVERITY_COLORS = {
    "Low": "#2ca02c",
    "Moderate": "#e6c60c",
    "High": "#ff7f0e",
    "Critical": "#d62728",
}


def severity_color(level: str) -> str:
    """Marker/polygon color for a pollution severity level."""
    return SEVERITY_COLORS.get(level, "#000000")


def pixel_area_m2_from_transform(transform: tuple) -> float | None:
    """Ground area of one pixel (m^2) from an affine transform, else None."""
    if not transform or not transform[0] or not transform[4]:
        return None
    return abs(transform[0] * transform[4])


def crs_bounds(width: int, height: int, transform: tuple) -> tuple:
    """(left, bottom, right, top) in the image CRS from an affine transform."""
    left = transform[2]
    top = transform[5]
    right = left + width * transform[0]
    bottom = top + height * transform[4]
    return left, bottom, right, top


def wgs84_bounds(
    width: int,
    height: int,
    transform: tuple,
    crs: str | None,
    densify_pts: int = 65,
) -> tuple:
    """WGS84 (west, south, east, north) bounds for the image extent."""
    left, bottom, right, top = crs_bounds(width, height, transform)
    if not crs or crs == WGS84:
        return left, bottom, right, top
    return warp_bounds(crs, WGS84, left, bottom, right, top, densify_pts=densify_pts)


def pixel_xy_to_wgs84(
    width: int,
    height: int,
    transform: tuple,
    crs: str | None,
    cols,
    rows,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel (col, row) centers -> (lons, lats) WGS84 arrays."""
    from rasterio.transform import Affine

    affine = transform if isinstance(transform, Affine) else Affine(*transform)
    xs, ys = rasterio.transform.xy(affine, rows, cols, offset="center")
    if crs and crs != WGS84:
        lons, lats = warp_coords(crs, WGS84, xs, ys)
        return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def corner_wgs84(width: int, height: int, transform: tuple, crs: str | None) -> list:
    """Closed polygon ring of WGS84 (lat, lon) corners (5 points)."""
    cols = [0, width - 1, width - 1, 0, 0]
    rows = [0, 0, height - 1, height - 1, 0]
    lons, lats = pixel_xy_to_wgs84(width, height, transform, crs, cols, rows)
    return list(zip(lats.tolist(), lons.tolist()))


def concentration_grid(
    concentration: np.ndarray,
    grid_n: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample a concentration map to an (grid_n, grid_n) cell grid.

    Returns (cols, rows, means) where rows[i]/cols[j] are the pixel-space
    center coordinates of cell (i, j) and means[i, j] its mean concentration.
    """
    h, w = concentration.shape
    means = np.zeros((grid_n, grid_n), dtype=np.float32)
    for i in range(grid_n):
        r0, r1 = i * h // grid_n, (i + 1) * h // grid_n
        for j in range(grid_n):
            c0, c1 = j * w // grid_n, (j + 1) * w // grid_n
            means[i, j] = float(np.mean(concentration[r0:r1, c0:c1]))
    half_h = max(h // grid_n // 2, 0)
    half_w = max(w // grid_n // 2, 0)
    rows = np.clip([i * h // grid_n + half_h for i in range(grid_n)], 0, h - 1)
    cols = np.clip([j * w // grid_n + half_w for j in range(grid_n)], 0, w - 1)
    return cols.astype(int), rows.astype(int), means


def concentration_color(value: float) -> str:
    """Green -> Yellow -> Red hex color for a concentration in [0, 1]."""
    from matplotlib import colormaps

    rgba = colormaps["RdYlGn_r"](min(max(float(value), 0.0), 1.0))
    r, g, b = (int(round(c * 255)) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def concentration_rgba(concentration: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """(H, W, 4) uint8 RGBA heatmap with transparent no-data (hot colormap).

    Passed directly to folium's ImageOverlay, which base64-encodes the array
    itself, so no intermediate PNG bytes are needed.
    """
    from matplotlib import colormaps

    scaled = np.clip(concentration, 0.0, 1.0)
    rgba = colormaps["hot"](scaled)
    rgba[concentration <= threshold, :] = 0
    return (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)


def concentration_png(concentration: np.ndarray, threshold: float = 0.0) -> bytes:
    """Transparent PNG bytes of the concentration heatmap (hot colormap).
    Used for in-app display; the folium overlay uses concentration_rgba.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=72)
    ax.imshow(concentration_rgba(concentration, threshold), interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def synthetic_meta(
    width: int,
    height: int,
    center_lat: float,
    center_lon: float,
    gsd_m: float,
) -> tuple[tuple, str]:
    """Approximate a local WGS84 frame for an ungeoreferenced image.

    Returns (transform, crs="EPSG:4326") treating each pixel as gsd_m on the
    ground; good enough to anchor drone photos / video frames on a map.
    """
    lat_per_px = gsd_m / 110540.0
    lon_per_px = gsd_m / (111320.0 * max(float(np.cos(np.radians(center_lat))), 0.01))
    a = lon_per_px
    e = -lat_per_px
    c = center_lon - (width / 2.0) * a
    f = center_lat - (height / 2.0) * e
    return (a, 0.0, c, 0.0, e, f), WGS84


def build_pollution_map(
    width: int,
    height: int,
    transform: tuple,
    crs: str | None,
    concentration: np.ndarray,
    severity: str,
    tile: str = "OpenStreetMap",
    heatmap_on: bool = True,
    grid_n: int = 16,
    zoom_start: int = 14,
):
    """Build a folium Map: patch polygon + colored debris cells + heatmap overlay.

    Circle markers are colored Green -> Yellow -> Red by local debris
    concentration and keep an on/off toggle via the folium layer control.
    """
    import folium

    west, south, east, north = wgs84_bounds(width, height, transform, crs)
    center = [(south + north) / 2.0, (west + east) / 2.0]
    tile_urls = {
        "Esri World Imagery": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
    }
    url = tile_urls.get(tile)
    if url is not None:
        m = folium.Map(location=center, zoom_start=zoom_start,
                       tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)
        folium.TileLayer(
            tiles=url,
            attr="Esri World Imagery &copy; Esri & contributors",
            name="Esri World Imagery",
            show=True,
        ).add_to(m)
    else:
        m = folium.Map(location=center, zoom_start=zoom_start, tiles=tile,
                       control_scale=True, prefer_canvas=True)

    corners = corner_wgs84(width, height, transform, crs)
    color = severity_color(severity)
    folium.Polygon(
        locations=corners,
        color=color,
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.12,
        tooltip=f"Patch extent | severity {severity}",
    ).add_to(m)

    cols, rows, means = concentration_grid(concentration, grid_n)
    cells_rc = [(i, j) for i in range(grid_n) for j in range(grid_n)]
    cell_cols = [cols[j] for _, j in cells_rc]
    cell_rows = [rows[i] for i, _ in cells_rc]
    lons, lats = pixel_xy_to_wgs84(width, height, transform, crs, cell_cols, cell_rows)
    cells = folium.FeatureGroup(name="Debris concentration cells", show=True).add_to(m)
    for idx, (i, j) in enumerate(cells_rc):
            value = float(means[i, j])
            if value <= 0.01:
                continue
            c = concentration_color(value)
            folium.CircleMarker(
                location=[lats[idx], lons[idx]],
                radius=3.0 + 12.0 * value,
                color=c,
                fill=True,
                fill_color=c,
                fill_opacity=0.65,
                tooltip=f"concentration {value:.2f}",
            ).add_to(cells)

    if heatmap_on:
        heat = folium.FeatureGroup(name="Concentration heatmap overlay", show=True).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=concentration_rgba(concentration),
            bounds=((south, west), (north, east)),
            opacity=0.55,
            interactive=True,
        ).add_to(heat)

    folium.LayerControl().add_to(m)
    return m