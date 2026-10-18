---
name: geospatial-rasterio-processing
description: Instructions for parsing, stacking, normalizing, and slicing 11-channel Sentinel-2 multispectral GeoTIFFs using rasterio.
---

# Multispectral Satellite & Rasterio Processing Guidelines

1. **Standard 11-Channel Band Specification:**
   Every satellite input must conform strictly to the 11 Sentinel-2 bands in exact index order:
   - Index 0: Band 1 (B01 - 60m Coastal Aerosol)
   - Index 1: Band 2 (B02 - 10m Blue)
   - Index 2: Band 3 (B03 - 10m Green)
   - Index 3: Band 4 (B04 - 10m Red)
   - Index 4: Band 5 (B05 - 20m Vegetation Red Edge 1)
   - Index 5: Band 6 (B06 - 20m Vegetation Red Edge 2)
   - Index 6: Band 7 (B07 - 20m Vegetation Red Edge 3)
   - Index 7: Band 8 (B08 - 10m NIR Broad)
   - Index 8: Band 8A (B8A - 20m Narrow NIR)
   - Index 9: Band 11 (B11 - 20m SWIR 1)
   - Index 10: Band 12 (B12 - 20m SWIR 2)

2. **Display RGB Composite:**
   - Red = Index 3 (B04)
   - Green = Index 2 (B03)
   - Blue = Index 1 (B02)
   - To make an RGB array for visual inspection: `rgb = image[[3, 2, 1], :, :].transpose(1, 2, 0)`

3. **Percentile Normalization:**
   - Satellite bands contain raw surface reflectance (often 0 - 10000 DN).
   - Never directly cast to uint8. Apply band-wise 2% to 98% percentile clipping and scale to [0.0, 1.0] float32:
     ```python
     low, high = np.percentile(finite_vals, [2.0, 98.0])
     normalized = np.clip((band - low) / (high - low + 1e-6), 0.0, 1.0)
     ```

4. **Georeferencing & Affine Preservation:**
   - When extracting 256x256 patches or generating heatmaps, preserve CRS and update the Affine transform using `rasterio.windows.transform(window, src.transform)`.
   - Never discard CRS or transform metadata when exporting GeoTIFF outputs.
