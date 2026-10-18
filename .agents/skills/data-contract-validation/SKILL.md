---
name: data-contract-validation
description: Validate dataset shapes, label ranges, split manifests, and georeferencing contracts at ingestion and split boundaries, with fail-loud errors instead of silent corruption.
---

# Data Contract Validation

## 1. Why This Exists
Silent shape, dtype, or label corruption is the most expensive bug class in this
project: it produces plausible-looking training curves and invalid results. A
contract check at the boundary turns a 6-hour training run into a one-line error.

Validate at **boundaries only**: ingestion, split creation/loading, `__getitem__`,
and checkpoint load. Do not re-validate inside matmuls or losses.

## 2. The Core Contracts

### Patch tensors (Sentinel-2)
| Property | Contract |
|---|---|
| Shape | `(11, H, W)`, `H == W` (256 for the standard pipeline) |
| dtype | `float32` after normalization; `uint16`/`uint8` only pre-normalization |
| Range | normalized to ~`[0, 1]` via 2%–98% percentile scaling |
| Band order | the fixed 11-channel order defined in `geospatial-rasterio-processing` |
| NaN/Inf | zero tolerated — raise if `not np.isfinite(x).all()` |

### Labels
| Property | Contract |
|---|---|
| Shape | `(H, W)`, integer |
| Class space | exactly 11 classes, ids `0..10` |
| Class 1 | **marine debris** — the metric that matters |
| Out-of-range | any id > 10 or negative is a hard error unless it is the documented void/ignore value |
| Split purity | val/test labels are never augmented, synthesized, or GAN-generated |

### Splits
| Invariant | Check |
|---|---|
| Official counts | train 694, val 328, test 359 |
| Coverage | union of the three splits == every patch id in the dataset (1381) |
| Disjointness | pairwise intersection is empty |
| Scene-level | no scene contributes patches to more than one split (`assert_scene_disjoint`) |
| Types | ids are `str`, paths are `Path`, no `Path`/`str` mixing in manifests |

### Georeferencing
| Property | Check |
|---|---|
| Transform | the `Affine` from source survives any crop/tile/write round trip |
| CRS | preserved and asserted non-null before map export |
| Bounds | derived bounds match the array shape × transform |

## 3. Fail Loudly, Not With `assert`
Production validation must survive `python -O`:

```python
def validate_labels(labels: np.ndarray, patch_id: str, num_classes: int = 11) -> None:
    if labels.ndim != 2:
        raise ValueError(f"{patch_id}: labels must be 2-D HxW, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{patch_id}: labels must be integer dtype, got {labels.dtype}")
    bad = labels[(labels < 0) | (labels >= num_classes)]
    if bad.size:
        raise ValueError(
            f"{patch_id}: {bad.size} label values outside 0..{num_classes - 1}; "
            f"examples={np.unique(bad)[:5].tolist()}"
        )
```

Rules: name the offending id, include the observed value, and state the expected
range. A message that says "invalid data" costs another hour of debugging.

## 4. Where To Call It
| Location | Validate |
|---|---|
| `src/data/ingestion.py` | scene presence, band count, dtype, CRS |
| `src/data/preprocessing.py` | output shape/range after normalization |
| `src/data/splits.py` | counts, coverage, disjointness, scene purity |
| `src/data/dataset.py` `__getitem__` | shape + label range (cheap checks only; hot path) |
| `src/data/gan_augment.py` | synthetic samples carry the `synthetic` flag and never enter val/test |
| `src/utils/checkpoint.py` | state-dict key/shape match before loading |
| `src/mapping/georeference.py` | transform + CRS before export |
| `scripts/audit_data_leakage.py` | the full audit across prepared artifacts |

## 5. Optional Dependencies
`pydantic` and `pandera` are **not** in `requirements.txt`. For manifests and
configs prefer stdlib + explicit checks, or add a dependency deliberately:

- Add to `requirements.txt` with a pinned version and install via the venv.
- Never import an optional validator unconditionally inside `src/` — guard it, or
  the pipeline breaks on a clean install.
- Numpy/pathlib checks are sufficient for arrays; pydantic earns its keep for
  run configs and manifests (see `cli-and-config`).

## 6. Repo Schema for New Manifests
When adding a JSON manifest, include a `schema_version` integer so future
migrations are detectable, and validate it on load — reject unknown versions with
`ValueError` rather than guessing.

## 7. Anti-Patterns
- `assert` for production validation.
- Validating inside the training loop (cost, and it hides the real boundary).
- Clamping out-of-range labels instead of raising — this fabricates data.
- Assuming band order from array position without checking the band list.
- Treating `0` as "no data" when `0` is a legitimate class (background). Use an
  explicit void value and document it.
