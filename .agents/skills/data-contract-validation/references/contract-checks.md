# Reference: Concrete contract checks

Copy-paste ready validators. Keep them in `src/data/validation.py` (create it) so
both `src/` and `tests/` import one source of truth.

## 1. Patch tensor

```python
import numpy as np

NUM_CLASSES = 11
NUM_BANDS = 11

def validate_patch(x: np.ndarray, patch_id: str, size: int = 256) -> None:
    if x.ndim != 3 or x.shape[0] != NUM_BANDS:
        raise ValueError(f"{patch_id}: expected (11, H, W), got {x.shape}")
    if x.shape[1] != x.shape[2]:
        raise ValueError(f"{patch_id}: non-square patch {x.shape[1:]} (size={size})")
    if not np.isfinite(x).all():
        n = int((~np.isfinite(x)).sum())
        raise ValueError(f"{patch_id}: {n} NaN/Inf values after normalization")
    lo, hi = float(x.min()), float(x.max())
    if lo < -0.5 or hi > 1.5:
        raise ValueError(f"{patch_id}: normalized range [{lo:.3f}, {hi:.3f}] outside [0, 1]")
```

## 2. Split manifest

```python
def validate_splits(splits: dict[str, set[str]], all_ids: set[str]) -> None:
    expected = {"train": 694, "val": 328, "test": 359}
    for name, ids in splits.items():
        if name in expected and len(ids) != expected[name]:
            raise ValueError(f"split '{name}': expected {expected[name]} ids, got {len(ids)}")

    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = splits[a] & splits[b]
            if overlap:
                raise ValueError(f"split overlap {a}/{b}: {sorted(overlap)[:5]}")

    union = set().union(*splits.values())
    if union != all_ids:
        missing = sorted(all_ids - union)[:5]
        extra = sorted(union - all_ids)[:5]
        raise ValueError(f"split coverage mismatch; missing={missing} extra={extra}")
```

## 3. Scene purity (group leakage)

```python
import re
from collections import defaultdict

def scene_of(patch_id: str) -> str:
    # MARIDA ids embed the scene; adjust the regex to the real id format
    return re.split(r"_\d+$", patch_id)[0]

def validate_scene_disjoint(splits: dict[str, set[str]]) -> None:
    owner: dict[str, str] = {}
    for name, ids in splits.items():
        for pid in ids:
            scene = scene_of(pid)
            if owner.setdefault(scene, name) != name:
                raise ValueError(
                    f"scene '{scene}' leaks from '{owner[scene]}' into '{name}'"
                )
```

## 4. Georeference round trip

```python
import rasterio
from affine import Affine

def validate_georef(path: str, expected: Affine, expected_crs: str) -> None:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path}: missing CRS")
        if src.crs.to_string() != expected_crs:
            raise ValueError(f"{path}: CRS {src.crs.to_string()} != {expected_crs}")
        a, b = src.transform, expected
        if max(abs(a[i] - b[i]) for i in range(6)) > 1e-6:
            raise ValueError(f"{path}: transform drift {tuple(a)[:6]} != {tuple(b)[:6]}")
```

## 5. Checkpoint load guard

```python
import torch

def load_state_dict_strict(model: torch.nn.Module, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise KeyError(f"{ckpt_path}: missing keys {missing[:8]} ({len(missing)} total)")
    if unexpected:
        raise KeyError(f"{ckpt_path}: unexpected keys {unexpected[:8]} ({len(unexpected)} total)")
```

## 6. Leakage audit invariant (project directive)

```python
def validate_no_gan_in_eval(splits: dict[str, set[str]], synthetic: set[str]) -> None:
    for name in ("val", "test"):
        leaked = splits[name] & synthetic
        if leaked:
            raise ValueError(
                f"GAN/synthetic ids present in '{name}': {sorted(leaked)[:5]}"
            )
```

## 7. Test hookup

```python
import numpy as np, pytest
from src.data.validation import validate_patch

@pytest.mark.parametrize("bad, why", [
    (np.zeros((11, 255, 256), np.float32), "non-square"),
    (np.zeros((10, 256, 256), np.float32), "wrong band count"),
    (np.full((11, 256, 256), np.nan, np.float32), "NaN"),
    (np.full((11, 256, 256), 3.0, np.float32), "out of range"),
])
def test_validate_patch_rejects(bad, why):
    with pytest.raises(ValueError):
        validate_patch(bad, patch_id="p0")
```
