# Reference: pytest.ini template

The project has no pytest configuration file. Adding one makes markers strict and
keeps the default loop fast. Place at `D:\TECH-ATHON\pytest.ini`.

```ini
[pytest]
testpaths = tests
addopts = -q --strict-markers --strict-config
markers =
    gpu: requires CUDA and the local RTX 4050
    slow: takes longer than ~30s (training, full-dataset scans)
    data: requires prepared datasets under patches/ MARIDA/ 20210626/
filterwarnings =
    ignore::DeprecationWarning:torch.*
```

Notes:
- `--strict-markers` turns a typo like `@pytest.mark.gpuu` into a hard error.
- Do not add `--cov` to `addopts`; keep coverage opt-in for the slow loop.
- If `rootdir` detection ever picks the wrong directory, set `rootdir = .`
  explicitly rather than adding `sys.path` hacks to test files.

## Fast / full command pairs

```powershell
# fast: pure logic, no GPU, no datasets
.\.venv\Scripts\python.exe -m pytest tests -m "not gpu and not slow and not data" -q

# everything, with coverage
.\.venv\Scripts\python.exe -m pytest tests --cov=src --cov-report=term-missing -q

# only the leakage guards (run before every faculty demo)
.\.venv\Scripts\python.exe -m pytest tests\test_leakage_audit.py tests\test_splits.py -q
```

## Module-scoped fixture example that matches this repo

```python
from pathlib import Path
import pytest
from src.data.ingestion import MaridaDataset

ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session")
def dataset() -> MaridaDataset:
    if not (ROOT / "MARIDA").exists():
        pytest.skip("MARIDA dataset not present")
    return MaridaDataset(ROOT)
```

## GPU test skeleton

```python
@pytest.mark.gpu
def test_unet_forward_shapes(unet_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = unet_factory().cuda().eval()
    x = torch.randn(2, 11, 256, 256, device="cuda")
    try:
        with torch.inference_mode():
            y = model(x)
        assert y.shape == (2, 11, 256, 256), f"unexpected output shape {tuple(y.shape)}"
    finally:
        torch.cuda.empty_cache()
```
