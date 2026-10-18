---
name: streamlit-testing
description: Headless tests for the Streamlit dashboard using AppTest, with heavy model loading mocked out and session-state flows verified.
---

# Streamlit Dashboard Testing

## 1. Use AppTest, Never a Live Server
`streamlit.testing.v1.AppTest` runs `dashboard/app.py` in-process, headless, with
no port and no browser. Tests must never call `streamlit run`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dashboard.py -q
```

```python
from streamlit.testing.v1 import AppTest

def test_app_boots_without_exception():
    at = AppTest.from_file("dashboard/app.py", default_timeout=30).run()
    assert not at.exception, at.exception
```

- `default_timeout=30` (or higher) because model loading is slow; the default 3 s
  will flake on this project.
- Always assert `not at.exception` **first**; otherwise later assertions fail
  confusingly on a broken app.
- `AppTest.from_file` resolves relative to the CWD, so run pytest from the project
  root (see `pytest-patterns`).

## 2. Mock the Heavy Parts (mandatory)
The dashboard loads model weights and reads patches. Tests must not depend on
weights existing or on GPU time.

```python
import numpy as np
import pytest

@pytest.fixture
def app(monkeypatch):
    import dashboard.pipeline as pipeline
    monkeypatch.setattr(pipeline, "load_model", lambda *a, **k: DummyModel(), raising=True)
    monkeypatch.setattr(pipeline, "list_patches", lambda *a, **k: ["p0", "p1"], raising=True)
    monkeypatch.setattr(
        pipeline, "run_inference",
        lambda img, *a, **k: {"masks": np.zeros((11, 64, 64), np.uint8), "boxes": []},
        raising=True,
    )
    return AppTest.from_file("dashboard/app.py", default_timeout=30)
```

Rules:
- Patch at the module the app actually calls (`dashboard.pipeline`), not the
  dependency it imports from — `raising=True` catches a renamed symbol.
- Prefer tiny synthetic arrays (e.g. 64×64) over real patches or GeoTIFFs.
- Never let a test reach the network or `torch.hub`.

## 3. What To Actually Assert
| Layer | Assertion |
|---|---|
| Boot | `not at.exception`; expected title/header text present |
| Tabs | all tabs render (`len(at.tabs)`); tab switch reruns without exception |
| Session state | keys exist after an interaction; values survive a rerun |
| Dataframes/tables | expected columns; row count matches the mocked patch list |
| Widgets | setting `at.selectbox[0].set_value(...)` then `.run()` changes output |
| Errors | a mocked failure surfaces a visible message, not a traceback page |
| Metrics | Class-1 F1 / mIoU strings appear from mocked metrics |

```python
def test_tab_navigation_preserves_selection(app):
    at = app.run()
    at.session_state["tab_nav"] = "Geospatial"
    at.run()
    assert not at.exception
    assert at.session_state["tab_nav"] == "Geospatial"
```

## 4. Session-State Rules To Verify
`dashboard/app.py` persists loaded images, detections, and georeference coords
across tabs, and switches tabs via `st.session_state["tab_nav"]` + `st.rerun()`.
Tests should confirm:
- A rerun does not clear selections.
- No widget key is written after the widget is instantiated (that raises).
- `tab_nav` values used by the app are exactly the ones tests set.

## 5. Caching In Tests
- `@st.cache_resource` / `@st.cache_data` persist for the process. In tests, clear
  them so mocks take effect: `at.run()` uses a fresh script run per `AppTest`
  instance, but module-level caches in `dashboard/pipeline.py` do not reset.
- If a mock is ignored, the cache is the culprit: call the cached function's
  `.clear()` in the fixture, or add a `_cache_key` argument that varies per test.

## 6. Keep One Real Smoke Test
Exactly one test should exercise the real import path and real config with the
smallest available patch set, marked slow:

```python
@pytest.mark.slow
def test_dashboard_real_smoke():
    at = AppTest.from_file("dashboard/app.py", default_timeout=120).run()
    assert not at.exception
```
This catches import-time breakage that mocks hide. It may `pytest.skip` when
patches are not prepared (see `pytest-patterns` §4).

## 7. Anti-Patterns
- Launching `streamlit run` from a test.
- Asserting on exact rendered HTML or CSS classes — brittle across Streamlit
  versions (1.63 here).
- Using real weights, real videos (`4146796-hd_1920_1080_25fps.mp4`), or folders
  like `20210626/` in a unit test.
- Monkeypatching `streamlit` itself; patch your own `dashboard/` modules instead.
