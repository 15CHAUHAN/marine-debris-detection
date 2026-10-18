# Reference: AppTest recipes

## 1. Inspect what the app exposes

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("dashboard/app.py", default_timeout=60).run()
print("exception:", at.exception)
print("tabs:", [t.label for t in at.tabs])
print("buttons:", [b.label for b in at.button])
print("selectboxes:", [s.label for s in at.selectbox])
print("dataframes:", len(at.dataframe))
print("session_state keys:", sorted(at.session_state.filtered_state))
```
`at.tabs` is only populated for `st.tabs`; if the app builds tabs from
`st.session_state["tab_nav"]` + `st.rerun()`, assert on `session_state` instead.

## 2. Interaction patterns

```python
# click, then re-run
at.button[0].click().run()

# set a selectbox value
at.selectbox[0].select("p0").run()

# multi-select
at.multiselect[0].set_value(["debris", "sargassum"]).run()

# slider / checkbox
at.slider[0].set_value(0.6).run()
at.checkbox[0].check().run()

# file-like inputs: write a temp file and point the app at it via env/mock
```

## 3. Asserting errors and messages

```python
at = AppTest.from_file("dashboard/app.py").run()
if at.exception:
    print(at.exception[0].value)        # traceback text
    print(at.exception[0].message)      # short message

# user-facing error surfaced with st.error
assert any("patch" in e.value.lower() for e in at.error)
assert at.success or at.warning        # whichever the app uses
```

## 4. Session-state helpers

```python
def test_state_keys_created(app):
    at = app.run()
    at.button[0].click().run()
    for key in ("tab_nav", "loaded_image", "detections"):
        assert key in at.session_state, f"missing session key {key}"
```

Writing state before `run()` seeds the first render:

```python
at = AppTest.from_file("dashboard/app.py", default_timeout=30)
at.session_state["tab_nav"] = "Geospatial"
at.run()
```

## 5. Full fixture with cache clearing

```python
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

class DummyModel:
    def eval(self): return self
    def __call__(self, x): return np.zeros((x.shape[0], 11, 64, 64), np.float32)

@pytest.fixture
def app(monkeypatch):
    import dashboard.pipeline as pipeline
    import dashboard.geo as geo

    for fn, value in {
        "load_model": DummyModel,
        "run_inference": lambda img, *a, **k: {
            "masks": np.zeros((11, 64, 64), np.uint8),
            "boxes": [],
            "confidence": 0.9,
        },
    }.items():
        if hasattr(pipeline, fn):
            monkeypatch.setattr(pipeline, fn, value, raising=True)

    # clear any cache_resource that memoized a real model earlier in the session
    for name in dir(pipeline):
        obj = getattr(pipeline, name)
        if hasattr(obj, "clear") and callable(obj.clear):
            obj.clear()

    return AppTest.from_file("dashboard/app.py", default_timeout=30)
```

## 6. Testing the pipeline and geo modules directly

AppTest is for UI wiring. Pure logic is faster and clearer as normal unit tests:

```python
import numpy as np
import pytest

def test_georeference_bounds_roundtrip():
    from dashboard.geo import bounds_from_transform
    ...
    pytest.importorskip("rasterio")

def test_precision_gating_threshold():
    from dashboard.pipeline import gate_by_precision
    assert gate_by_precision(0.2, threshold=0.5) is False
```

Rule of thumb: if a behaviour does not involve a Streamlit widget, do not test it
through AppTest.

## 7. Timing and flakiness

- Model-free tests should finish in <1 s each; keep them unmarked.
- A 30 s timeout means a *test* hung, not that the app is slow — check for a real
  model load that the fixture failed to mock.
- Failures that appear only when run after other dashboard tests indicate shared
  `@st.cache_*` state; fix by clearing caches in the fixture.
