---
name: pytest-patterns
description: Pytest conventions, fixture scoping, markers, GPU-safety, and leakage-test rules for the TECH-ATHON test suite.
---

# Pytest Conventions & Test Rules

## 1. Invocation
- Always run from the project root so `import src...` resolves:
  `D:\TECH-ATHON\.venv\Scripts\python.exe -m pytest tests -q`
- Fast loop (skip slow/GPU work): `... -m "not gpu and not slow"`
- Coverage: `... --cov=src --cov-report=term-missing`
- Never call bare `python -m pytest`.

## 2. File Layout (mirror the package)
- One test file per module: `src/data/splits.py` → `tests/test_splits.py`.
- Docstring on line 1 naming the module under test (see `tests/test_splits.py`).
- Keep `tests/` flat; no nested packages unless a subpackage grows past ~10 modules.
- Shared helpers go in `tests/conftest.py` (create it when the third test file needs the same fixture). There is no `conftest.py` today.

## 3. Fixture Scoping
- `scope="session"` for artifacts that are immutable and expensive (`MaridaDataset`, loaded split manifests).
- `scope="module"` for per-file derived data (default choice in this repo).
- Function scope only for mutable state; never share a mutated fixture across tests.
- Fixtures must not write into `artifacts/` or `runs/`. Use pytest's `tmp_path` / `tmp_path_factory`.
- Return paths as `Path`, not `str`, and build them from `ROOT = Path(__file__).resolve().parent.parent`.

## 4. Optional-Data Guard
Dataset-dependent tests must skip instead of fail when preparation has not run (existing pattern in `tests/test_plastic_l2a.py`):

```python
if not (ROOT / "patches").exists():
    pytest.skip("plastic_l2a artifacts not prepared")
```

Guard on the specific artifact directory (`patches/`, `MARIDA/`, `20210626/`), not on the whole project.

## 5. GPU Tests
- Start with `torch.cuda.is_available()` and skip if absent — the suite must pass on CPU-only machines.
- `num_workers=0` is mandatory (Windows spawn deadlock; see `windows-powershell-env`).
- Wrap forward passes in `torch.inference_mode()` or `torch.no_grad()`.
- Free memory at teardown: `torch.cuda.empty_cache()`.
- Keep batch sizes ≤ 2 in tests for the 6 GB RTX 4050; never train in a unit test.
- Add `@pytest.mark.gpu` so CI can exclude these.

## 6. Parametrization
- Prefer `@pytest.mark.parametrize` over loops that re-load fixtures — the pattern already used in `tests/test_models.py` for `VARIANTS` and builder functions.
- Give each case an `id=` so failures name the variant.
- Parametrize over *classes* (registry variants), not over random seeds (see §7).

## 7. Determinism
- Seed in a fixture: `torch.manual_seed(0)`, `np.random.seed(0)`, `random.seed(0)`.
- Do not compare float tensors exactly; use `pytest.approx` or `torch.allclose(..., atol=1e-6)`.
- Do not let tests depend on wall-clock, network, or file mtimes.

## 8. Invariants That Must Stay Tested
Never delete or relax these without an explicit decision:
- **Leakage**: `tests/test_leakage_audit.py` and the scene-disjoint assertions in `tests/test_splits.py` guard the "no data leakage" project directive. Official train/val/test counts are 694 / 328 / 359 over 1381 patches.
- **Class contract**: 11-class label space with class 1 = marine debris (`tests/test_encoding.py`, `tests/test_losses.py`).
- **Georeferencing**: `tests/test_mapping.py` must assert the Affine transform survives the round trip.

## 9. Markers
Register markers in `pytest.ini` (does not exist yet) so `--strict-markers` stays viable. See `references/pytest-ini.md`.

## 10. Anti-Patterns
- Downloading data or weights inside a test.
- Assertions with no message on multi-value comparisons — include the offending patch id.
- `assert` for validating *production* inputs (stripped under `-O`); raise `ValueError` there instead (see `data-contract-validation`).
- Letting tests mutate `splits/` or checkpoint files in place.
