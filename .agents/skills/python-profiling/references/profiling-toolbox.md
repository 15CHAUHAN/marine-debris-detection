# Reference: Profiling toolbox

## 1. Reproducible CPU profile of a script

```powershell
$env:PYTHONHASHSEED = "0"
.\.venv\Scripts\python.exe -X importtime -m cProfile -o prof.out scripts\evaluate_test.py
.\.venv\Scripts\python.exe -c @"
import pstats
s = pstats.Stats('prof.out')
s.strip_dirs().sort_stats('cumtime').print_stats(25)
s.sort_stats('tottime').print_stats(15)
"@
```
`-X importtime` surfaces slow imports (torch, rasterio, ultralytics) — useful when
the dashboard's cold start is the complaint rather than the compute.

## 2. Line-level timing without a profiler

```python
import contextlib, time
from dataclasses import dataclass, field

@dataclass
class Phase:
    name: str
    t0: float = field(default_factory=time.perf_counter)
    def __enter__(self): return self
    def __exit__(self, *exc):
        print(f"{self.name:<24} {time.perf_counter() - self.t0:8.3f}s")

with Phase("ingest"):
    ...
with Phase("forward"):
    ...
```
Cheap, greppable, and safe to leave in `__main__` blocks.

## 3. Dataloader throughput probe

```python
import time, torch
from src.data.factory import build_dataset   # adjust to the real factory

ds = build_dataset(...)
t0 = time.perf_counter()
for i, batch in enumerate(ds):          # num_workers=0 on Windows
    if i == 50: break
dt = time.perf_counter() - t0
print(f"{50/dt:.1f} patches/s  ({50} sampled)")
```

Then compute the budget: if the GPU step takes 0.20 s and the loader yields
1.8 patches/s, the loader is the bottleneck and a decoded cache is the fix.

## 4. GPU phase timing

```python
import torch
def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    print(f"{bench(lambda: model(x)):.2f} ms/iter")
```

## 5. Peak memory & leak check

```python
torch.cuda.reset_peak_memory_stats()
model(x)
print(torch.cuda.max_memory_allocated() / 2**20, "MB peak")
```

```python
import tracemalloc
tracemalloc.start()
run_long_loop()
snap = tracemalloc.take_snapshot()
for stat in snap.statistics("lineno")[:10]:
    print(stat)
```

## 6. Expected baseline shape (fill in on first run)

| Stage | Metric | Baseline | Target |
|---|---|---|---|
| Patch decode | patches/s | _measure_ | ≥ GPU consumption |
| Train epoch | s/epoch | _measure_ | — |
| Inference | ms/patch (256²) | _measure_ | — |
| Dashboard cold start | s | _measure_ | < 5 |
| Peak VRAM (train) | MB | _measure_ | < 5600 |
