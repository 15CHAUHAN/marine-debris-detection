---
name: python-profiling
description: Measure-first profiling workflow for the MARIDA data pipeline, PyTorch training loops, and the Streamlit dashboard on the RTX 4050.
---

# Profiling & Performance Work

## 1. Measure First, Always
- Never optimize from intuition. Capture a baseline number, write it down, change one thing, re-measure.
- Record baselines in `docs/` or the run manifest so faculty can see the delta.
- Reuse the existing hooks before writing new ones:
  - `scripts/profile_model_complexity.py` — FLOPs / parameter counts per model variant.
  - `scripts/benchmark_weight_init.py` — init strategy cost.
  - `scripts/detect_overfitting.py` — train/val divergence over epochs.

## 2. Where Time Actually Goes Here
Ranked by expected payoff in this project:

| Suspect | Symptom | Fix |
|---|---|---|
| GeoTIFF decode of 11 bands | GPU idle, CPU pegged, low `nvidia-smi` util | Cache decoded patches (`.npz`/`np.memmap`), read windowed instead of full scene |
| `num_workers=0` IO serialization | Dataloader starves the GPU on Windows | Mandatory setting; compensate with a pre-decoded cache, not more workers |
| Full-scene rasterio reads | Multi-GB memory spikes | `rasterio.windows.Window` + `read(window=...)` |
| FP32 math on a 6 GB laptop GPU | OOM, thrashing | AMP autocast + `channels_last`, batch ≤ 16 |
| Streamlit reruns | UI lag on every click | See `streamlit-performance` |

## 3. CPU Profiling
```powershell
# deterministic function profile
.\.venv\Scripts\python.exe -m cProfile -o prof.out scripts\train_one.py --config configs\x.yaml
.\.venv\Scripts\python.exe -c "import pstats; pstats.Stats('prof.out').sort_stats('cumtime').print_stats(25)"
```
- Sort by `cumtime` for the real bottleneck, then `tottime` to confirm it is self-time.
- `py-spy` for sampling a live process without code changes (optional dependency):
  `py-spy record -o prof.svg --pid <pid>`
- `tracemalloc` for growth in long loops (ingestion, dashboard session, video inference):
  snapshot before/after and diff the top 10 allocations.

## 4. GPU Profiling
- Always `torch.cuda.synchronize()` before starting and stopping a timer; async kernels lie otherwise.
- Use `torch.utils.benchmark.Timer` for kernel-level or module-level timing, not `time.time()` around a launch.
- Memory: `torch.cuda.max_memory_allocated() / 2**20` MB and `reset_peak_memory_stats()` between phases.
- Verify AMP is actually active — check that autocast regions cover the forward+loss, not just the model.
- `torch.cuda.set_per_process_memory_fraction(0.9)` when co-running the dashboard and a trainer.

## 5. Model-Level Cost
- Report parameters, FLOPs, and GPU inference latency together; a smaller model that is slower is not a win.
- Compare variants from `src/models/registry.py` (`unet`, `unet_attention`, `unet_cnn`, `unet_effnet_transformer`, `yolo_vit_lstm`, `yolo_bilstm`, `cnn_lstm`) on the same input shape.
- Fix the input contract when benchmarking: `(1, 11, 256, 256)` for segmentation.

## 6. Reporting a Speedup
State four things or the claim is unusable:
1. Baseline and new value with units (s/epoch, patches/s, MB peak).
2. Hardware and precision mode (RTX 4050 6 GB, AMP on/off).
3. The exact command used.
4. Whether output quality changed (mIoU / Class-1 F1 must not regress).

## 7. Anti-Patterns
- Timing GPU work without `synchronize()`.
- "Optimizing" by shrinking validation or skipping bands — that changes the science.
- Raising `num_workers` on Windows; it deadlocks rather than speeds up.
- Profiling a cached path and reporting it as the uncached cost.
