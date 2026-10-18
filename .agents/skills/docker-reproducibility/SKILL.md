---
name: docker-reproducibility
description: Containerize the pinned Python 3.12 + CUDA 12.6 environment for verification off this laptop, mounting datasets read-only instead of copying them.
---

# Docker & Reproducibility

## 1. Goal and Scope
The laptop environment is pinned in `requirements.txt` (Python 3.12, torch
2.14.0+cu126, RTX 4050 6 GB). A container exists to let an evaluator (or another
machine) reproduce **tests and inference** without matching the laptop exactly.

Scope rules:
- The container is for reproducibility and verification, not for squeezing more
  training performance out of the laptop GPU.
- It must never contain the datasets — 36 GB is mounted read-only at runtime.
- Windows + NVIDIA container GPU passthrough is fragile; treat GPU-in-container as
  optional and CPU verification as the guaranteed path.

## 2. Image Strategy
- Base on a CUDA 12.6 runtime image that ships cuDNN, plus Python 3.12.
- Install torch from the CUDA index the same way `requirements.txt` documents:
  `--extra-index-url https://download.pytorch.org/whl/cu126`.
- Multi-stage: builder resolves wheels into a wheelhouse; runtime installs from it
  and contains no compilers.
- Never install the full CUDA toolkit (`cuda`/`devel` images are 6–8 GB) in the
  runtime stage.

## 3. Data and Weights: Mount, Never Copy
```powershell
docker run --rm -v D:\TECH-ATHON\patches:/app/patches:ro `
  -v D:\TECH-ATHON\splits:/app/splits:ro `
  -v D:\TECH-ATHON\artifacts:/app/artifacts `
  tech-athon:verify python -m pytest tests -m "not gpu and not slow" -q
```
- `:ro` for datasets and splits; writable only for `artifacts/` output.
- `.dockerignore` must mirror the `.gitignore` exclusions, or `docker build` will
  try to send 43 GB as build context. See `references/docker-templates.md`.

## 4. Required Files
| File | Purpose |
|---|---|
| `Dockerfile` | CPU verification image (default, always works) |
| `Dockerfile.gpu` | CUDA runtime + torch cu126 variant |
| `.dockerignore` | excludes datasets, artifacts, venv, weights |
| `docker-compose.yml` | optional: dashboard service on 8501 |

Keep the default target CPU-only so a reviewer with no GPU still gets a green test
run; the GPU image is opt-in.

## 5. Entrypoints To Provide
```powershell
# verification suite (no datasets required parts skip themselves)
docker run --rm tech-athon:verify python -m pytest tests -q

# dashboard
docker run --rm -p 8501:8501 -v D:\TECH-ATHON\patches:/app/patches:ro tech-athon:verify `
  streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501

# single-patch inference smoke
docker run --rm -v D:\TECH-ATHON\patches:/app/patches:ro tech-athon:verify `
  python scripts/train_one.py --config configs/unet_marida.yaml --dry-run
```

## 6. Reproducibility Rules
- `requirements.txt` is the single dependency contract; the Dockerfile installs
  **from that file**, never a hand-written pip list that can drift.
- Pin the base image by digest (`@sha256:...`) for a truly frozen environment.
- Set a fixed working directory (`/app`) and copy only tracked source
  (`src`, `scripts`, `tests`, `dashboard`, `docs`, `splits`).
- Enforce determinism flags via env: `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`.
- Do not bake `--gpus all` into a compose file used by reviewers without GPUs.

## 7. Windows GPU Caveats
- GPU passthrough needs Docker Desktop with the WSL2 backend and the
  `nvidia-container-toolkit`; on this laptop it may simply be unavailable.
- Before promising GPU-in-container, verify:
  `docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi`.
- If it fails, do not redesign the project around it — the CPU image plus the local
  venv is the supported path, and the run manifest already records the laptop's
  GPU/CUDA versions for reproducibility.

## 8. Anti-Patterns
- `COPY . .` as the first Dockerfile step (43 GB build context).
- Datasets inside the image.
- A second dependency list that drifts from `requirements.txt`.
- Using `devel`/`cuda` images for the runtime (multi-GB bloat, no benefit).
- `num_workers>0` inside a container on Windows-hosted Docker (same spawn issues as
  the host; see `windows-powershell-env`).
