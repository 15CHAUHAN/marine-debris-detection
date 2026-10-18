# Reference: Dockerfile and compose templates

## 1. `.dockerignore` (create BEFORE any build)

Mirrors `.gitignore` so the build context stays a few MB instead of 43 GB.

```dockerignore
.git
.gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/

# datasets - mounted at runtime, never baked in
20210626/
MARIDA/
patches/
taco_raw/
TACO-master/
marine-debris-fls-datasets-master/
utd2.v8i.yolo26/
shapefiles/
data/

# artifacts and weights
artifacts/
runs/
weights/
mlruns/
wandb/
*.pt
*.pth
*.ckpt
*.onnx
*.engine
*.zip
*.npz
*.npy
*.tif
*.tiff
*.mp4

# docs/media not needed in the image
*.docx
*.html
docs/
```

Verify the context size before building:

```powershell
docker build --no-cache -t tech-athon:verify . 2>&1 | Select-String "Sending build context"
```

If that reports hundreds of MB or GB, `.dockerignore` is wrong — fix it first.

## 2. `Dockerfile` - CPU verification image (default)

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /wheels
COPY requirements.txt .
# resolve everything (incl. cu126 torch wheels) into a wheelhouse
RUN pip wheel --wheel-dir /wheels \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      -r requirements.txt

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1

# minimal system libs used by rasterio/PIL/opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libexpat1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links /wheels -r /dev/null || true

COPY requirements.txt .
RUN pip install --no-index --find-links /wheels -r requirements.txt

# source only - datasets come from mounts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY dashboard/ ./dashboard/
COPY splits/ ./splits/
COPY docs/ ./docs/
COPY AGENTS.md CONTEXT.md ./

# datasets and artifacts are expected at runtime
RUN mkdir -p /app/patches /app/artifacts

CMD ["python", "-m", "pytest", "tests", "-q"]
```

Note: `python:3.12-slim` cannot run CUDA kernels. This image is for the CPU-safe
test subset and for the dashboard. For GPU, use `Dockerfile.gpu`.

## 3. `Dockerfile.gpu` - CUDA 12.6 runtime (opt-in)

```dockerfile
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip \
      libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /app
COPY requirements.txt .
RUN pip install --break-system-packages --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      -r requirements.txt

COPY src/ scripts/ tests/ dashboard/ splits/ docs/ ./
COPY AGENTS.md CONTEXT.md ./
RUN mkdir -p /app/patches /app/artifacts
```

Run with `--gpus all`. First prove passthrough works:

```powershell
docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi
```

## 4. `docker-compose.yml`

```yaml
services:
  verify:
    build: { context: ., dockerfile: Dockerfile }
    command: python -m pytest tests -m "not gpu and not slow" -q
    volumes:
      - ./patches:/app/patches:ro
      - ./splits:/app/splits:ro
      - ./artifacts:/app/artifacts

  dashboard:
    build: { context: ., dockerfile: Dockerfile }
    command: >
      streamlit run dashboard/app.py
      --server.address 0.0.0.0 --server.port 8501 --server.headless true
    ports: ["8501:8501"]
    volumes:
      - ./patches:/app/patches:ro
      - ./weights:/app/weights:ro
      - ./artifacts:/app/artifacts
```

## 5. Verify the image is honest

```powershell
# 1. environment matches the pinned contract
docker run --rm tech-athon:verify python -c "import torch, rasterio, streamlit; print(torch.__version__, rasterio.__version__, streamlit.__version__)"
docker run --rm tech-athon:verify python scripts/verify_env.py

# 2. the suite passes with no datasets on the image
docker run --rm tech-athon:verify python -m pytest tests -m "not gpu and not slow" -q

# 3. leakage guards still run
docker run --rm -v D:\TECH-ATHON\splits:/app/splits:ro tech-athon:verify python -m pytest tests/test_splits.py -q

# 4. size sanity (target < 3 GB)
docker images tech-athon:verify
```

## 6. Off-laptop handoff checklist

1. `git archive` a source-only zip (see `git-workflow`) - no datasets inside.
2. Ship `Dockerfile` + `.dockerignore` + `requirements.txt` + the zip.
3. Reviewer runs `docker compose run --rm verify`.
4. Datasets are transferred separately (drive or dataset host) and mounted `:ro`.
5. The run manifest (`run.json`) documents the original GPU/CUDA for reference, so
   "it ran on the RTX 4050" is a recorded fact rather than a claim.
