# Reference: run manifest writer & comparator

## 1. Write an immutable run manifest

```python
"""src/utils/run_manifest.py"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1

def _sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _git_info() -> dict[str, Any]:
    def run(*cmd: str) -> str | None:
        try:
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=5)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None
    return {
        "commit": run("git", "rev-parse", "--short", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }

def _env_info(amp: bool, cudnn_benchmark: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "amp": amp,
        "cudnn_benchmark": cudnn_benchmark,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        pass
    return info

def make_run_id(model: str, dataset: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    sha = _git_info()["commit"] or "nogit"
    return f"{model}_{dataset}_{when:%Y%m%d-%H%M}_{sha}"

def write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    config: dict[str, Any],
    command: str,
    data: dict[str, Any],
    seed: int,
    amp: bool = True,
    cudnn_benchmark: bool = True,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "config": config,
        "command": command,
        "git": _git_info(),
        "environment": _env_info(amp, cudnn_benchmark),
        "data": data,
        "seed": seed,
        "requirements_sha256": _sha256_of(ROOT / "requirements.txt"),
    }
    path = run_dir / "run.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite a run")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "config.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path

def finish_manifest(run_dir: Path, status: str = "completed") -> None:
    path = run_dir / "run.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = status
    data["finished_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
```

## 2. Histogram writer (per-epoch, append-only)

```python
import csv
from pathlib import Path

METRIC_COLUMNS = ["epoch", "train_loss", "val_loss", "miou", "macro_f1", "f1_class_1", "lr", "sec"]

def open_history(run_dir: Path, columns: list[str] = METRIC_COLUMNS):
    fh = (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=columns)
    writer.writeheader()
    return fh, writer

def log_epoch(writer, **values) -> None:
    writer.writerow({k: values.get(k, "") for k in writer.fieldnames})
```

## 3. Comparer

```python
"""scripts/compare_runs.py"""
import json
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts"
COLS = ["run_id", "model", "epochs", "miou", "f1_class_1", "peak_mb", "minutes"]

def load_runs(root: Path = ART) -> list[dict]:
    rows = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        man_f, met_f = run_dir / "run.json", run_dir / "metrics.json"
        if not (man_f.exists() and met_f.exists()):
            continue
        man, met = json.loads(man_f.read_text()), json.loads(met_f.read_text())
        rows.append({
            "run_id": man["run_id"],
            "model": man["config"].get("model", {}).get("name", "?"),
            "epochs": man["config"].get("train", {}).get("epochs"),
            "miou": met.get("miou"),
            "f1_class_1": met.get("f1_class_1"),
            "peak_mb": met.get("peak_mem_mb"),
            "minutes": met.get("minutes"),
            "gan": man["data"].get("gan_augmented", False),
            "splits": tuple(sorted(man["data"]["split_counts"].items())),
        })
    return rows

def main() -> None:
    runs = load_runs()
    if not runs:
        print("no complete runs under artifacts/")
        return
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["splits"], []).append(r)      # only compare like-for-like
    for splits, rows in groups.items():
        print(f"\nsplit counts: {dict(splits)}")
        print(" | ".join(COLS))
        for r in oddsort(rows):
            print(" | ".join(str(r[c]) for c in COLS))

def oddsort(rows):
    return sorted(rows, key=lambda r: -(r["f1_class_1"] or 0))

if __name__ == "__main__":
    main()
```

## 4. Why `f1_class_1` is a stable key

The dashboard (`dashboard/pipeline.py`, `dashboard/report_generator.py`) and the
report script consume metrics programmatically. A stable snake_case key means the
report never has to re-derive the headline metric from a plot or a formatted
string. If a metric name must change, migrate all runs' `metrics.json` and bump
`schema_version`.
