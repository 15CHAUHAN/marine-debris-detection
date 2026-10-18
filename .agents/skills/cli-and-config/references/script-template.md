# Reference: Full script skeleton

Drop this in `scripts/_template.py` and copy it for new entrypoints.

```python
"""Train a segmentation model.  Docs: configs/README.md"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("train_one")

# ---------------------------------------------------------------- config ----

REQUIRED = {
    "seed": int,
    "model": dict,
    "train": dict,
    "out": str,
}

def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[config] {path} not found")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    for expr in overrides:                      # --set train.lr=0.001
        key, _, raw = expr.partition("=")
        if not _ or not key:
            raise SystemExit(f"[config] bad --set '{expr}' (want key=value)")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)   # keeps ints/floats/bools typed

    for key, typ in REQUIRED.items():
        if key not in cfg:
            raise SystemExit(f"[config] missing required key '{key}'")
        if not isinstance(cfg[key], typ):
            raise SystemExit(f"[config] '{key}' must be {typ.__name__}")
    return cfg

# ------------------------------------------------------------------- args ----

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_one",
        description="Train a segmentation model on prepared MARIDA patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "unet_marida.yaml")
    p.add_argument("--set", action="append", default=[], metavar="K=V",
                   help="override a config leaf, repeatable")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="cap iterations (smoke test)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)

# ------------------------------------------------------------------- seed ----

def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log.info("seed=%d", seed)

# ------------------------------------------------------------------- main ----

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        cfg = load_config(args.config, args.set)
    except SystemExit as exc:
        log.error("%s", exc)
        return 1

    seed = args.seed if args.seed is not None else cfg["seed"]
    out = Path(args.out) if args.out else ROOT / cfg["out"]
    if out.exists() and any(out.iterdir()) and not args.force:
        log.error("output %s already populated; pass --force to overwrite", out)
        return 1

    device = args.device or ("cuda" if _cuda_ok() else "cpu")
    patches = ROOT / "patches"
    if not patches.exists():
        log.error("missing %s - run scripts/prepare_plastic_l2a.py first", patches)
        return 2

    log.info("config=%s out=%s device=%s seed=%d", args.config, out, device, seed)
    if args.dry_run:
        log.info("dry run ok")
        return 0

    set_seed(seed)
    try:
        _run(cfg, out, device, args)
    except RuntimeError as exc:                 # OOM, cudnn failures, ...
        log.error("runtime failure: %s", exc)
        return 3
    return 0

def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

if __name__ == "__main__":
    sys.exit(main())
```

## Config precedence (single rule)

`CLI flag > --set override > config file value > code default`

Implement it by resolving in exactly that order in `main()`, storing the result in
one dict, and passing that dict down. Never let a downstream module re-read the
config file.

## Testable CLI

```python
from scripts.train_one import main

def test_dry_run_ok(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.train_one.ROOT", tmp_path)
    (tmp_path / "patches").mkdir()
    cfg = tmp_path / "c.yaml"
    cfg.write_text("seed: 0\nmodel: {name: unet}\ntrain: {epochs: 1}\nout: artifacts/x\n")
    assert main(["--config", str(cfg), "--dry-run"]) == 0

def test_missing_data_exit_2(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.train_one.ROOT", tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("seed: 0\nmodel: {}\ntrain: {}\nout: artifacts/x\n")
    assert main(["--config", str(cfg)]) == 2
```
